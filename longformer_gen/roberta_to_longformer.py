
'''

This script intended for a base Roberta (ie, AllenAI's Biomed Roberta) to 
be converted to a "longformer", with a max-token length of some value > 512.

Includes a speed-fix in the global attention window (see Issues of AllenAI Longformer)

After this script completes, can access the pre-trained model from:

2311015 rows of mimic-iii + cxr are combined 

    # tokenizer = RobertaTokenizerFastFast.from_pretrained(model_path)
    # model = RobertaLongForMaskedLM.from_pretrained(model_path)

Simon Levine-Gottreich, 2020
'''


from loguru import logger
import os
import copy
import math
from dataclasses import dataclass, field
from transformers import RobertaForMaskedLM, RobertaTokenizerFast, DataCollatorForLanguageModeling, Trainer, LineByLineTextDataset
from transformers import TrainingArguments, HfArgumentParser

# from datasets import load_dataset

# from transformers.modeling_longformer import LongformerSelfAttention #CAN UNCOMMENT AND REMOVE AFTER HF>>3.02 RELEASES, RERUN
from self_attn import LongformerSelfAttention


import yaml

import logging
import warnings

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss, MSELoss
from torch.nn import functional as F


# Format: each document should be separated by an empty line
TRAIN_FPATH = 'data/filtered_all_notes_train.txt'
VAL_FPATH = 'data/filtered_all_notes_val.txt'

MODEL_OUT_DIR = './longformer_gen'
LOCAL_ATTN_WINDOW = 512 #params['local_attention_window']
GLOBAL_MAX_POS = 4096 #params['global_attention_window']


FAST_DEV_RUN=True

if FAST_DEV_RUN == True:
    TRAIN_FPATH = VAL_FPATH

def main():


    base_model_name_HF = 'allenai/biomed_roberta_base' #params['base_model_name']
    base_model_name = base_model_name_HF.split('/')[-1]
    model_path = f'{MODEL_OUT_DIR}/bioclinical-longformer' #includes speedfix
    unpretrained_model_path = f'{MODEL_OUT_DIR}/{base_model_name}-{GLOBAL_MAX_POS}' #includes speedfix

    if not os.path.exists(model_path):
        os.makedirs(model_path)

    logger.info(
        f'Converting roberta-biomed-base --> {base_model_name} with global attn. window of {GLOBAL_MAX_POS} tokens.')

    model, tokenizer, config = create_long_model(
        model_specified=base_model_name_HF, attention_window=LOCAL_ATTN_WINDOW, max_pos=GLOBAL_MAX_POS)

    logger.info('Long model, tokenizer, and config created.')

    model.save_pretrained(unpretrained_model_path) #save elongated, not pre-trained model, to the disk.
    tokenizer.save_pretrained(unpretrained_model_path)
    config.save_pretrained(unpretrained_model_path)

    logger.warning('SAVED elongated (but not pretrained) model, tokenizer, and config!')

    os.environ["CUDA_VISIBLE_DEVICES"] = "0" #GPU

    logger.info(f'Pretraining roberta-biomed-{GLOBAL_MAX_POS} ... ')

    model.config.gradient_checkpointing = True #set this to ensure GPU memory constraints are OK.


    if FAST_DEV_RUN == True:
        training_args = TrainingArguments(
            output_dir="./longformer_gen/checkpoints",
            overwrite_output_dir=True,
            max_steps=2,
            warmup_steps= 0,
            logging_steps=1,
            save_steps=1,
            max_grad_norm= 5.0,
            per_device_eval_batch_size=8,
            per_device_train_batch_size=2,
            gradient_accumulation_steps= 32,
            learning_rate = 0.00003,
            adam_epsilon= 1e-6,
            weight_decay= 0.01,
            do_eval= True,
            do_train=True,
            )
    
    else:
        logger.critical(f'Pre-Training {model.num_parameters()}-parameter model. This will take ~ 2-3 days!!!!')

        training_args = TrainingArguments(
        output_dir=f"./longformer_gen/checkpoints/bioclinicaLongformer",
        overwrite_output_dir=True,
        warmup_steps= 500,
        logging_steps=500,
        max_steps = 3000,
        save_steps=500,
        max_grad_norm= 5.0,
        per_device_eval_batch_size=8,
        per_device_train_batch_size=2,
        gradient_accumulation_steps= 32,
        learning_rate = 0.00003,
        adam_epsilon= 1e-6,
        weight_decay= 0.01,
        do_eval= True,
        do_train=True
        )

    pretrain_and_evaluate(training_args, model, tokenizer, eval_only=False, model_path=training_args.output_dir)

    model.save_pretrained(model_path) #save elongated AND pre-trained model, to the disk.
    tokenizer.save_pretrained(model_path)
    model.config.save_pretrained(model_path)

    logger.critical('Final pre-trained model, tokenizer,and config saved!')

class FixedLongformerSelfAttention(LongformerSelfAttention):
    super()


class RobertaLongSelfAttention(FixedLongformerSelfAttention):
    '''
    Inherits above...
    '''
    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=False,
        **kwargs
        ):
        return super().forward(hidden_states, attention_mask=attention_mask, output_attentions=output_attentions,**kwargs)

class RobertaLongModel(RobertaForMaskedLM):
    """RobertaLongForMaskedLM represents the "long" version of the RoBERTa model.
     It replaces BertSelfAttention with RobertaLongSelfAttention, which is 
     a thin wrapper around LongformerSelfAttention."""
    def __init__(self, config):
        super().__init__(config)
        for i, layer in enumerate(self.encoder.layer):
            # replace the `modeling_bert.BertSelfAttention` object with `LongformerSelfAttention`
            layer.attention.self = RobertaLongSelfAttention(config, layer_id=i)

def create_long_model(model_specified, attention_window, max_pos):

    """Starting from the `roberta-base` (or similar) checkpoint, the following function converts it into an instance of `RobertaLong`.
     It makes the following changes:
        1)extend the position embeddings from `512` positions to `max_pos`. In Longformer, we set `max_pos=4096`
        2)initialize the additional position embeddings by copying the embeddings of the first `512` positions.
            This initialization is crucial for the model performance (check table 6 in [the paper](https://arxiv.org/pdf/2004.05150.pdf)
            for performance without this initialization)
        3) replaces `modeling_bert.BertSelfAttention` objects with `modeling_longformer.LongformerSelfAttention` with a attention window size `attention_window`

        The output of this function works for long documents even without pretraining.
        Check tables 6 and 11 in [the paper](https://arxiv.org/pdf/2004.05150.pdf) to get a sense of 
        the expected performance of this model before pretraining."""

    model = RobertaForMaskedLM.from_pretrained(model_specified,gradient_checkpointing=True)
    tokenizer = RobertaTokenizerFast.from_pretrained(
        model_specified, model_max_length=max_pos)
    config = model.config

    # extend position embeddings
    tokenizer.model_max_length = max_pos
    tokenizer.init_kwargs['model_max_length'] = max_pos
    current_max_pos, embed_size = model.roberta.embeddings.position_embeddings.weight.shape
    max_pos += 2  # NOTE: RoBERTa has positions 0,1 reserved, so embedding size is max position + 2
    config.max_position_embeddings = max_pos
    assert max_pos > current_max_pos
    # allocate a larger position embedding matrix
    new_pos_embed = model.roberta.embeddings.position_embeddings.weight.new_empty(
        max_pos, embed_size)
    # copy position embeddings over and over to initialize the new position embeddings
    k = 2
    step = current_max_pos - 2
    while k < max_pos - 1:
        new_pos_embed[k:(
            k + step)] = model.roberta.embeddings.position_embeddings.weight[2:]
        k += step
    model.roberta.embeddings.position_embeddings.weight.data = new_pos_embed
    model.roberta.embeddings.position_ids.data = torch.tensor([i for i in range(max_pos)]).reshape(1, max_pos)
    

    # replace the `modeling_bert.BertSelfAttention` object with `LongformerSelfAttention`
    config.attention_window = [attention_window] * config.num_hidden_layers
    for i, layer in enumerate(model.roberta.encoder.layer):
        longformer_self_attn = LongformerSelfAttention(config, layer_id=i)
        longformer_self_attn.query = copy.deepcopy(layer.attention.self.query)
        longformer_self_attn.key = copy.deepcopy(layer.attention.self.key)
        longformer_self_attn.value = copy.deepcopy(layer.attention.self.value)

        longformer_self_attn.query_global = copy.deepcopy(layer.attention.self.query)
        longformer_self_attn.key_global = copy.deepcopy(layer.attention.self.key)
        longformer_self_attn.value_global = copy.deepcopy(layer.attention.self.value)

        layer.attention.self = longformer_self_attn

    return model, tokenizer, config




def pretrain_and_evaluate(args, model, tokenizer, eval_only, model_path):
    logger.info(f'Loading and tokenizing data is usually slow: {VAL_FPATH}')
    val_dataset = LineByLineTextDataset(tokenizer=tokenizer,
                              file_path=VAL_FPATH,
                              block_size=tokenizer.max_len)

    if eval_only:
        train_dataset = val_dataset
    else:
        logger.info(f'Loading and tokenizing training data is usually slow: {TRAIN_FPATH}')
        train_dataset = LineByLineTextDataset(tokenizer=tokenizer,
                                    file_path=TRAIN_FPATH,
                                    block_size=tokenizer.max_len)

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)
    trainer = Trainer(model=model, args=args, data_collator=data_collator,
                      train_dataset=train_dataset, eval_dataset=val_dataset, prediction_loss_only=True,)

    eval_loss = trainer.evaluate()
    eval_loss = eval_loss['eval_loss']
    logger.info(f'Initial eval bpc: {eval_loss/math.log(2)}')
    
    if not eval_only:
        trainer.train(model_path=model_path)
        trainer.save_model()

        eval_loss = trainer.evaluate()
        eval_loss = eval_loss['eval_loss']
        logger.info(f'Eval bpc after pretraining: {eval_loss/math.log(2)}')





if __name__ == "__main__":
    main()
