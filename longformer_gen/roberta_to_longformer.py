
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
from transformers import RobertaForMaskedLM, RobertaTokenizerFast, DataCollatorForLanguageModeling, Trainer
from transformers import TrainingArguments, HfArgumentParser

from torch.utils.data import Dataset

# from transformers import LongformerForMaskedLM, LongformerTokenizerFast
from transformers.modeling_longformer import LongformerSelfAttention

import logging
import warnings

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss, MSELoss
from torch.nn import functional as F


import json
import os
import pickle
import random
import time
import warnings
from typing import Dict, List, Optional

import torch
from torch.utils.data.dataset import Dataset

from filelock import FileLock
import pandas as pd


# Format: each document should be separated by an empty line
TRAIN_FPATH = 'data/filtered_all_notes_train.txt'
VAL_FPATH = 'data/filtered_all_notes_val.txt'
SAMPLE_FPATH = 'data/filtered_all_notes_SAMPLE.txt'

MODEL_OUT_DIR = './longformer_gen'
LOCAL_ATTN_WINDOW = 512 #params['local_attention_window']
GLOBAL_MAX_POS = 4096 #params['global_attention_window']


FAST_DEV_RUN = True

if FAST_DEV_RUN == True:

    pd.read_csv(VAL_FPATH,sep='\t', header=None).sample(100).to_csv(SAMPLE_FPATH,header=None,index=None,sep='\t')

    TRAIN_FPATH = SAMPLE_FPATH
    VAL_FPATH = SAMPLE_FPATH

def main():


    if FAST_DEV_RUN == True:
        training_args = TrainingArguments(
            output_dir="./longformer_gen/checkpoints",
            overwrite_output_dir=True,
            max_steps=2,
            warmup_steps= 0, #-->3000
            logging_steps=1,
            save_steps=1,
            max_grad_norm= 5.0,
            per_device_eval_batch_size=2,
            per_device_train_batch_size=2,
            gradient_accumulation_steps= 32,
            learning_rate = 0.00003,
            adam_epsilon= 1e-6,
            weight_decay= 0.01,
            do_eval= True,
            do_train=True,
            )
    
    elif FAST_DEV_RUN == False:

        training_args = TrainingArguments(
        output_dir=f"./longformer_gen/checkpoints/bioclinicaLongformer",
        overwrite_output_dir=True,
        warmup_steps= 500,
        logging_steps=500,
        max_steps = 3000,
        save_steps=500,
        max_grad_norm= 5.0,
        per_device_eval_batch_size=2,
        per_device_train_batch_size=8,
        gradient_accumulation_steps= 32,
        learning_rate = 0.00003,
        adam_epsilon= 1e-6,
        weight_decay= 0.01,
        do_eval= True,
        do_train=True,
        # n_gp
        )

    base_model_name_HF = 'allenai/biomed_roberta_base' #params['base_model_name']

    base_model_name = base_model_name_HF.split('/')[-1]
    model_path = f'{MODEL_OUT_DIR}/bioclinical-longformer' #includes speedfix
    unpretrained_model_path = f'{MODEL_OUT_DIR}/{base_model_name}-{GLOBAL_MAX_POS}' #includes speedfix

    if not os.path.exists(model_path):
        os.makedirs(model_path)

    logger.info(
        f'Converting roberta-biomed-base --> {base_model_name} with global attn. window of {GLOBAL_MAX_POS} tokens.')

    model, tokenizer = create_long_model(
        model_specified=base_model_name_HF, attention_window=LOCAL_ATTN_WINDOW, max_pos=GLOBAL_MAX_POS, save_model_to=unpretrained_model_path)


    logger.critical(f'Long model, tokenizer created, saved to disk at {unpretrained_model_path}.')


    os.environ["CUDA_VISIBLE_DEVICES"] = "0" #GPU

    logger.info(f'Pretraining roberta-biomed-{GLOBAL_MAX_POS} ... ')

    # model, tokenizer = LongformerForMaskedLM.from_pretrained('allenai/longformer-base-4096'), LongformerTokenizerFast.from_pretrained('allenai/longformer-base-4096')

    logger.info(f'Loading the model from {unpretrained_model_path}')
    tokenizer = RobertaTokenizerFast.from_pretrained(unpretrained_model_path,model_max_length=GLOBAL_MAX_POS)
    model = RobertaLongForMaskedLM.from_pretrained(unpretrained_model_path)

    logger.warning(f'Tokenizer {tokenizer} parameterized with model_max_len as {tokenizer.model_max_length}')

    # model.config.gradient_checkpointing = True #set this to ensure GPU memory constraints are OK.

    logger.critical(f'Pre-Training {model.num_parameters()}-parameter model. This could take ~ 2-3 days!!!!')
    pretrain_and_evaluate(training_args, model, tokenizer, eval_only=False, model_path_out=training_args.output_dir)

    logger.warning(f'Copying local projection layers into global projection layers ... ')
    model = copy_proj_layers(model)
    logger.warning(f'Saving model to {model_path}/final')
    model.save_pretrained(f'{model_path}/final') # --> "./longformer_gen/bioclinicaLongformer/final"
    logger.critical('Final pre-trained model, tokenizer,and config saved!')



def pretrain_and_evaluate(training_args, model, tokenizer, eval_only, model_path_out):
    logger.info(f'Loading and tokenizing data is usually slow: {VAL_FPATH}')

    val_dataset = LineByLineTextDataset(tokenizer=tokenizer,
                              file_path=VAL_FPATH,
                              block_size=GLOBAL_MAX_POS)


    if eval_only:
        train_dataset = val_dataset
    else:
        logger.info(f'Loading and tokenizing training data is usually slow: {TRAIN_FPATH}')
        train_dataset = LineByLineTextDataset(tokenizer=tokenizer,
                                    file_path=TRAIN_FPATH,
                                    block_size= GLOBAL_MAX_POS)


    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)

    trainer = Trainer(model=model, args=training_args, data_collator=data_collator,
                      train_dataset=train_dataset, eval_dataset=val_dataset, prediction_loss_only=True)

    eval_loss = trainer.evaluate()
    eval_loss = eval_loss['eval_loss']
    logger.info(f'Initial eval bpc: {eval_loss/math.log(2)}')
    
    if not eval_only:
        trainer.train(model_path=model_path_out)
        trainer.save_model()

        eval_loss = trainer.evaluate()
        eval_loss = eval_loss['eval_loss']
        logger.info(f'Eval bpc after pretraining: {eval_loss/math.log(2)}')



class LineByLineTextDataset(Dataset):
    """
    This will be superseded by a framework-agnostic approach soon.
    """

    def __init__(self, tokenizer, file_path: str, block_size: int):


        logger.warning(f'Block size in dataset set as {block_size}')
        # warnings.warn(DEPRECATION_WARNING, FutureWarning)
        assert os.path.isfile(file_path), f"Input file path {file_path} not found"
        # Here, we do not cache the features, operating under the assumption
        # that we will soon use fast multithreaded tokenizers from the
        # `tokenizers` repo everywhere =)
        logger.info(f'Creating features from dataset file at {file_path}')

        with open(file_path, encoding="utf-8") as f:
            lines = [line for line in f.read().splitlines() if (len(line) > 10 and not line.isspace())]

        batch_encoding = tokenizer(lines, add_special_tokens=True, truncation=True, max_length=block_size, padding='max_length')
        
        self.examples = batch_encoding["input_ids"]
        self.examples = [{"input_ids": torch.tensor(e, dtype=torch.long)} for e in self.examples]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i) -> Dict[str, torch.tensor]:
        return self.examples[i]


def copy_proj_layers(model):
    for i, layer in enumerate(model.roberta.encoder.layer):
        layer.attention.self.query_global = layer.attention.self.query
        layer.attention.self.key_global = layer.attention.self.key
        layer.attention.self.value_global = layer.attention.self.value
    return model



def create_long_model(model_specified, attention_window, max_pos, save_model_to):

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

    model = RobertaForMaskedLM.from_pretrained(model_specified) #,gradient_checkpointing=True)

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
    model.roberta.embeddings.position_embeddings.num_embeddings = len(new_pos_embed.data)
    
    # # first, check that model.roberta.embeddings.position_embeddings.weight.data.shape is correct — has to be 4096 (default) of your desired length
    # model.roberta.embeddings.position_ids = torch.arange(
    #     0, model.roberta.embeddings.position_embeddings.num_embeddings
    # )[None]

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

    logger.info(f'saving model to {save_model_to}')
    model.save_pretrained(save_model_to)
    tokenizer.save_pretrained(save_model_to)

    return model, tokenizer


class RobertaLongSelfAttention(LongformerSelfAttention):
    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=False,
    ):
        return super().forward(hidden_states, attention_mask=attention_mask, output_attentions=output_attentions)


class RobertaLongForMaskedLM(RobertaForMaskedLM):
    def __init__(self, config):
        super().__init__(config)
        for i, layer in enumerate(self.roberta.encoder.layer):
            # replace the `modeling_bert.BertSelfAttention` object with `LongformerSelfAttention`
            layer.attention.self = RobertaLongSelfAttention(config, layer_id=i)

if __name__ == "__main__":
    main()
