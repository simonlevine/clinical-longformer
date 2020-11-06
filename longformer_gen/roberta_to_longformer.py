
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

from transformers import LongformerForMaskedLM, LongformerTokenizerFast

# from datasets import load_dataset

from transformers.modeling_longformer import LongformerSelfAttention #CAN UNCOMMENT AND REMOVE AFTER HF>>3.02 RELEASES, RERUN
# from self_attn import LongformerSelfAttention


import yaml

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




# Format: each document should be separated by an empty line
TRAIN_FPATH = 'data/filtered_all_notes_train.txt'
VAL_FPATH = 'data/filtered_all_notes_val.txt'

MODEL_OUT_DIR = './longformer_gen'
LOCAL_ATTN_WINDOW = 512 #params['local_attention_window']
GLOBAL_MAX_POS = 4096 #params['global_attention_window']


FAST_DEV_RUN = True

if FAST_DEV_RUN == True:
    TRAIN_FPATH = VAL_FPATH

def main():


    if FAST_DEV_RUN == True:
        training_args = TrainingArguments(
            output_dir="./longformer_gen/checkpoints",
            overwrite_output_dir=True,
            max_steps=2,
            warmup_steps= 0,
            logging_steps=1,
            save_steps=1,
            max_grad_norm= 5.0,
            per_device_eval_batch_size=2,
            per_device_train_batch_size=1,
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
        per_device_eval_batch_size=2,
        per_device_train_batch_size=1,
        gradient_accumulation_steps= 32,
        learning_rate = 0.00003,
        adam_epsilon= 1e-6,
        weight_decay= 0.01,
        do_eval= True,
        do_train=True
        )



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

    # model, tokenizer = LongformerForMaskedLM.from_pretrained('allenai/longformer-base-4096'), LongformerTokenizerFast.from_pretrained('allenai/longformer-base-4096')

    model.config.gradient_checkpointing = True #set this to ensure GPU memory constraints are OK.


    pretrain_and_evaluate(training_args, model, tokenizer, eval_only=False, model_path=training_args.output_dir)


    # model.save_pretrained(model_path) #save elongated AND pre-trained model, to the disk.
    # tokenizer.save_pretrained(model_path)
    # model.config.save_pretrained(model_path)


    logger.info(f'Copying local projection layers into global projection layers ... ')
    model = copy_proj_layers(model)
    logger.info(f'Saving model to {model_path}')
    model.save_pretrained(model_path)

    logger.critical('Final pre-trained model, tokenizer,and config saved!')


class LineByLineTextDataset(Dataset):
    """
    This will be superseded by a framework-agnostic approach soon.
    """

    def __init__(self, tokenizer, file_path: str, block_size: int):
        # warnings.warn(DEPRECATION_WARNING, FutureWarning)
        assert os.path.isfile(file_path), f"Input file path {file_path} not found"
        # Here, we do not cache the features, operating under the assumption
        # that we will soon use fast multithreaded tokenizers from the
        # `tokenizers` repo everywhere =)
        logger.info(f'Creating features from dataset file at {file_path}')

        with open(file_path, encoding="utf-8") as f:
            lines = [line for line in f.read().splitlines() if (len(line) > 0 and not line.isspace())]

        batch_encoding = tokenizer(lines, add_special_tokens=True, truncation=True, max_length=block_size, padding=True)
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

    return model, tokenizer, config


class CustomSelfAttention(LongformerSelfAttention):
    def forward(
        self, hidden_states, attention_mask=None, is_index_masked=None, is_index_global_attn=None, is_global_attn=None,**kwargs
    ):
        """
        LongformerSelfAttention expects `len(hidden_states)` to be multiple of `attention_window`. Padding to
        `attention_window` happens in LongformerModel.forward to avoid redoing the padding on each layer.
        The `attention_mask` is changed in `BertModel.forward` from 0, 1, 2 to -ve: no attention
              0: local attention
            +ve: global attention
        """
        hidden_states = hidden_states.transpose(0, 1)

        # project hidden states
        query_vectors = self.query(hidden_states)
        key_vectors = self.key(hidden_states)
        value_vectors = self.value(hidden_states)

        seq_len, batch_size, embed_dim = hidden_states.size()
        assert (
            embed_dim == self.embed_dim
        ), f"hidden_states should have embed_dim = {self.embed_dim}, but has {embed_dim}"

        # normalize query
        query_vectors /= math.sqrt(self.head_dim)

        query_vectors = query_vectors.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1)
        key_vectors = key_vectors.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1)

        attn_scores = self._sliding_chunks_query_key_matmul(
            query_vectors, key_vectors, self.one_sided_attn_window_size
        )

        # values to pad for attention probs
        remove_from_windowed_attention_mask = (attention_mask != 0)[:, :, None, None]

        # cast to fp32/fp16 then replace 1's with -inf
        float_mask = remove_from_windowed_attention_mask.type_as(query_vectors).masked_fill(
            remove_from_windowed_attention_mask, -10000.0
        )
        # diagonal mask with zeros everywhere and -inf inplace of padding
        diagonal_mask = self._sliding_chunks_query_key_matmul(
            float_mask.new_ones(size=float_mask.size()), float_mask, self.one_sided_attn_window_size
        )

        # pad local attention probs
        attn_scores += diagonal_mask

        assert list(attn_scores.size()) == [
            batch_size,
            seq_len,
            self.num_heads,
            self.one_sided_attn_window_size * 2 + 1,
        ], f"local_attn_probs should be of size ({batch_size}, {seq_len}, {self.num_heads}, {self.one_sided_attn_window_size * 2 + 1}), but is of size {attn_scores.size()}"

        # compute local attention probs from global attention keys and contact over window dim
        if is_global_attn:
            # compute global attn indices required through out forward fn
            (
                max_num_global_attn_indices,
                is_index_global_attn_nonzero,
                is_local_index_global_attn_nonzero,
                is_local_index_no_global_attn_nonzero,
            ) = self._get_global_attn_indices(is_index_global_attn)
            # calculate global attn probs from global key

            global_key_attn_scores = self._concat_with_global_key_attn_probs(
                query_vectors=query_vectors,
                key_vectors=key_vectors,
                max_num_global_attn_indices=max_num_global_attn_indices,
                is_index_global_attn_nonzero=is_index_global_attn_nonzero,
                is_local_index_global_attn_nonzero=is_local_index_global_attn_nonzero,
                is_local_index_no_global_attn_nonzero=is_local_index_no_global_attn_nonzero,
            )
            # concat to local_attn_probs
            # (batch_size, seq_len, num_heads, extra attention count + 2*window+1)
            attn_scores = torch.cat((global_key_attn_scores, attn_scores), dim=-1)

            # free memory
            del global_key_attn_scores

        local_attn_probs_fp32 = F.softmax(attn_scores, dim=-1, dtype=torch.float32)  # use fp32 for numerical stability
        local_attn_probs = local_attn_probs_fp32.type_as(attn_scores)

        # free memory
        del local_attn_probs_fp32

        # softmax sometimes inserts NaN if all positions are masked, replace them with 0
        local_attn_probs = torch.masked_fill(local_attn_probs, is_index_masked[:, :, None, None], 0.0)

        # apply dropout
        local_attn_probs = F.dropout(local_attn_probs, p=self.dropout, training=self.training)

        value_vectors = value_vectors.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1)

        # compute local attention output with global attention value and add
        if is_global_attn:
            # compute sum of global and local attn
            attn_output = self._compute_attn_output_with_global_indices(
                value_vectors=value_vectors,
                attn_probs=local_attn_probs,
                max_num_global_attn_indices=max_num_global_attn_indices,
                is_index_global_attn_nonzero=is_index_global_attn_nonzero,
                is_local_index_global_attn_nonzero=is_local_index_global_attn_nonzero,
            )
        else:
            # compute local attn only
            attn_output = self._sliding_chunks_matmul_attn_probs_value(
                local_attn_probs, value_vectors, self.one_sided_attn_window_size
            )

        assert attn_output.size() == (batch_size, seq_len, self.num_heads, self.head_dim), "Unexpected size"
        attn_output = attn_output.transpose(0, 1).reshape(seq_len, batch_size, embed_dim).contiguous()

        # compute value for global attention and overwrite to attention output
        # TODO: remove the redundant computation
        if is_global_attn:
            global_attn_output, global_attn_probs = self._compute_global_attn_output_from_hidden(
                hidden_states=hidden_states,
                max_num_global_attn_indices=max_num_global_attn_indices,
                is_local_index_global_attn_nonzero=is_local_index_global_attn_nonzero,
                is_index_global_attn_nonzero=is_index_global_attn_nonzero,
                is_local_index_no_global_attn_nonzero=is_local_index_no_global_attn_nonzero,
                is_index_masked=is_index_masked,
            )

            # get only non zero global attn output
            nonzero_global_attn_output = global_attn_output[
                is_local_index_global_attn_nonzero[0], :, is_local_index_global_attn_nonzero[1]
            ]

            # overwrite values with global attention
            attn_output[is_index_global_attn_nonzero[::-1]] = nonzero_global_attn_output.view(
                len(is_local_index_global_attn_nonzero[0]), -1
            )
            # The attention weights for tokens with global attention are
            # just filler values, they were never used to compute the output.
            # Fill with 0 now, the correct values are in 'global_attn_probs'.
            local_attn_probs[is_index_global_attn_nonzero] = 0

        outputs = (attn_output.transpose(0, 1), local_attn_probs)

        return outputs + (global_attn_probs,) if is_global_attn else outputs
    

def pretrain_and_evaluate(training_args, model, tokenizer, eval_only, model_path):
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
                                    block_size= tokenizer.max_len)

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)
    trainer = Trainer(model=model, args=training_args, data_collator=data_collator,
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


# class RobertaLongSelfAttention(LongformerSelfAttention):
#     '''
#     Inherits above...
#     '''
#     def forward(
#         self,
#         hidden_states,
#         attention_mask=None,
#         head_mask=None,
#         encoder_hidden_states=None,
#         encoder_attention_mask=None,
#         output_attentions=False,
#         ):
#         return super().forward(hidden_states, attention_mask=attention_mask, output_attentions=output_attentions)

# class RobertaLongForMaskedLM(RobertaForMaskedLM):
#     """RobertaLongForMaskedLM represents the "long" version of the RoBERTa model.
#      It replaces BertSelfAttention with RobertaLongSelfAttention, which is 
#      a thin wrapper around LongformerSelfAttention."""
#     def __init__(self, config):
#         super().__init__(config)
#         for i, layer in enumerate(self.encoder.layer):
#             # replace the `modeling_bert.BertSelfAttention` object with `RobertaLongSelfAttention`
#             layer.attention.self = RobertaLongSelfAttention(config, layer_id=i)



if __name__ == "__main__":
    main()
