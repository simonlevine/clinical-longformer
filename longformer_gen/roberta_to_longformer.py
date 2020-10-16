
from loguru import logger
import os
import math
from dataclasses import dataclass, field
from transformers import RobertaModel, RobertaTokenizer, TextDataset, DataCollatorForLanguageModeling, Trainer
from transformers import TrainingArguments, HfArgumentParser
from transformers.modeling_longformer import LongformerSelfAttention #UNCOMMMENT AND REMOVE AFTER HF>3.02 RELEASES, RERUN

import yaml

import logging
import warnings

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss, MSELoss
from torch.nn import functional as F


with open('params.yaml', 'r') as f:
    params = yaml.safe_load(f.read())


MODEL_OUT_DIR = './custom_models'
LOCAL_ATTN_WINDOW = params['local_attention_window']
GLOBAL_MAX_POS = params['global_attention_window']

def main():
    base_model_name_HF = params['base_model_name']
    base_model_name = base_model_name_HF.split('/')[-1]
    model_path = f'{MODEL_OUT_DIR}/{base_model_name}-{GLOBAL_MAX_POS}-speedfix'

    if not os.path.exists(model_path):
        os.makedirs(model_path)

    logger.info(
        f'Converting roberta-biomed-base into {base_model_name}-{GLOBAL_MAX_POS}')

    model, tokenizer, config = create_long_model(
        model_specified=base_model_name_HF, attention_window=LOCAL_ATTN_WINDOW, max_pos=GLOBAL_MAX_POS)

    model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)
    config.save_pretrained(model_path)



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

class RobertaLongModel(RobertaModel):
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

    model = RobertaModel.from_pretrained(model_specified)
    tokenizer = RobertaTokenizer.from_pretrained(
        model_specified, model_max_length=max_pos)
    config = model.config

    # extend position embeddings
    tokenizer.model_max_length = max_pos
    tokenizer.init_kwargs['model_max_length'] = max_pos
    current_max_pos, embed_size = model.embeddings.position_embeddings.weight.shape
    max_pos += 2  # NOTE: RoBERTa has positions 0,1 reserved, so embedding size is max position + 2
    config.max_position_embeddings = max_pos
    assert max_pos > current_max_pos
    # allocate a larger position embedding matrix
    new_pos_embed = model.embeddings.position_embeddings.weight.new_empty(
        max_pos, embed_size)
    # copy position embeddings over and over to initialize the new position embeddings
    k = 2
    step = current_max_pos - 2
    while k < max_pos - 1:
        new_pos_embed[k:(
            k + step)] = model.embeddings.position_embeddings.weight[2:]
        k += step
    model.embeddings.position_embeddings.weight.data = new_pos_embed

    # replace the `modeling_bert.BertSelfAttention` object with `LongformerSelfAttention`
    config.attention_window = [attention_window] * config.num_hidden_layers
    for i, layer in enumerate(model.encoder.layer):
        longformer_self_attn = LongformerSelfAttention(config, layer_id=i)
        longformer_self_attn.query = layer.attention.self.query
        longformer_self_attn.key = layer.attention.self.key
        longformer_self_attn.value = layer.attention.self.value

        longformer_self_attn.query_global = layer.attention.self.query
        longformer_self_attn.key_global = layer.attention.self.key
        longformer_self_attn.value_global = layer.attention.self.value

        layer.attention.self = longformer_self_attn

    return model, tokenizer, config



if __name__ == "__main__":
    main()
