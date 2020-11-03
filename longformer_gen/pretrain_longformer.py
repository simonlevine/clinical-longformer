from loguru import logger
import os
import math
from dataclasses import dataclass, field
from transformers import RobertaModel, RobertaTokenizer, TextDataset, DataCollatorForLanguageModeling, Trainer
from transformers import TrainingArguments, HfArgumentParser
# from transformers.modeling_longformer import LongformerSelfAttention UNCOMMMENT AND REMOVE AFTER HF>3.02 RELEASES, RERUN

import yaml

import logging
import warnings

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss, MSELoss
from torch.nn import functional as F



def copy_proj_layers(model):
    '''
    Pretraining on Masked Language Modeling (MLM) doesn't update the global projection layers.
    After pretraining, the following function copies `query`, `key`, `value` to their global counterpart projection matrices.
    For more explanation on "local" vs. "global" attention,
    please refer to the documentation [here](https://huggingface.co/transformers/model_doc/longformer.html#longformer-self-attention).
    '''
    for i, layer in enumerate(model.roberta.encoder.layer):
        layer.attention.self.query_global = layer.attention.self.query
        layer.attention.self.key_global = layer.attention.self.key
        layer.attention.self.value_global = layer.attention.self.value
    return model

