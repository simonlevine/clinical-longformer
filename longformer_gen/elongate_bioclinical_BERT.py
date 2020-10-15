import logging
import os
import math
from dataclasses import dataclass, field
from transformers import BertModel, BertTokenizer, TextDataset, DataCollatorForLanguageModeling, Trainer
from transformers import TrainingArguments, HfArgumentParser
from transformers.modeling_longformer import LongformerSelfAttention
import yaml

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

with open('params.yaml', 'r') as f:
    params = yaml.safe_load(f.read())

MODEL_OUT_DIR = 'custom_models'
LOCAL_ATTN_WINDOW = params['local_attention_window']
GLOBAL_MAX_POS = params['global_attention_window']

def main():
    base_model_name_HF = 'emilyalsentzer/Bio_ClinicalBERT'
    base_model_name = 'Bio_ClinicalBERT'
    model_path = f'{MODEL_OUT_DIR}/{base_model_name}-long'
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    logger.info(
        f'Converting {base_model_name} into {base_model_name}-long')

    model, tokenizer, config = create_long_model(
        model_specified=base_model_name_HF, attention_window=LOCAL_ATTN_WINDOW, max_pos=GLOBAL_MAX_POS)

    model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)
    config.save_pretrained(model_path)


class BertLongSelfAttention(LongformerSelfAttention):
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


class BertLongModel(BertModel):
    def __init__(self, config):
        super().__init__(config)
        for i, layer in enumerate(self.encoder.layer):
            # replace the `modeling_bert.BertSelfAttention` object with `LongformerSelfAttention`
            layer.attention.self = BertLongSelfAttention(config, layer_id=i)


import tensorflow as tf
def create_long_model(save_model_to, attention_window, max_pos, model_path_args):
    model = BertForQuestionAnswering.from_pretrained(model_path_args)
    tokenizer = AutoTokenizer.from_pretrained(model_path_args)
    config = model.config
    print(max_pos)
    # extend position embeddings
    tokenizer.model_max_length = max_pos
    tokenizer.init_kwargs['model_max_length'] = max_pos
    current_max_pos, embed_size = model.bert.embeddings.position_embeddings.weight.shape
    config.max_position_embeddings = max_pos
    assert max_pos > current_max_pos
    # allocate a larger position embedding matrix
    new_pos_embed = model.bert.embeddings.position_embeddings.weight.new_empty(max_pos, embed_size)
    print(new_pos_embed.shape)
    print(model.bert.embeddings.position_embeddings)
    # copy position embeddings over and over to initialize the new position embeddings
    k = 0
    step = current_max_pos
    while k < max_pos - 1:
        new_pos_embed[k:(k + step)] = model.bert.embeddings.position_embeddings.weight
        k += step
    print(new_pos_embed.shape)
    model.bert.embeddings.position_ids = torch.from_numpy(tf.range(new_pos_embed.shape[0], dtype=tf.int32).numpy()[tf.newaxis, :])
    model.bert.embeddings.position_embeddings = torch.nn.Embedding.from_pretrained(new_pos_embed)
    
    # replace the `modeling_bert.BertSelfAttention` object with `LongformerSelfAttention`
    config.attention_window = [attention_window] * config.num_hidden_layers
    for i, layer in enumerate(model.bert.encoder.layer):
        longformer_self_attn = LongformerSelfAttention(config, layer_id=i)
        longformer_self_attn.query = layer.attention.self.query
        longformer_self_attn.key = layer.attention.self.key
        longformer_self_attn.value = layer.attention.self.value

        longformer_self_attn.query_global = layer.attention.self.query
        longformer_self_attn.key_global = layer.attention.self.key
        longformer_self_attn.value_global = layer.attention.self.value

        layer.attention.self = longformer_self_attn
    print(model.bert.embeddings.position_ids.shape)
    logger.info(f'saving model to {save_model_to}')
    model.save_pretrained(save_model_to)
    tokenizer.save_pretrained(save_model_to)
    return model, tokenizer, new_pos_embed

if __name__ == "__main__":
    main()
