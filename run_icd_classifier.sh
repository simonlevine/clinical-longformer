#!/usr/bin/env bash

python classifier_pipeline/training_onelabel.py \
    --transformer_type roberta-long \
    --encoder_model simonlevine/biomed_roberta_base-4096-speedfix \
    --fast_dev_run True \
    --gpus 1 \
