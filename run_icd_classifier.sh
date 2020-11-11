#!/usr/bin/env bash

python classifier_pipeline/training_onelabel.py \
    --transformer_type roberta-long \
    --encoder_model simonlevine/bioclinical-roberta-long \
    --batch_size 2 \
    --gpus 1 \
    --fast_dev_run True