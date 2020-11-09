#!/usr/bin/env bash

python classifier_pipeline/training_onelabel.py \
    --transformer_type bert \
    --encoder_model bert-base-uncased \
    --fast_dev_run True \
    --gpus 0

python classifier_pipeline/training_onelabel.py \
    --transformer_type bert \
    --encoder_model emilyalsentzer/Bio_ClinicalBERT \
    --fast_dev_run True \
    --gpus 0 \
python classifier_pipeline/training_onelabel.py \
    --transformer_type roberta \
    --encoder_model allenai/biomed_roberta_base \
    --fast_dev_run True \
    --gpus 0 \


python classifier_pipeline/training_onelabel.py
    --transformer_type roberta-long \
    --encoder_model simonlevine/biomed_roberta_base-4096 \
    --fast_dev_run True \
    --gpus 0 \

python classifier_pipeline/training_onelabel.py \
    --transformer_type roberta-long \
    --encoder_model simonlevine/bioclinical-longformer \
    --fast_dev_run True \
    --gpus 0 \