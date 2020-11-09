#!/usr/bin/env bash

python classifier_pipeline/training_onelabel.py \
    --transformer_type bert \
    --encoder_model bert-base-uncased

python classifier_pipeline/training_onelabel.py \
    --transformer_type bert \
    --encoder_model emilyalsentzer/Bio_ClinicalBERT

python classifier_pipeline/training_onelabel.py \
    --transformer_type roberta \
    --encoder_model allenai/biomed_roberta_base


python classifier_pipeline/training_onelabel.py
    --transformer_type roberta-long \
    --encoder_model simonlevine/biomed_roberta_base-4096

python classifier_pipeline/training_onelabel.py \
    --transformer_type roberta-long \
    --encoder_model simonlevine/bioclinical-longformer