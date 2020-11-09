#!/usr/bin/env bash

python classifier_pipeline/training_onelabel.py \
    --transformer_type bert \
    --encoder_model emilyalsentzer/Bio_ClinicalBERT \
    --gpus 1 \
    --fast_dev_run True \