#!/usr/bin/env bash

python ./preprocessing_pipeline/0_format_notes.py && python ./preprocessing_pipeline/1_format_data_for_training.py && python ./preprocessing_pipeline/2_formatted_to_transformer.py