#!/bin/bash

python3 -u process/ontology_utils.py --dataset aispeech
python3 -u process/dataset_utils.py
python3 -u process/ontology_utils.py --dataset topv2