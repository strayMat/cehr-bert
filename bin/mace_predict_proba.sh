#!/bin/bash

local_cohort_dir="/export/home/cse210037/Matthieu/medical_embeddings_transfer/data/mace__age_min_18__dates_2018_2020__task__MACE@360__index_visit_random/"


pretrained_model_name="bert_model-pretrained__trainsize_1.h5"

/export/home/cse210037/.user_conda/miniconda/envs/cehr-bert/bin/python ~/Matthieu/cehr-bert/evaluations/eds_mace_predict_proba.py -c $local_cohort_dir -p $pretrained_model_name -r 2
