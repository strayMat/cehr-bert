#!/bin/bash

local_cohort_dir="/export/home/cse210037/Matthieu/medical_embeddings_transfer/data/mace__age_min_18__dates_2018_2020__task__MACE@360__index_visit_random/"


pretrained_dir=$local_cohort_dir"cehr_bert_pretrained_pipeline"
pretrain_sequence=$local_cohort_dir"cehr_bert_sequences"
train_sequence_dir=$local_cohort_dir"cehr_bert_finetuning_sequences_train"
test_sequence_dir=$local_cohort_dir"cehr_bert_finetuning_sequences_external_test"
evaluation_dir=$local_cohort_dir"evaluation_train_val_split"

/export/home/cse210037/.user_conda/miniconda/envs/cehr-bert/bin/python evaluations/eds_mace_pipeline.py -i $pretrain_sequence -o $pretrained_dir -sd $train_sequence_dir -sdt $test_sequence_dir -ef $evaluation_dir -smn CEHR_BERT_512_pipeline -ut -pr -sp
