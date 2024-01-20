### 1. Convert APHP OMOP format to CEHR-BERT fully compatible OMOP tables as parquet files

The format between APHP and CEHR-BERT OMOP are not exactly the same. 

Goal: Provide OMOP tables as parquet files that can be ingested by CEHR-BERT
preprocessing to be transformed into sequences appropriate for the transformer
model. 

I wrote a [script to restrict the database to the train id of a given cohort](https://gitlab.inria.fr/soda/medical_embeddings_transfer/-/blob/main/scripts/experiences/cehr_bert_prepare_pretrain_dataset.py) and copy in a dedicated folder the domain tables used by cehrt_bert after joining them to a cohort of interest: (procedure_occurrence, condition_occurrence, drug_exposure_administration, person, visit_occurrence). It uses polars and some codes I am using for the other experiments.

WARNING: Incompatibility between Polars parquet files and Spark parquet files: I had to rewrite the files with Pandas so that Spark could read them. This seems to be due to different datetimes between the two libraries.

### 2. Generate training data for CEHR-BERT: create sequences from OMOP tables

```console
cohort_dir="file:///export/home/cse210037/Matthieu/medical_embeddings_transfer/data/mace__age_min_18__dates_2018_2020__task__MACE@360__index_visit_random/"
input_dir=$cohort_dir"cehr_bert_train"
output_dir=$cohort_dir"cehr_bert_sequences"
PYTHONPATH=./: spark-submit --driver-memory=12g --executor-memory 16g spark_apps/generate_training_data.py -i $input_dir -o $output_dir -tc condition_occurrence procedure_occurrence drug_exposure -d 2017-06-01 --is_new_patient_representation -iv 
```


### 3. Pre-train Cehr-bert (skipped for MACE since, we only pretrain on task data)


```console
cohort_dir="/export/home/cse210037/Matthieu/medical_embeddings_transfer/data/mace__age_min_18__dates_2018_2020__task__MACE@360__index_visit_random/"
input_dir=$cohort_dir"cehr_bert_sequences"
output_dir=$cohort_dir"cehr_bert_pretrained_model"
mkdir -p $output_dir

/export/home/cse210037/.user_conda/miniconda/envs/cehr_bert/bin/python trainers/train_bert_only.py -i $input_dir -o $output_dir -iv -m 512 -e 2 -b 32 -d 5 --use_time_embedding
```

### 4. Generate cehr-bert compatible data for the prediction task

```console
input_dir="file:///export/home/cse210037/Matthieu/medical_embeddings_transfer/data/mace__age_min_18__dates_2018_2020__task__MACE@360__index_visit_random/"
output_dir=$input_dir"cehr_bert_finetuning_sequences"
train_test_split_folder=$input_dir"/hospital_split.parquet" 

PYTHONPATH=./: spark-submit --driver-memory=12g --executor-memory 16g spark_apps/prediction_cohorts/from_eds_stored_cohort.py -i $input_dir -o $output_dir -s $train_test_split_folder -sg "most_visited_hospital"
```

