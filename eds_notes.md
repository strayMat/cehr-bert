# Following the README on EDS data

The tag `eds-modified:` allows to see where a modified the cehr_bert code for
our use case on the APHP eds.

## 1. Download OMOP tables as parquet files

I wrote a [script to restrict the database to the train id of a given cohort](https://gitlab.inria.fr/soda/medical_embeddings_transfer/-/blob/main/scripts/experiences/cehr_bert_prepare_train_dataset.py) and copy the in a dedicated folder the domain tables used by cehrt_bert after joining them to a cohort of interest: (procedure_occurrence, condition_occurrence, drug_exposure_administration, person, visit_occurrence)

## 2. Generate training data for CEHR-BERT

Checking what does `/spark_apps/generate_training_data.py::main`
- `preprocess_domain_table`: I deactivated the rollup, so it does nothing but force the colnames to lower
- I force datetime conversion from string since pyspark is super null for this task... 
- Then it produces event tables with `join_domain_tables`, then it joins person and domain tables
- `create_sequence_data_with_att` creates the sequences : I had to make sure the datetime are well converted from string to datetime. It returned empty dataframe.

The command on eds should be 
```console
cohort_dir="file:///export/home/cse210038/Matthieu/medical_embeddings_transfer/data/icd10_prognosis__age_min_18__dates_2017-01-01_2022-06-01__task__prognosis@cim10lvl_1__rs_0__min_prev_0.01/"
input_dir=$cohort_dir"cehr_bert_train"
output_dir=$cohort_dir"cehr_bert_sequences"
PYTHONPATH=./: spark-submit spark_apps/generate_training_data.py -i $input_dir -o $output_dir -tc condition_occurrence procedure_occurrence drug_exposure -d 2017-06-01 --is_new_patient_representation -iv 
```