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
- `create_sequence_data_with_att` creates the sequences : 
    - I had to make sure the datetime are well converted from string to datetime. 
    - I had to change the columns of interest for getting the good codes (`utils/spark_utils.py::DOMAIN_KEY_FIELDS`).

The command on eds should be 
```console
cohort_dir="file:///export/home/cse210038/Matthieu/medical_embeddings_transfer/data/icd10_prognosis__age_min_18__dates_2017-01-01_2022-06-01__task__prognosis@cim10lvl_1__rs_0__min_prev_0.01/"
input_dir=$cohort_dir"cehr_bert_train"
output_dir=$cohort_dir"cehr_bert_sequences"
PYTHONPATH=./: spark-submit spark_apps/generate_training_data.py -i $input_dir -o $output_dir -tc condition_occurrence procedure_occurrence drug_exposure -d 2017-06-01 --is_new_patient_representation -iv 
```

## 3. Pre-train CEHR-BERT

NB: Don't forget to create the `output_dir` folder. 

```console
cohort_dir="/export/home/cse210038/Matthieu/medical_embeddings_transfer/data/icd10_prognosis__age_min_18__dates_2017-01-01_2022-06-01__task__prognosis@cim10lvl_1__rs_0__min_prev_0.01/"
input_dir=$cohort_dir"cehr_bert_sequences"
output_dir=$cohort_dir"cehr_bert_pretrained"

/export/home/cse210038/.user_conda/miniconda/envs/cehr_bert/bin/python trainers/train_bert_only.py -i $input_dir -o $output_dir -iv -m 512 -e 2 -b 32 -d 5 --use_time_embedding
```

CPU: should work as is.
GPU: Waiting EDS access

## 4. Generate next visit icd10 chapter prediction task

### Use the medem code to create an event cohort

- you should obtain a folder with two dataframes:
 - a person : static informations (sex, ) and task specific information (followup_start, y).
 - an event dataframe: events only present in the observation period for the predictive task
- use the dedicated functions for the eds: `spark_apps.prediction_cohorts.from_eds_stored_cohort.py::create_cohort_from_eds_eventCohort`. The command should looks like: 

```console

```

### Matthieu's note

The code of cert-behrt for cohort creation is difficult to read: too many
arguments, mix between sql and spark.
Q: 
- What is the `cohort_member_id` in `spark_apps/cohorts/spark_app_base.py::NestedCohortBuilder.cohort_member_id`, is it important ? It is a rank over person_id, index_date and visit_occurrence_id. In my case, it can be the person_id, since I am only considering one visit per person. 
- Why not including visit types in the fine tuning task (option -iv) ? Maybe because it only
  serves as a pretraining task in addition to the MLM task.

- In the hf_readmission samples that they give, there is the following columns in addition to the patient_sequence used for pretraining: `['gender_concept_id', 'race_concept_id', 'index_date', 'visit_occurrence_id', 'label', 'age']`. These are all statics that I can populate, then join to the sequences using the `utils.spark_utils.py::create_sequence_data_with_att`. 


### 5. Fine-tune CEHR-BERT for hf readmission
