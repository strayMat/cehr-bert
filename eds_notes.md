# Following the README on EDS data

The tag `eds-modified:` allows to see where a modified the cehr_bert code for
our use case on the APHP eds.

## 1. Download OMOP tables as parquet files

I wrote a [script to restrict the database to the train id of a given cohort](https://gitlab.inria.fr/soda/medical_embeddings_transfer/-/blob/main/scripts/experiences/cehr_bert_prepare_train_dataset.py) and copy the in a dedicated folder the domain tables used by cehrt_bert after joining them to a cohort of interest: (procedure_occurrence, condition_occurrence, drug_exposure_administration, person, visit_occurrence)

## 2. Generate training data for CEHR-BERT

Checking what does `/spark_apps/generate_training_data.py::main`
- preprocess_domain_table: I deactivated the rollup, so it does nothing but force the colnames to lower
- Then it produces event tables with `join_domain_tables`, then it joins person and domain tables
- `create_sequence_data_with_att` creates the sequences : TODO: