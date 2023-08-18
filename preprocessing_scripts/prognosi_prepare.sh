
# Prepare data for pretraining (to OMOP CDM)
medem_dir="$HOME/Matthieu/medical_embeddings_transfer"
cohort_dir="$medem_dir/data/icd10_prognosis__age_min_18__dates_2017_2022__task__prognosis@cim10lvl_1__rs_0__min_prev_0.01"
/export/home/cse210038/.user_conda/miniconda/envs/event2vec-py310/bin/python $medem_dir/scripts/experiences/cehr_bert_prepare_pretrain_dataset.py --cohort_name "prognosis"

# Create pretraining sequences from OMOP CDM (to TF compatible)
## By default spark is looking into the hdfs files, so I have to prepend "file://" to the path
cehr_bert_dir="$HOME/Matthieu/cehr-bert/"
input_dir="file://"$cohort_dir"cehr_bert_train"
output_dir="file://"$cohort_dir"cehr_bert_sequences"
conda activate cehr_bert
PYTHONPATH=./: spark-submit $cehr_bert_dir/spark_apps/generate_training_data.py -i $input_dir -o $output_dir -tc condition_occurrence procedure_occurrence drug_exposure -d 2017-06-01 --is_new_patient_representation -iv 

# Generate the finetuning sequences 
input_dir="file://"$cohort_dir
output_dir=$input_dir"cehr_bert_finetuning_sequences"
train_test_split_folder=$input_dir"/dataset_split.parquet" 

/export/home/cse210038/.user_conda/miniconda/envs/cehr_bert/bin/python $cehr_bert_dir/spark_apps/prediction_cohorts/from_eds_stored_cohort.py -i $input_dir -o $output_dir -s $train_test_split_folder -sg "hospital_split" --index_stay_chapters  
