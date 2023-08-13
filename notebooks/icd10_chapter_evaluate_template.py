import sys
sys.path.append("../")
from sklearn.utils import Bunch
import spark_apps.parameters as p 
from pathlib import Path
from evaluations.model_evaluators import *
import tensorflow as tf
from models.custom_layers import get_custom_objects
from evaluations.model_evaluators import compute_binary_metrics
from utils.model_utils import log_function_decorator, create_folder_if_not_exist

# +
cohort_dir = Path("/export/home/cse210038/Matthieu/medical_embeddings_transfer/medem/../data/icd10_prognosis__age_min_18__dates_2017-01-01_2022-06-01__task__prognosis@cim10lvl_1__rs_0__min_prev_0.01chap_9/")
evaluation_dir = cohort_dir / "evaluation_train_val_split"
finetuned_dir=evaluation_dir/"CEHR_BERT_512/CEHR_BERT_512.h5"
pretrained_dir=cohort_dir/"cehr_bert_pretrained_model"
#train_sequence_dir=cohort_dir/"cehr_bert_finetuning_sequences_train"
test_sequence_dir = cohort_dir/ "cehr_bert_finetuning_sequences_external_test"
args = Bunch(**{
    "finetuned_model_file": str(finetuned_dir),
    "vanilla_bert_model_folder": str(pretrained_dir),
    "evaluation_folder": str(evaluation_dir),
    #"sequence_model_data_path": str(train_sequence_dir),
    "sequence_model_data_path_test": str(test_sequence_dir),
    "training_percentage": 1,
    "num_of_folds": 1,
    "max_seq_length": 512, 
    "batch_size":32,
    "epochs": 10,
    "sequence_model_name":"CEHR_BERT_512", 
})

bert_tokenizer_path = os.path.join(args.vanilla_bert_model_folder,
                                   p.tokenizer_path)
bert_model_path = os.path.join(args.vanilla_bert_model_folder,
                               p.bert_model_validation_path)
#train_dataset = pd.read_parquet(args.sequence_model_data_path)
test_dataset = pd.read_parquet(args.sequence_model_data_path_test)
# -

bert_model = BertLstmModelEvaluator(
    dataset=test_dataset,
    evaluation_folder=args.evaluation_folder,
    num_of_folds=args.num_of_folds,
    is_transfer_learning=False,
    training_percentage=args.training_percentage,
    max_seq_length=args.max_seq_length,
    batch_size=args.batch_size,
    epochs=args.epochs,
    bert_model_path=bert_model_path,
    tokenizer_path=bert_tokenizer_path,
    is_temporal=False,
    sequence_model_name=args.sequence_model_name)

finetuned_model = tf.keras.models.load_model(args.finetuned_model_file,
                                                    custom_objects=dict(**get_custom_objects()))


finetuned_model.inputs

test_inputs, test_labels = bert_model.extract_model_inputs(test_dataset)
test_set = (
            tf.data.Dataset.from_tensor_slices((test_inputs, test_labels))
            .cache()
            .batch(bert_model._batch_size)
        )

compute_binary_metrics(
            finetuned_model, test_set, create_folder_if_not_exist(str(evaluation_dir/"CEHR_BERT_512/"),"manual_test_metric")
        )


