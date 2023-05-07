import sys
sys.path.append("../")
from sklearn.utils import Bunch
import spark_apps.parameters as p 
from pathlib import Path
from evaluations.model_evaluators import *

# +
cohort_dir = Path("/export/home/cse210038/Matthieu/medical_embeddings_transfer/medem/../data/icd10_prognosis__age_min_18__dates_2017-01-01_2022-06-01__task__prognosis@cim10lvl_1__rs_0__min_prev_0.01chap_9/")
evaluation_dir = cohort_dir / "evaluation_train_val_split"
pretrained_dir=cohort_dir/"cehr_bert_pretrained_model"
train_sequence_dir=cohort_dir/"cehr_bert_finetuning_sequences_train"
test_sequence_dir = cohort_dir/ "cehr_bert_finetuning_sequences_external_test"
args = Bunch(**{
    "vanilla_bert_model_folder": str(pretrained_dir),
    "evaluation_folder": str(evaluation_dir),
    "sequence_model_data_path": str(train_sequence_dir),
    "sequence_model_data_path_test": str(test_sequence_dir),
    "training_percentage": 0.1,
    "num_of_folds": 1,
    "max_seq_length": 512, 
    "batch_size":32,
    "epochs": 10,
    "sequence_model_name":"CEHR_BERT_512", 
    })


validate_folder(args.vanilla_bert_model_folder)
bert_tokenizer_path = os.path.join(args.vanilla_bert_model_folder,
                                   p.tokenizer_path)
bert_model_path = os.path.join(args.vanilla_bert_model_folder,
                               p.bert_model_validation_path)
train_dataset = pd.read_parquet(args.sequence_model_data_path)
test_dataset = pd.read_parquet(args.sequence_model_data_path_test)
# -

bert_model = BertLstmModelEvaluator(
    dataset=train_dataset,
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

inputs, labels = bert_model.extract_model_inputs() 

from sklearn.model_selection import KFold, train_test_split
train, val = train_test_split(np.arange(len(labels)))

training_input = {k: v[train] for k, v in inputs.items()}
val_input = {k: v[val] for k, v in inputs.items()}

test_inputs, test_labels = bert_model.extract_model_inputs(test_dataset)


