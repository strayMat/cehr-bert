from evaluations.evaluation import create_evaluation_args
import spark_apps.parameters as p
from evaluations.model_evaluators import *

if __name__ == "__main__":
    args = create_evaluation_args().parse_args()
    assert hasattr(args, "sequence_model_data_path_test")

    validate_folder(args.vanilla_bert_model_folder)
    bert_tokenizer_path = os.path.join(
        args.vanilla_bert_model_folder, p.tokenizer_path
    )
    bert_model_path = os.path.join(
        args.vanilla_bert_model_folder, p.bert_model_validation_path
    )
    train_dataset = pd.read_parquet(args.sequence_model_data_path)
    test_dataset = pd.read_parquet(args.sequence_model_data_path_test)

    # Create model and train/transfer
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
        sequence_model_name=args.sequence_model_name,
    ).train_transfer(test_dataset=test_dataset)
