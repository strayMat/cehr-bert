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

    if args.multi_output_classifier:
        # compute, sort and save prevalences.
        targets, counts = np.unique(
            np.hstack(train_dataset["label"].values), return_counts=True
        )
        target_counts = pd.DataFrame(
            {"target": targets, "prevalence": counts / len(train_dataset)}
        ).sort_values("prevalence", ascending=False)
        target_counts.to_csv(
            args.evaluation_folder + "/prevalences.csv", index=False
        )
        if args.target_label is not None:
            targets_to_run = [args.target_label]
        else:
            targets_to_run = target_counts["target"].values
        for target_ in targets_to_run:
            # Create model and train/transfer
            logging.getLogger().info(f"Finetuning for target: {target_}")
            set_seed(args.random_seed)
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
                sequence_model_name=args.sequence_model_name
                + f"__target_{target_}__rs_{args.random_seed}",
                target_label=target_,
            ).train_transfer(test_dataset=test_dataset)
    else:
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
            target_label=None,
        ).train_transfer(test_dataset=test_dataset)
