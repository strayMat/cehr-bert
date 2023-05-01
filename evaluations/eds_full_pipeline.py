import argparse
import logging
import os
from pathlib import Path
import shutil
import numpy as np
import pandas as pd
import spark_apps.parameters as p

from sklearn.model_selection import ParameterGrid, train_test_split
from config.parse_args import create_parse_args_base_bert
from evaluations.evaluation import (
    SEQUENCE_MODEL,
    VANILLA_BERT_LSTM,
    create_evaluation_args,
)
from evaluations.model_evaluators import BertLstmModelEvaluator
from trainers.train_bert_only import (
    VanillaBertTrainer,
    create_bert_model_config,
)
from utils.model_utils import set_seed

ICD10_CHAPTERS = [
    "21",
    "9",
    "4",
    "18",
    "2",
    "5",
    "6",
    "11",
    "14",
    "10",
    "13",
    "19",
    "3",
    "1",
    "15",
    "12",
    "20",
    "7",
    "22",
    "17",
    "8",
]
ICD10_CHAPTERS = ["2"]

GRID_RANDOM_SEED = list(range(0, 5))

PARAMETER_GRID = ParameterGrid(
    {
        "random_seed": GRID_RANDOM_SEED,
        # "target_label": ["2"],
        "train_percentage": [0.1, 0.25, 0.5, 0.9, 1],
    }
)


def main(pretrain_config, evaluation_config):
    for run_config in PARAMETER_GRID:
        # set paths
        random_seed_ = run_config["random_seed"]
        pretrain_percentage_ = run_config["train_percentage"]
        # maybe doublon because it is set in both models
        set_seed(random_seed_)
        # create the sequence data from the full data.
        # Do this with subsetting the existing sequences from the full train data.
        full_sequences = pd.read_parquet(pretrain_config.parquet_data_path)
        if pretrain_percentage_ < 1:
            train, _ = train_test_split(
                np.arange(len(full_sequences)),
                train_size=pretrain_percentage_,
                random_state=random_seed_,
            )
            train_sequences = full_sequences.iloc[train]
        else:
            train_sequences = full_sequences.sample(
                frac=1.0, random_state=random_seed_
            )

        path2train_sequences = Path(
            pretrain_config.parquet_data_path + f"_effective_train"
        )
        shutil.rmtree(path2train_sequences, ignore_errors=True)
        train_sequences.to_parquet(path2train_sequences)
        VanillaBertTrainer(
            training_data_parquet_path=pretrain_config.parquet_data_path,
            model_path=pretrain_config.model_path,
            tokenizer_path=pretrain_config.tokenizer_path,
            visit_tokenizer_path=pretrain_config.visit_tokenizer_path,
            embedding_size=pretrain_config.concept_embedding_size,
            context_window_size=pretrain_config.max_seq_length,
            depth=pretrain_config.depth,
            num_heads=pretrain_config.num_heads,
            batch_size=pretrain_config.batch_size,
            epochs=pretrain_config.epochs,
            learning_rate=pretrain_config.learning_rate,
            include_visit_prediction=pretrain_config.include_visit_prediction,
            include_prolonged_length_stay=pretrain_config.include_prolonged_length_stay,
            use_time_embedding=pretrain_config.use_time_embedding,
            time_embeddings_size=pretrain_config.time_embeddings_size,
            use_behrt=pretrain_config.use_behrt,
            use_dask=pretrain_config.use_dask,
            tf_board_log_path=pretrain_config.tf_board_log_path,
            random_seed=random_seed_,
        ).train_model()
        # Copy the last epoch pretrained model to bert_model.h5 which is the
        # default name taken by the evaluator.
        folder_list = [
            f_name.name
            for f_name in Path(pretrain_config.model_path).iterdir()
            if f_name.name.find(".h5") != -1
        ]
        folder_list.sort()
        last_model_name = folder_list[-1]
        shutil.copyfile(
            str(Path(pretrain_config.model_path) / last_model_name),
            str(Path(pretrain_config.model_path) / "bert_model.h5"),
        )
        # Fine tune and evaluate:
        bert_tokenizer_path = os.path.join(bert_model_path, p.tokenizer_path)
        bert_model_path = os.path.join(
            bert_model_path, p.bert_model_validation_path
        )
        evaluator_config = create_evaluation_args().parse_args()
        train_dataset = pd.read_parquet(path2train_sequences)
        test_dataset = pd.read_parquet(args.sequence_model_data_path_test)
        available_targets_counts = np.unique(
            np.hstack(train_dataset["label"].values)
        )

        targets_to_run = [
            t_ for t_ in ICD10_CHAPTERS if t_ in available_targets_counts
        ]
        for target_ in targets_to_run:
            logging.getLogger().info(
                f"Finetuning for 🎯={target_}, 🌱={random_seed_}"
            )
            bert_model = BertLstmModelEvaluator(
                dataset=train_dataset,
                evaluation_folder=evaluator_config.evaluation_folder,
                num_of_folds=evaluator_config.num_of_folds,
                is_transfer_learning=False,
                # this does nothing for train_transfer function, but is given to
                # the metric logger.
                training_percentage=pretrain_percentage_,
                max_seq_length=evaluator_config.max_seq_length,
                batch_size=evaluator_config.batch_size,
                epochs=evaluator_config.epochs,
                bert_model_path=bert_model_path,
                tokenizer_path=bert_tokenizer_path,
                is_temporal=False,
                sequence_model_name=evaluator_config.sequence_model_name
                + f"__target_{target_}",
                target_label=target_,
                random_seed=random_seed_,
            ).train_transfer(test_dataset=test_dataset)


if __name__ == "__main__":
    pretrain_config = create_bert_model_config(
        create_parse_args_base_bert().parse_args()
    )
    # Force the pretrain config to be the same as the one from [cehr_bert
    # README](https://github.com/cumc-dbmi/cehr-bert#3-pre-train-cehr-bert).
    setattr(pretrain_config, "epochs", 2)
    setattr(pretrain_config, "batch_size", 32)
    setattr(pretrain_config, "depth", 5)
    setattr(pretrain_config, "include_visit", True)
    setattr(pretrain_config, "max_seq_length", 512)

    evaluation_config = create_evaluation_args().parse_args()
    # setattr(evaluation_config, "sequence_model_data_path_test", )
    setattr(evaluation_config, "action", SEQUENCE_MODEL)
    setattr(evaluation_config, "max_seq_length", 512)
    setattr(evaluation_config, "batch_size", 128)
    setattr(evaluation_config, "epochs", 10)
    setattr(evaluation_config, "model_evaluators", VANILLA_BERT_LSTM)
    main(pretrain_config=pretrain_config, evaluation_config=evaluation_config)
