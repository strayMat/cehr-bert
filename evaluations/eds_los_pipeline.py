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

GRID_RANDOM_SEED = list(range(0, 5))
GRID_PERCENTAGE = [0.02]  # [0.02, 0.1, 0.5, 1]

PARAMETER_GRID = ParameterGrid(
    {
        "random_seed": GRID_RANDOM_SEED,
        "train_percentage": GRID_PERCENTAGE,
    }
)


def main(pipeline_config):
    for run_config in PARAMETER_GRID:
        # set paths
        random_seed_ = run_config["random_seed"]
        pretrain_percentage_ = run_config["train_percentage"]
        # maybe doublon because it is set in both models
        set_seed(random_seed_)
        pretrain_config = create_bert_model_config(pipeline_config)

        # create the effective train set from the full sequence data.
        # Do this with subsetting the existing sequences from the full train data.
        full_sequences = pd.read_parquet(pretrain_config.parquet_data_path)
        if pretrain_percentage_ < 1:
            train, _ = train_test_split(
                np.arange(len(full_sequences)),
                train_size=pretrain_percentage_,
                random_state=random_seed_,
            )
            effective_train_sequences = full_sequences.iloc[train]
        else:
            effective_train_sequences = full_sequences.sample(
                frac=1.0, random_state=random_seed_
            )

        path2effective_train_sequences = Path(
            pretrain_config.parquet_data_path + f"_effective_train"
        )
        if path2effective_train_sequences.exists():
            shutil.rmtree(path2effective_train_sequences, ignore_errors=True)
        effective_train_sequences.to_parquet(path2effective_train_sequences)
        evaluation_pretrain_model_path = str(
            Path(pipeline_config.output_folder) / p.bert_model_validation_path
        )
        if not pipeline_config.skip_pretraining:
            VanillaBertTrainer(
                training_data_parquet_path=str(path2effective_train_sequences),
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
                for f_name in Path(pipeline_config.output_folder).iterdir()
                if f_name.name.find(".h5") != -1
            ]
            folder_list.sort()
            last_pretrain_model_path = str(
                Path(pipeline_config.output_folder) / folder_list[-1]
            )
            if Path(evaluation_pretrain_model_path).exists():
                Path(evaluation_pretrain_model_path).unlink()
            shutil.copyfile(
                last_pretrain_model_path, evaluation_pretrain_model_path
            )
        # Fine tune and evaluate:
        bert_tokenizer_path = os.path.join(
            pipeline_config.output_folder, p.tokenizer_path
        )
        train_dataset = pd.read_parquet(
            pipeline_config.sequence_model_data_path
        )
        effective_train_dataset = train_dataset.loc[
            train_dataset["person_id"].isin(
                effective_train_sequences["person_id"].values
            )
        ].reset_index(drop=True)
        test_dataset = pd.read_parquet(
            pipeline_config.sequence_model_data_path_test
        )
        available_targets_counts = np.unique(
            np.hstack(effective_train_dataset["label"].values)
        )

        logging.getLogger().info(f"Finetuning for 🎯=LOS, 🌱={random_seed_}, {pretrain_percentage_} percents of train")
        bert_model = BertLstmModelEvaluator(
            bert_model_path=evaluation_pretrain_model_path,
            dataset=effective_train_dataset,
            evaluation_folder=pipeline_config.evaluation_folder,
            num_of_folds=1,  # no incidence for our transfer evaluation choice.
            is_transfer_learning=False,
            # this does nothing for train_transfer function, but is given to
            # the metric logger.
            training_percentage=pretrain_percentage_,
            max_seq_length=pipeline_config.max_seq_length,
            batch_size=pipeline_config.evaluation_batch_size,
            epochs=pipeline_config.evaluation_epochs,
            tokenizer_path=bert_tokenizer_path,
            is_temporal=False,
            sequence_model_name=pipeline_config.sequence_model_name
            + f"__target_LOS",
            target_label=None,# only used for multi-classification
            random_seed=random_seed_,
            split_group=pipeline_config.split_group,
        ).train_transfer(test_dataset=test_dataset)


def create_parse_args_pipeline_evaluation():
    pretrain_args = create_parse_args_base_bert()
    pretrain_args.add_argument(
        "-sdt",
        "--sequence_model_data_path_test",
        dest="sequence_model_data_path_test",
        action="store",
        required=True,
    )
    pretrain_args.add_argument(
        "-sd",
        "--sequence_model_data_path",
        dest="sequence_model_data_path",
        action="store",
        required=True,
    )
    pretrain_args.add_argument(
        "-ef",
        "--evaluation_folder",
        dest="evaluation_folder",
        action="store",
        required=True,
    )
    pretrain_args.add_argument(
        "-smn",
        "--sequence_model_name",
        dest="sequence_model_name",
        action="store",
        required=True,
    )
    # for debugging
    pretrain_args.add_argument(
        "-sp",
        "--skip_pretraining",
        dest="skip_pretraining",
        action="store_true",
        required=False,
    )
    pipeline_config = pretrain_args.parse_args()
    # Force the pretrain config to be the same as the one from [cehr_bert
    # README](https://github.com/cumc-dbmi/cehr-bert#3-pre-train-cehr-bert).
    setattr(pipeline_config, "epochs", 2)
    setattr(pipeline_config, "batch_size", 32)
    setattr(pipeline_config, "depth", 5)
    setattr(pipeline_config, "include_visit", True)
    setattr(pipeline_config, "max_seq_length", 512)
    # Force the evaluation config to be the same as the one from [cehr_bert
    # README](https://github.com/cumc-dbmi/cehr-bert#5-fine-tune-cehr-bert-for-hf-readmission).
    # I removed all required argument from the evaluation config.
    setattr(pipeline_config, "action", SEQUENCE_MODEL)
    setattr(pipeline_config, "evaluation_batch_size", 32)
    setattr(pipeline_config, "evaluation_epochs", 10)
    setattr(pipeline_config, "model_evaluators", VANILLA_BERT_LSTM)
    # add split group for train, val as most visited hospital
    setattr(pipeline_config, "split_group", "split_group")
    return pipeline_config


if __name__ == "__main__":
    # setattr(evaluation_config, "sequence_model_data_path_test", )
    # The required arguments are:
    # "-i,--input_folder"
    # "-o,--output_folder"
    # "-sdt, --sequence_model_data_path_test"
    # "-smn ,--sequence_model_name"
    # "-ef, --evaluation_folder"
    # don-t forget to set -ut to True to use time embeddings
    pipeline_config = create_parse_args_pipeline_evaluation()
    main(pipeline_config=pipeline_config)
