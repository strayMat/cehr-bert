import argparse
from evaluations.model_evaluators import BertLstmModelEvaluator
from pathlib import Path
import spark_apps.parameters as p
import pandas as pd

"""
For the MACE task, I wonder if the model has an overall good calibration but a poor calibration for higher risk patients. 

To test that, I need to compute the calibration curve for one of my best transformer model (ie. trained on all data).

This script focus on : 
- loading a pretrained models, 
- fine-tuning on the MACE task,
- computing the probabilities for the test set.
"""

def main(config: argparse.Namespace):
    path2cohort = Path(config.path2cohort)
    pretrained_model_name = config.pretrained_model_name
    random_seed = config.random_seed


    evaluation_dir = path2cohort/"evaluation_train_val_split"
    pretrained_sequence_dir = path2cohort / "cehr_bert_pretrained_pipeline"
    pretrained_path = pretrained_sequence_dir / pretrained_model_name
    ## Not used since the checkpoint of finetuned model is not loadable due to two layers with the same name (age).
    #finetuned_path = path2cohort / "evaluation_train_val_split/CEHR_BERT_512_pipeline__target_MACE/CEHR_BERT_512_pipeline__target_MACE.h5"

    # Load sequences
    train_dataset = pd.read_parquet(path2cohort / "cehr_bert_finetuning_sequences_train").sample(
                    frac=0.05, random_state=random_seed
                ).reset_index()
    test_dataset = pd.read_parquet(path2cohort/"cehr_bert_finetuning_sequences_external_test")

    # instantiation of the model with the Cehr-bert library
    bert_model = BertLstmModelEvaluator(
                bert_model_path=str(pretrained_path),
                dataset=train_dataset, 
                evaluation_folder=str(evaluation_dir),
                num_of_folds=1,  # no incidence for our transfer evaluation choice.
                is_transfer_learning=False,
                # this does nothing for train_transfer function, but is given to
                # the metric logger.
                training_data_ratio=1, # no effect for transfer
                max_seq_length=512,
                batch_size=32,
                epochs=1,
                tokenizer_path=str(pretrained_sequence_dir/p.tokenizer_path),
                is_temporal=False,
                sequence_model_name="CEHR_BERT_512_pipeline"
                + f"__target_MACE",
                target_label=None,  # only used for multi-classification
                random_seed=random_seed,
                split_group="split_group",
            ).train_transfer(test_dataset=test_dataset, save_probabilities=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Arguments for evaluation model"
    )
    parser.add_argument(
        "-c",
        "--path2cohort",
        dest="path2cohort",
        action="store",
        help="The path to the cohort",
        required=True,
    )
    #path2cohort = Path("/export/home/cse210037/Matthieu/medical_embeddings_transfer/data/mace__age_min_18__dates_2018_2020__task__MACE@360__index_visit_random")
    parser.add_argument(
        "-p",
        "--pretrained_model_name",
        dest="pretrained_model_name",
        action="store",
        help="The name of the pretrained model",
        required=True,
    )
    # "bert_model-pretrained__trainsize_1.h5"
    parser.add_argument(
        "-r",
        "--random_seed",
        dest="random_seed",
        action="store",
        type=int,
        help="The random seed",
        required=True,
    )
    config = parser.parse_args()
    main(config)

