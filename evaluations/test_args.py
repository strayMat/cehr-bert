from config.model_configs import create_bert_model_config
from config.parse_args import create_parse_args_base_bert
from evaluations.evaluation import create_evaluation_args


def main(pretrain_config, evaluation_config):
    breakpoint()
    print(pretrain_config, evaluation_config)


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

    main(pretrain_config=pretrain_config, evaluation_config=evaluation_config)
