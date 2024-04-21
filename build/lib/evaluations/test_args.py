from evaluations.eds_prognosis_pipeline import (
    create_parse_args_pipeline_evaluation,
)


def main(pipeline_config):
    print(pipeline_config)


if __name__ == "__main__":
    pipeline_config = create_parse_args_pipeline_evaluation()
    main(pipeline_config=pipeline_config)
