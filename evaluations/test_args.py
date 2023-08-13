from evaluations.eds_full_pipeline import create_parse_args_pipeline_evaluation


def main(pipeline_config):
    breakpoint()
    print(pipeline_config)


if __name__ == "__main__":
    pipeline_config = create_parse_args_pipeline_evaluation()
    main(pipeline_config=pipeline_config)
