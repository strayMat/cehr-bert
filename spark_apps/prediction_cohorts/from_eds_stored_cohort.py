import argparse
from pathlib import Path
from typing import List
import pandas as pd
from pyspark.sql import SparkSession
from utils.spark_utils import create_sequence_data_with_att, extract_ehr_records
import pyspark.sql.functions as F


# Column names specific to the medem project:
COLNAME_EVENT_START = "start"
COLNAME_EVENT_DOMAIN = "event_source_type_concept_id"
COLNAME_EVENT_CODE = "event_source_concept_id"
COLNAME_OUTCOME = "y"
COLNAME_FOLLOWUP_START = "followup_start"


def create_cohort_from_eds_eventCohort(
    input_folder: str, output_folder: str, path2train_split: str = None
):
    """
    From a [EventCohort](), create the sequence of events necessary for cerh-bert
    fine tuning prediction tasks.

    TODO: Only support binary label right now.

    NB: the loaded events should be only those in the observation period for the
    predictive tasks.

    Args:
        input_folder (str): Directory of the Cohort

        output_folder (str): _description_

        train_split (str): path to a train split dataset.
    """
    input_folder_path = Path(input_folder)
    cohort_name = input_folder_path.name
    spark = SparkSession.builder.appName(
        f"Generate {cohort_name}"
    ).getOrCreate()
    # extract the events the input folder containing the events parquet.
    # The events should be only the one in the observation period.
    event = spark.read.parquet(str(input_folder_path / "event.parquet"))
    target = spark.read.parquet(str(input_folder_path / "person.parquet"))
    if path2train_split is not None:
        train_split_dataset = pd.read_parquet(path2train_split)
        train_ids = train_split_dataset.loc[
            train_split_dataset["dataset"] == "train"
        ]["person_id"]
        target = target.join(
            spark.createDataFrame(train_ids),
            on="person_id",
            how="inner",
        )
    event_w_cols_renamed = event.withColumn(
        "domain", F.col(COLNAME_EVENT_DOMAIN)
    ).withColumn("date", F.to_date(F.col(COLNAME_EVENT_START)))
    event_w_subset_columns = event_w_cols_renamed.select(
        "person_id",
        F.col(COLNAME_EVENT_CODE).alias("standard_concept_id"),
        "date",
        "visit_occurrence_id",
        "domain",
    )
    target_w_statics = (
        target.withColumn(
            "gender_concept_id",
            F.when(F.col("gender_source_value") == "m", 8507).otherwise(
                F.when(F.col("gender_source_value") == "f", 8532).otherwise(
                    F.lit(0)
                )
            ),
        )
        .withColumn("cohort_member_id", F.col("person_id"))
        .withColumn("race_concept_id", F.lit(0))
        .withColumn("index_date", F.col(COLNAME_FOLLOWUP_START))
        .select(
            "person_id",
            "cohort_member_id",
            "gender_concept_id",
            "race_concept_id",
            "birth_datetime",
            F.col(COLNAME_OUTCOME).alias("label"),
            "index_date",
        )
    )
    patient_ehr_records = target_w_statics.join(
        event_w_subset_columns, on="person_id", how="inner"
    )
    patient_ehr_records_w_age = patient_ehr_records.withColumn(
        "age",
        F.ceil(
            F.months_between(F.col("date"), F.col("birth_datetime")) / F.lit(12)
        ),
    )
    # TODO: right now, I dont't extract visit type, but it should not be
    # important, since it seems to be used only for the pretraining.
    # If necessary, a join with the visit_occurrence table is necessary.
    cohort_sequence = create_sequence_data_with_att(
        patient_ehr_records_w_age,
        include_visit_type=False,
        exclude_visit_tokens=False,
    )
    # add target and demographics
    cohort_sequence_target = target_w_statics.join(
        cohort_sequence, on=["person_id", "cohort_member_id"], how="inner"
    ).withColumn(
        "age",
        F.ceil(
            F.months_between(F.col("index_date"), F.col("birth_datetime"))
            / F.lit(12)
        ),
    )
    cohort_sequence_target.write.mode("overwrite").parquet(str(output_folder))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Arguments for generating cohort from APHP-eds stored EventCohort"
    )
    parser.add_argument(
        "-i",
        "--input_folder",
        dest="input_folder",
        action="store",
        help="The path for your input_folder where the sequence data is",
        required=True,
    )
    parser.add_argument(
        "-o",
        "--output_folder",
        dest="output_folder",
        action="store",
        help="The path for your output_folder",
        required=True,
    )
    args = parser.parse_args()
    create_cohort_from_eds_eventCohort(
        input_folder=args.input_folder, output_folder=args.output_folder
    )
