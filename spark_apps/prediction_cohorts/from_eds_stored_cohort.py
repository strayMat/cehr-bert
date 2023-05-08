import argparse
import logging
from pathlib import Path
from typing import List
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql import DataFrame
from utils.spark_utils import create_sequence_data_with_att, extract_ehr_records
import pyspark.sql.functions as F


# Column names specific to the medem project:
COLNAME_EVENT_START = "start"
COLNAME_EVENT_DOMAIN = "event_source_type_concept_id"
COLNAME_EVENT_CODE = "event_source_concept_id"
COLNAME_OUTCOME = "y"
COLNAME_FOLLOWUP_START = "followup_start"


def create_cohort_from_eds_eventCohort_dir(
    input_folder: str,
    output_folder: str,
    train_test_split_folder: str = None,
    split_group: str = None,
):
    """From a [EventCohort](), create the sequence of events necessary for cerh-bert
    fine tuning prediction tasks.

    NB: the loaded events should be only those in the observation period for the
    predictive tasks.

    Args:
        input_folder (str): Directory of the Cohort

        output_folder (str): _description_

        train_split (str): path to a train split dataset.

        split_group (str, optional): if a split_group for specifing splitting
        (for example in cross validation), has been defined in the train_test_split_dataframe
    """
    input_folder_path = Path(input_folder)
    cohort_name = input_folder_path.name
    spark = SparkSession.builder.appName(
        f"Generate {cohort_name}"
    ).getOrCreate()
    # extract the events the input folder containing the events parquet.
    # The events should be only the one in the observation period.
    event = spark.read.parquet(str(input_folder_path / "event.parquet"))
    person = spark.read.parquet(str(input_folder_path / "person.parquet"))

    if train_test_split_folder is not None:
        train_split_dataset = pd.read_parquet(train_test_split_folder)
        train_test_split_cols = ["person_id"]
        if split_group is not None:
            train_test_split_cols += [split_group]
        for split_name in train_split_dataset["dataset"].unique():
            split_ids = train_split_dataset.loc[
                train_split_dataset["dataset"] == split_name
            ][train_test_split_cols]
            split_person = person.join(
                spark.createDataFrame(split_ids),
                on="person_id",
                how="inner",
            )
            split_event = event.join(
                spark.createDataFrame(split_ids[["person_id"]]),
                on="person_id",
                how="inner",
            )
            split_cohort_sequence_target = create_cohort_from_eds_eventCohort(
                person=split_person, event=split_event, split_group=split_group
            )
            split_cohort_sequence_target.write.mode("overwrite").parquet(
                str(output_folder) + f"_{split_name}"
            )

    else:
        cohort_sequence_target = create_cohort_from_eds_eventCohort(
            person=person, event=event
        )
        cohort_sequence_target.write.mode("overwrite").parquet(
            str(output_folder)
        )


def create_cohort_from_eds_eventCohort(
    person: DataFrame, event: DataFrame, split_group: str = None
) -> DataFrame:
    """
    From person and event, return the sequences for cehr-bert inputs.

    NB: the loaded events should be only those in the observation period for the
    predictive tasks.
    """

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
    statics_cols = [
        "person_id",
        "cohort_member_id",
        "gender_concept_id",
        "race_concept_id",
        "birth_datetime",
        F.col(COLNAME_OUTCOME).alias("label"),
        "index_date",
    ]
    if split_group is not None:
        statics_cols += [F.col(split_group).alias("split_group")]
    target_w_statics = (
        person.withColumn(
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
        .select(statics_cols)
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
    return cohort_sequence_target


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
    parser.add_argument(
        "-s",
        "--train_test_split_folder",
        dest="train_test_split_folder",
        action="store",
        help="The path to the train test split data frame formatted as ['person_id', 'dataset']",
        default=None,
    )
    parser.add_argument(
        "-sg",
        "--split_group",
        dest="split_group",
        action="store",
        help="The column name for a split_group present in the train test split to be stored in the final sequences.",
        default=None,
    )
    args = parser.parse_args()
    create_cohort_from_eds_eventCohort_dir(
        input_folder=args.input_folder,
        output_folder=args.output_folder,
        train_test_split_folder=args.train_test_split_folder,
        split_group=args.split_group,
    )
