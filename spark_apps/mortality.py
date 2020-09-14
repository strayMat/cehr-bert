import pyspark.sql.functions as F

from spark_apps.spark_app_base import LastVisitCohortBuilderBase
from spark_apps.parameters import create_spark_args

QUALIFIED_DEATH_DATE_QUERY = """
WITH max_death_date_cte AS 
(
    SELECT 
        person_id,
        MAX(death_date) AS death_date
    FROM global_temp.death
    GROUP BY person_id
)

SELECT
    dv.person_id,
    dv.death_date
FROM
(
    SELECT DISTINCT
        d.person_id,
        d.death_date,
        FIRST(DATE(v.visit_start_date)) OVER(PARTITION BY v.person_id ORDER BY DATE(v.visit_start_date) DESC) AS last_visit_start_date
    FROM max_death_date_cte AS d
    JOIN global_temp.visit_occurrence AS v
        ON d.person_id = v.person_id
) dv
WHERE dv.last_visit_start_date <= dv.death_date
"""

COHORT_QUERY_TEMPLATE = """
WITH last_visit_cte AS (
    SELECT DISTINCT
        v.person_id,
        FIRST(v.visit_occurrence_id) OVER(PARTITION BY v.person_id ORDER BY DATE(v.visit_start_date) DESC) AS last_visit_occurrence_id,
        FIRST(DATE(v.visit_start_date)) OVER(PARTITION BY v.person_id ORDER BY DATE(v.visit_start_date) DESC) AS last_visit_start_date
    FROM global_temp.visit_occurrence AS v
)

SELECT
    p.person_id,
    p.last_visit_occurrence_id AS visit_occurrence_id,
    COALESCE(d.death_date, p.last_visit_start_date) AS index_date,
    CAST(ISNOTNULL(d.person_id) AS INT) AS label
FROM last_visit_cte AS p
LEFT JOIN global_temp.death AS d
    ON p.person_id = d.person_id
WHERE v.last_visit_start_date >= '{date_lower_bound}' AND v.last_visit_start_date <= '{date_upper_bound}'
"""

DOMAIN_TABLE_LIST = ['condition_occurrence', 'drug_exposure', 'procedure_occurrence']

COHORT_TABLE = 'cohort'
DEATH = 'death'
PERSON = 'person'
VISIT_OCCURRENCE = 'visit_occurrence'
DEPENDENCY_LIST = [DEATH, PERSON, VISIT_OCCURRENCE]


class MortalityCohortBuilder(LastVisitCohortBuilderBase):

    def preprocess_dependency(self):
        self.spark.sql(QUALIFIED_DEATH_DATE_QUERY).createOrReplaceGlobalTempView(DEATH)

        cohort_query = COHORT_QUERY_TEMPLATE.format(date_lower_bound=self._date_lower_bound,
                                                    date_upper_bound=self._date_upper_bound)

        cohort = self.spark.sql(cohort_query)
        cohort.createOrReplaceGlobalTempView(COHORT_TABLE)

        self._dependency_dict[COHORT_TABLE] = cohort

    def create_incident_cases(self):
        cohort = self._dependency_dict[COHORT_TABLE]
        person = self._dependency_dict[PERSON]

        incident_cases = cohort.where(F.col('label') == 1) \
            .join(person, 'person_id') \
            .withColumn('age', F.year('index_date') - F.col('year_of_birth')) \
            .select(F.col('person_id'),
                    F.col('age'),
                    F.col('gender_concept_id'),
                    F.col('race_concept_id'),
                    F.col('year_of_birth'),
                    F.col('visit_start_date'),
                    F.col('visit_occurrence_id'),
                    F.col('label')).distinct()

        return incident_cases

    def create_control_cases(self):
        cohort = self._dependency_dict[COHORT_TABLE]
        person = self._dependency_dict[PERSON]

        control_cases = cohort.where(F.col('label') == 0) \
            .join(person, 'person_id') \
            .withColumn('age', F.year('visit_start_date') - F.col('year_of_birth')) \
            .select(F.col('person_id'),
                    F.col('age'),
                    F.col('gender_concept_id'),
                    F.col('race_concept_id'),
                    F.col('year_of_birth'),
                    F.col('visit_start_date'),
                    F.col('visit_occurrence_id'),
                    F.col('label')).distinct()

        return control_cases


def main(cohort_name, input_folder, output_folder, date_lower_bound, date_upper_bound,
         age_lower_bound, age_upper_bound,
         observation_window, prediction_window):
    cohort_builder = MortalityCohortBuilder(cohort_name,
                                            input_folder,
                                            output_folder,
                                            date_lower_bound,
                                            date_upper_bound,
                                            age_lower_bound,
                                            age_upper_bound,
                                            observation_window,
                                            prediction_window,
                                            DOMAIN_TABLE_LIST,
                                            DEPENDENCY_LIST)

    cohort_builder.build()


if __name__ == '__main__':
    spark_args = create_spark_args()

    main(spark_args.cohort_name,
         spark_args.input_folder,
         spark_args.output_folder,
         spark_args.date_lower_bound,
         spark_args.date_upper_bound,
         spark_args.lower_bound,
         spark_args.upper_bound,
         spark_args.observation_window,
         spark_args.prediction_window)
