#!/bin/bash
#SBATCH --job-name=los_cbert
#SBATCH --time 48:00:00
#SBATCH --gres=gpu:t4:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=5
#SBATCH --partition gpuT4
#SBATCH --mem=24G
#SBATCH --container-image /scratch/images/sparkhadoop.sqsh  --container-mounts=/export/home/$USER:/export/home/$USER,/export/home/share:/export/home/share,/data/scratch/$USER:/data/scratch/$USER --container-mount-home --container-writable --container-workdir=/
#SBATCH --output=logs/slurm-%j-stdout.log
#SBATCH --error=logs/slurm-%j-stderr.log

export PATH=/usr/hdp/current/accumulo-client/bin:/usr/hdp/current/atlas-server/bin:/usr/hdp/current/beacon-client/bin:/usr/hdp/current/hive-client/bin:/usr/hdp/current/falcon-client/bin:/usr/hdp/current/flume-server/bin:/usr/hdp/current/hadoop-client/bin:/usr/hdp/current/hbase-client/bin:/usr/hdp/current/hadoop-hdfs-client/bin:/usr/hdp/current/hadoop-mapreduce-client/bin:/usr/hdp/current/hadoop-yarn-client/bin:/usr/hdp/current/hive-client/bin:/usr/hdp/current/hive-hcatalog/bin:/usr/hdp/current/hive-server2/bin:/usr/hdp/current/kafka-broker/bin:/usr/hdp/current/mahout-client/bin:/usr/hdp/current/oozie-client/bin:/usr/hdp/current/oozie-server/bin:/usr/hdp/current/phoenix-client/bin:/usr/hdp/current/pig-client/bin:/usr/hdp/share/hst/hst-agent/python-wrap:/usr/hdp/current/slider-client/bin:/usr/hdp/current/sqoop-client/bin:/usr/hdp/current/sqoop-server/bin:/usr/hdp/current/storm-slider-client/bin:/usr/hdp/current/zookeeper-client/bin:/usr/hdp/current/zookeeper-server/bin:/export/home/opt/jupyterhub/conda/bin:/export/home/opt/jupyterhub/node/bin:/export/home/opt/bin:/export/home/cse210038/.user_conda/miniconda/envs/event2vec/bin:/usr/hdp/current/accumulo-client/bin:/usr/hdp/current/atlas-server/bin:/usr/hdp/current/beacon-client/bin:/usr/hdp/current/hive-client/bin:/usr/hdp/current/falcon-client/bin:/usr/hdp/current/flume-server/bin:/usr/hdp/current/hadoop-client/bin:/usr/hdp/current/hbase-client/bin:/usr/hdp/current/hadoop-hdfs-client/bin:/usr/hdp/current/hadoop-mapreduce-client/bin:/usr/hdp/current/hadoop-yarn-client/bin:/usr/hdp/current/hive-client/bin:/usr/hdp/current/hive-hcatalog/bin:/usr/hdp/current/hive-server2/bin:/usr/hdp/current/kafka-broker/bin:/usr/hdp/current/mahout-client/bin:/usr/hdp/current/oozie-client/bin:/usr/hdp/current/oozie-server/bin:/usr/hdp/current/phoenix-client/bin:/usr/hdp/current/pig-client/bin:/usr/hdp/share/hst/hst-agent/python-wrap:/usr/hdp/current/slider-client/bin:/usr/hdp/current/sqoop-client/bin:/usr/hdp/current/sqoop-server/bin:/usr/hdp/current/storm-slider-client/bin:/usr/hdp/current/zookeeper-client/bin:/usr/hdp/current/zookeeper-server/bin:/export/home/opt/jupyterhub/conda/bin:/export/home/opt/jupyterhub/node/bin:/export/home/opt/bin:/usr/hdp/current/accumulo-client/bin:/usr/hdp/current/atlas-server/bin:/usr/hdp/current/beacon-client/bin:/usr/hdp/current/hive-client/bin:/usr/hdp/current/falcon-client/bin:/usr/hdp/current/flume-server/bin:/usr/hdp/current/hadoop-client/bin:/usr/hdp/current/hbase-client/bin:/usr/hdp/current/hadoop-hdfs-client/bin:/usr/hdp/current/hadoop-mapreduce-client/bin:/usr/hdp/current/hadoop-yarn-client/bin:/usr/hdp/current/hive-client/bin:/usr/hdp/current/hive-hcatalog/bin:/usr/hdp/current/hive-server2/bin:/usr/hdp/current/kafka-broker/bin:/usr/hdp/current/mahout-client/bin:/usr/hdp/current/oozie-client/bin:/usr/hdp/current/oozie-server/bin:/usr/hdp/current/phoenix-client/bin:/usr/hdp/current/pig-client/bin:/usr/hdp/share/hst/hst-agent/python-wrap:/usr/hdp/current/slider-client/bin:/usr/hdp/current/sqoop-client/bin:/usr/hdp/current/sqoop-server/bin:/usr/hdp/current/storm-slider-client/bin:/usr/hdp/current/zookeeper-client/bin:/usr/hdp/current/zookeeper-server/bin:/export/home/opt/jupyterhub/conda/bin:/export/home/opt/jupyterhub/node/bin:/export/home/opt/bin:/usr/hdp/current/accumulo-client/bin:/usr/hdp/current/atlas-server/bin:/usr/hdp/current/beacon-client/bin:/usr/hdp/current/hive-client/bin:/usr/hdp/current/falcon-client/bin:/usr/hdp/current/flume-server/bin:/usr/hdp/current/hadoop-client/bin:/usr/hdp/current/hbase-client/bin:/usr/hdp/current/hadoop-hdfs-client/bin:/usr/hdp/current/hadoop-mapreduce-client/bin:/usr/hdp/current/hadoop-yarn-client/bin:/usr/hdp/current/hive-client/bin:/usr/hdp/current/hive-hcatalog/bin:/usr/hdp/current/hive-server2/bin:/usr/hdp/current/kafka-broker/bin:/usr/hdp/current/mahout-client/bin:/usr/hdp/current/oozie-client/bin:/usr/hdp/current/oozie-server/bin:/usr/hdp/current/phoenix-client/bin:/usr/hdp/current/pig-client/bin:/usr/hdp/share/hst/hst-agent/python-wrap:/usr/hdp/current/slider-client/bin:/usr/hdp/current/sqoop-client/bin:/usr/hdp/current/sqoop-server/bin:/usr/hdp/current/storm-slider-client/bin:/usr/hdp/current/zookeeper-client/bin:/usr/hdp/current/zookeeper-server/bin:/export/home/opt/jupyterhub/conda/bin:/export/home/opt/jupyterhub/node/bin:/export/home/opt/bin:/usr/hdp/current/spark-2.4.3-client/bin:/usr/hdp/current/spark-2.4.3-client/bin:/usr/local/hadoop/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/export/home/opt/apps/texlive-20190227/2018/bin/x86_64-linux:/export/home/opt/apps/texlive-20190227/2018/bin/x86_64-linux:/export/home/opt/apps/texlive-20190227/2018/bin/x86_64-linux:/export/home/opt/apps/texlive-20190227/2018/bin/x86_64-linux:/export/home/cse210038/.local/bin:/export/home/cse210038/bin:$PATH
cd '/export/home/cse210038/Matthieu/cehr-bert/'


# This script launches a multi-output-classification problem by finetuning one model for each binary target in the cehr_bert_finetunique_sequences_train.label column.
local_cohort_dir="Matthieu/medical_embeddings_transfer/data/complete_hospitalization_los__age_min_18__dates_2017_2022__task__length_of_stay_categorical@3/"

## copie des data en local vers le ssd
scratch=/data/scratch
mySourcePath=$HOME/$local_cohort_dir
myDestPath=$scratch/$USER/$local_cohort_dir
mkdir -p $myDestPath

(cd ${mySourcePath}; tar cf - .) | (cd ${myDestPath}; tar xpf -)

pretrained_dir=$mySourcePath"cehr_bert_pretrained_pipeline"
pretrain_sequence=$myDestPath"cehr_bert_sequences"
train_sequence_dir=$myDestPath"cehr_bert_finetuning_sequences_train"
test_sequence_dir=$myDestPath"cehr_bert_finetuning_sequences_external_test"

# Create the evaluation folder
myOutPut="evaluation_train_val_split"
evaluation_dir=$mySourcePath$myOutPut
mkdir -p $evaluation_dir 
mkdir -p $pretrained_dir

/export/home/cse210038/.user_conda/miniconda/envs/cehr_bert/bin/python evaluations/eds_los_pipeline.py -i $pretrain_sequence -o $pretrained_dir -sd $train_sequence_dir -sdt $test_sequence_dir -ef $evaluation_dir -smn CEHR_BERT_512_pipeline -ut
