# Automatic Pipeline for KGML-xDTD Model Training

This repo is built for generating a automatic pipeline for training [KGML-xDTD](https://github.com/chunyuma/KGML-xDTD) model based on [Snakemake](https://snakemake.readthedocs.io/en/stable/).

## Installation
1. To run this pipeline, please first install [conda](https://conda.io/projects/conda/en/latest/user-guide/install/index.html) and then run the following commands:
```
conda env create -f envs/graphsage_p2.7env.yml
conda env create -f envs/xDTD_training_pipeline_env.yml

## activiate the 'xDTD_training_pipeline' conda environment
conda activate xDTD_training_pipeline
```

## Download Config Json Files from RTX Github Repo
You need to have permission to access the latest `config_dbs.json` and `config_secrets.json` from [RTX](https://github.com/RTXteam/RTX) Github Repo. If you have, run the following commands:
```
ln -s [RTX_Repo_path]/code/config_dbs.json
ln -s [RTX_Repo_path]/code/config_secrets.json
```

## Modify Configuration for Your Machine
You need to change some parameters such as `SYSTEMINFO.NUM_CPU`, `PARALLEL_PRECOMPUTE.NUM_GPU`, `KG2INFO.BIOLINK_VERSION` based on your machine setting and biolink version.

## Run Pipeline
You can run the following command to run the pipeline:
```
nohup snakemake --cores 16 -s Run_Pipeline.smk targets &
```
