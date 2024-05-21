# Automatic Pipeline for KGML-xDTD Model Training

This repository is built for generating a automatic pipeline to train [KGML-xDTD](https://github.com/chunyuma/KGML-xDTD) model based on [Snakemake](https://snakemake.readthedocs.io/en/stable/).

## Installation
1. To run this pipeline, please first install [conda](https://conda.io/projects/conda/en/latest/user-guide/install/index.html) and then run the following commands:
```bash
conda env create -f envs/graphsage_p2.7env.yml
conda env create -f envs/xDTD_training_pipeline_env.yml

## activiate the 'xDTD_training_pipeline' conda environment
conda activate xDTD_training_pipeline
```

## Get Config Json Files from RTX Github Repo
You need to have permission to access the latest `config_dbs.json` and `config_secrets.json` from [RTX](https://github.com/RTXteam/RTX) Github Repo. If you have, run the `snakemake` program will automatically download these two files. Otherwise, you will get an error.

## Modify the `config.yaml` File
You may need to change the following parameters in the `config.yaml` before you run the pipeline:
```yaml
RTXINFO:
  GITHUB_LINK: " https://raw.githubusercontent.com/RTXteam/RTX/master" ## you might need to change this linke to specific branch that has correct config_secrets.json and config_dbs.json

KG2INFO:
  BIOLINK_VERSION: "3.1.2" ## change this according to what biolink version from which the KG2 that you uses was built.

SYSTEMINFO:
  NUM_CPU: 200 ## change this according to your machine configuration

TRAINING_DATA:
  MOLEPRO_API_LINK: https://molepro-trapi.transltr.io/molepro/trapi/v1.4 ## please make sure this is the latest Molepro API. Check it from https://t.biothings.io/registry?q=molepro&tags=asyncquery_status

MODELINFO:
  PARAMS:
    GPU: 1 ## if your machine has only one GPU, you should set this to the default value, that is 0.
    
PARALLEL_PRECOMPUTE:
  K: 50 ## You may need to consider your machine RAM to set this parameter. We have 3T RAM to allow it to be 50.
  N_drugs: 150
  N_paths: 50
  BATCH_SIZE: 200

DATABASE:
  DATABASE_NAME: 'ExplainableDTD_v1.0_KG2.x.x.db' ## you may want to change it to something like ExplainableDTD_v1.3_KG2.8.0.1.db.
```

## Download the `drugbank.xml` File from DrugBank
You will need a drugbank acoount and request a permission from them to download that file from [here](https://go.drugbank.com/releases/latest).

## Run Pipeline
You can run the following command to run the pipeline:
```
nohup snakemake --cores 16 -s Run_Pipeline.smk targets &
```

__Please note that the last two steps (e.g., steps 24 and 25) can't be automatically executed in the pipeline since step 23 needs to be run in the background. I have commented the steps 24 and 25. Once step 23 is done, please comment out the steps 24 and 25 part in Run_Pipeline.smk and run the above command again__

## More Descriptions About Each Step in the Pipeline

### step1_download_RTXconfig
This step is to download the required RTX config files from the Github server and its internal server. You wil need a permission to download the `config_secrets.json` from its internal server.

### step2_download_trainingdata
This step is to download the training data `training_data.tar.gz` and from [Zendo](https://zenodo.org/record/7582233), as well as the DrugMechDB yaml file `indication_paths.yaml` from [DrugMechDB](https://github.com/SuLab/DrugMechDB).

### step3_download_data_and_kg2
This step is to download the necessary graph data from the KG2 neo4j endpoint. This step also needs `config_secrets.json` from its internal server. So please make sure the step1 can successufally downalod this file.

### step4_filtered_graph_nodes_and_edges
This step is to filter out some nodes with "unused" node types (as least for the drug treatment prediction) and the "SemMedDB" edges based on certain thresholds (e.g., Number of Publication Abstracts and NGD). It will take 1~2 days.

### step5_generate_tp_and_tn_pairs
This step is to generate high-quality true positve and true negative training drug-disease pairs.

### step6_preprocess_data
This step is to generate the ncessary input data for the downstream model training steps.

### step7_process_drugbank_action_desc
This step is to process data `drugbank.xml` file downloaded from DrugBank above. So please make sure you have successfully downloaded this dataset above.

### step8_integrate_drugbank_and_molepro_data
This step is to extract the relationship between drug and genes from both DrugBank data and MolePro data for the downstream model training steps. It will takes a long time to run because it depends on the speed of the molepro API. To avoid calling the molepro API, we use the data `molepro_df_bakup.txt` collected before in default. But if you want to re-collect it, please delete this file (BUT DON'T PUSH THIS ACTION TO GITHUB).

### step9_check_reachable
This step is to check whether there are 3-hop reachable paths between a given drug and disease through a specific gene.

### step10_generate_expert_paths
This step is to generate the input path data for the download model training steps

### step11_split_data_train_val_test
This step is to split data into training, validation, and test data.

### step12_calculate_attribute_embedding
This step is to calculate the attribute embedding using the PubMedBert Model.

### step13_graphsage_data_generation
This step is to generate the input data for runnig [GraphSage](https://snap.stanford.edu/graphsage/#:~:text=GraphSAGE%20is%20a%20framework%20for,Datasets) model. It wil take a few hours.

### step14_generate_random_walk
This step is to generate random walk data for running GraphSage, which will take 3~4 days.

### step15_generate_graphsage_embedding
This step is to run GraphSage model to generate the input node embeddings for the Random Forest model below.

### step16_transform_format
This step is tranform the file format of the input node embeddings.

### step17_pretrain_RF_model
This step it to train a Random Forest model for drug-disease treatment prediction.

### step18_generate_expert_path_transition
This step is to prepare the guided path for model training and conver it to an appropriate format.

## step19_pretrain_ac_model
This step is pre-trained the ADAC model for drug-disease treatment path explanation.

## step20_train_adac_model
This step is to formally traing the ADAC model.

## step21_select_best_model
This step is to evaluate the model in each training epoch and select the best one.

## step22_split_disease_into_K_pieces
This step is to split disease into K pieces for download pre-computation.

## step23_precompute_all_drug_disease_pairs_in_parallel
This step is to call multiple CPUs to do pre-computation for all potential drug-disease pairs.

## step24_build_sql_database
This step is to build the SQL database.

## step25_build_mapping_database
This step is to build the mapping tables and add them into the SQL database.

## Contact
If you have any questions or need help, please contact @chunyuma or @dkoslicki.
