"""
This file is a SnakeMake Script to automate the xDTD model training

Usage:
    snakemake --cores 16 -s Run_Pipeline.smk targets
"""
## Import Config Files
configfile: "./config.yaml"

## Import Python standard libraries
import os, sys

## Define Some Global Variables
CURRENT_PATH = os.getcwd()

## Create Required Folders
if not os.path.exists(os.path.join(CURRENT_PATH, "data")):
    os.makedirs(os.path.join(CURRENT_PATH, "data"))
if not os.path.exists(os.path.join(CURRENT_PATH, "log_folder")):
    os.makedirs(os.path.join(CURRENT_PATH, "log_folder"))
if not os.path.exists(os.path.join(CURRENT_PATH, "models")):
    os.makedirs(os.path.join(CURRENT_PATH, "models"))
if not os.path.exists(os.path.join(CURRENT_PATH, "results")):
    os.makedirs(os.path.join(CURRENT_PATH, "results"))

## Build Rules
rule targets:
    input:
        ancient(os.path.join(CURRENT_PATH, config['RTXINFO']['SECRET_CONFIGFILE'])),
        ancient(os.path.join(CURRENT_PATH, config['RTXINFO']['DB_CONFIGFILE'])),
        ancient(os.path.join(CURRENT_PATH, "data", config['ZENODOINFO']['TRAINING_DATA'])),
        ancient(os.path.join(CURRENT_PATH, "data", config['DRUGMECHDBINFO']['DRUGMECHDB_PATH'])),
        ancient(os.path.join(CURRENT_PATH, "data", 'graph_edges.txt')),
        ancient(os.path.join(CURRENT_PATH, "data", 'all_graph_nodes_info.txt')),
        ancient(os.path.join(CURRENT_PATH, "data", 'filtered_graph_edges.txt')),
        ancient(os.path.join(CURRENT_PATH, "data", 'filtered_graph_nodes_info.txt')),
        ancient(os.path.join(CURRENT_PATH, "data", 'tp_pairs.txt')),
        ancient(os.path.join(CURRENT_PATH, "data", 'tn_pairs.txt')),
        ancient(os.path.join(CURRENT_PATH, "data", 'all_known_tps.txt')),
        ancient(os.path.join(CURRENT_PATH, "data", 'entity2freq.txt')),
        ancient(os.path.join(CURRENT_PATH, "data", 'relation2freq.txt')),
        ancient(os.path.join(CURRENT_PATH, "data", 'type2freq.txt')),
        ancient(os.path.join(CURRENT_PATH, "data", 'adj_list.pkl')),
        ancient(os.path.join(CURRENT_PATH, "data", 'entity2typeid.pkl')),
        ancient(os.path.join(CURRENT_PATH, "data", 'kg.pgrk')),
        ancient(os.path.join(CURRENT_PATH, "data", 'expert_path_files', 'drugbank_dict.pkl')),
        ancient(os.path.join(CURRENT_PATH, "data", 'expert_path_files', 'drugbank_mapping.txt')),
        ancient(os.path.join(CURRENT_PATH, "data", 'expert_path_files', 'p_expert_paths.txt')),
        ancient(os.path.join(CURRENT_PATH, "data", 'expert_path_files', 'all_drugs.txt')),
        ancient(os.path.join(CURRENT_PATH, "data", 'expert_path_files', 'p_expert_paths_combined.txt')),
        ancient(os.path.join(CURRENT_PATH, "data", 'expert_path_files', f"reachable_expert_paths_max{config['MODELINFO']['PARAMS']['MAX_PATH']}.txt")),
        ancient(os.path.join(CURRENT_PATH, "data", 'expert_path_files', f"reachable_tp_pairs_max{config['MODELINFO']['PARAMS']['MAX_PATH']}.txt")),
        ancient(os.path.join(CURRENT_PATH, "data", 'expert_path_files', f"unreachable_tp_pairs_max{config['MODELINFO']['PARAMS']['MAX_PATH']}.txt")),
        ancient(os.path.join(CURRENT_PATH, "data", 'expert_path_files', f"expert_demonstration_paths_max{config['MODELINFO']['PARAMS']['MAX_PATH']}_raw.pkl")),
        ancient(os.path.join(CURRENT_PATH, "data", 'expert_path_files', f"expert_demonstration_paths_max{config['MODELINFO']['PARAMS']['MAX_PATH']}_filtered.pkl")),
        ancient(os.path.join(CURRENT_PATH, "data", 'expert_path_files', f"expert_demonstration_paths_translate_max{config['MODELINFO']['PARAMS']['MAX_PATH']}_filtered.pkl")),
        ancient(os.path.join(CURRENT_PATH, "data", 'expert_path_files', f"expert_demonstration_relation_entity_max{config['MODELINFO']['PARAMS']['MAX_PATH']}_filtered.pkl")),
        ancient(os.path.join(CURRENT_PATH, "data", "pretrain_reward_shaping_model_train_val_test_random_data_3class", "train_pairs.txt")),
        ancient(os.path.join(CURRENT_PATH, "data", "pretrain_reward_shaping_model_train_val_test_random_data_3class", "val_pairs.txt")),
        ancient(os.path.join(CURRENT_PATH, "data", "pretrain_reward_shaping_model_train_val_test_random_data_3class", "test_pairs.txt")),
        ancient(os.path.join(CURRENT_PATH, "data", "pretrain_reward_shaping_model_train_val_test_random_data_3class", "random_pairs.txt")),
        ancient(os.path.join(CURRENT_PATH, "data", 'expert_path_files', f"train_expert_demonstration_relation_entity_max{config['MODELINFO']['PARAMS']['MAX_PATH']}_filtered.pkl")),
        ancient(os.path.join(CURRENT_PATH, "data", 'expert_path_files', f"val_expert_demonstration_relation_entity_max{config['MODELINFO']['PARAMS']['MAX_PATH']}_filtered.pkl")),
        ancient(os.path.join(CURRENT_PATH, "data", 'expert_path_files', f"test_expert_demonstration_relation_entity_max{config['MODELINFO']['PARAMS']['MAX_PATH']}_filtered.pkl")),
        ancient(os.path.join(CURRENT_PATH, "data", "text_embedding", "embedding_biobert_namecat.pkl")),
        ancient(os.path.join(CURRENT_PATH, "data", "graphsage_input", "id_map.txt")),
        ancient(os.path.join(CURRENT_PATH, "data", "graphsage_input", "category_map.txt")),
        ancient(os.path.join(CURRENT_PATH, "data", "graphsage_input", "data-G.json")),
        ancient(os.path.join(CURRENT_PATH, "data", "graphsage_input", "data-class_map.json")),
        ancient(os.path.join(CURRENT_PATH, "data", "graphsage_input", "data-id_map.json")),
        ancient(os.path.join(CURRENT_PATH, "data", "graphsage_input", "data-feats.npy")),
        ancient(os.path.join(CURRENT_PATH, "data", "graphsage_input", "data-walks.txt")),
        ancient(os.path.join(CURRENT_PATH, "unsup-graphsage_input", "graphsage_mean_big_0.001000", "val.npy")),
        ancient(os.path.join(CURRENT_PATH, "unsup-graphsage_input", "graphsage_mean_big_0.001000", "val.txt")),
        ancient(os.path.join(CURRENT_PATH, "data", "graphsage_output", "unsuprvised_graphsage_entity_embeddings.pkl")),
        ancient(os.path.join(CURRENT_PATH, "models", "RF_model_3class", "RF_model.pt")),
        ancient(os.path.join(CURRENT_PATH, "data", "expert_path_files", f"train_expert_transitions_history{config['MODELINFO']['PARAMS']['STATE_HISTORY']}.pkl")),
        ancient(os.path.join(CURRENT_PATH, "models", "pretrain_AC_model", "pretrained_ac_model.pt")),
        ancient(os.path.join(CURRENT_PATH, "models", "ADAC_model", "policy_net", "policy_model_epoch51.pt")),
        ancient(os.path.join(CURRENT_PATH, "models", "ADAC_model", "policy_net", "best_moa_model.pt")),
        ancient(os.path.join(CURRENT_PATH, "data", "disease_sets", "disease_set1.txt")),
        ancient(os.path.join(CURRENT_PATH, "data", "filtered_drug_nodes_for_precomputation.pkl")),
        ancient(os.path.join(CURRENT_PATH, "results", "step23_done.txt"))
        # ancient(os.path.join(CURRENT_PATH, config['DATABASE']['DATABASE_NAME'])),
        # ancient(os.path.join(CURRENT_PATH, "results", "step25_done.txt")),



# download RTX config file from server
rule step1_download_RTXconfig:
    output:
        os.path.join(CURRENT_PATH, config['RTXINFO']['SECRET_CONFIGFILE']),
        os.path.join(CURRENT_PATH, config['RTXINFO']['DB_CONFIGFILE'])
    params:
        server_name = config['RTXINFO']['CONFIG_SERVER'],
        github_link = config['RTXINFO']['GITHUB_LINK'],
        secret_configfile = config['RTXINFO']['SECRET_CONFIGFILE'],
        db_configfile = config['RTXINFO']['DB_CONFIGFILE']
    run:
        shell("scp {params.server_name}:{params.secret_configfile} ."),
        shell("wget {params.github_link}/code/{params.db_configfile}")

# download training data from Zenodo (https://zenodo.org/record/7582233)
rule step2_download_trainingdata:
    output:
        os.path.join(CURRENT_PATH, "data", config['ZENODOINFO']['TRAINING_DATA']),
        os.path.join(CURRENT_PATH, "data", config['DRUGMECHDBINFO']['DRUGMECHDB_PATH'])
    params:
        zenodo_link = config['ZENODOINFO']['LINK'],
        drugmechdb_link = config['DRUGMECHDBINFO']['LINK'],
        training_data = config['ZENODOINFO']['TRAINING_DATA'],
        drugmechdb_path = config['DRUGMECHDBINFO']['DRUGMECHDB_PATH']
    run:
        shell("curl {params.zenodo_link}/files/{params.training_data} -o ./data/{params.training_data}"),
        shell("tar zxvf ./data/{params.training_data} -C ./data/"),
        shell("curl {params.drugmechdb_link}/{params.drugmechdb_path} -o ./data/{params.drugmechdb_path}")

rule step3_download_data_and_kg2:
    input:
        script = ancient(os.path.join(CURRENT_PATH, "scripts", "download_data_and_kg.py")),
        secret_configfile = ancient(os.path.join(CURRENT_PATH, config['RTXINFO']['SECRET_CONFIGFILE'])),
        db_configfile = ancient(os.path.join(CURRENT_PATH, config['RTXINFO']['DB_CONFIGFILE']))
    output:
        os.path.join(CURRENT_PATH, "data", 'graph_edges.txt'),
        os.path.join(CURRENT_PATH, "data", 'all_graph_nodes_info.txt')
    shell:
        """
        python {input.script} --db_config_path {input.db_configfile} --secret_config_path {input.secret_configfile}
        """

rule step4_filtered_graph_nodes_and_edges:
    input:
        script = ancient(os.path.join(CURRENT_PATH, "scripts", "filter_kg2_nodes_and_edges.py")),
        graph_edges = ancient(os.path.join(CURRENT_PATH, "data", 'graph_edges.txt')),
        all_node_info = ancient(os.path.join(CURRENT_PATH, "data", 'all_graph_nodes_info.txt')),
        db_configfile = ancient(os.path.join(CURRENT_PATH, config['RTXINFO']['DB_CONFIGFILE']))
    output:
        os.path.join(CURRENT_PATH, "data", 'filtered_graph_edges.txt'),
        os.path.join(CURRENT_PATH, "data", 'filtered_graph_nodes_info.txt')
    params:
        pub_threshold = config['KG2INFO']['PUBLICATION_CUTOFF'],
        ngd_threshold = config['KG2INFO']['NGD_CUTOFF'],
        num_core = config['SYSTEMINFO']['NUM_CPU'],
        biolink_version = config['KG2INFO']['BIOLINK_VERSION']
    shell:
        """
        python {input.script} --db_config_path {input.db_configfile} \
                              --graph_nodes {input.all_node_info} \
                              --graph_edges {input.graph_edges} \
                              --pub_threshold {params.pub_threshold} \
                              --ngd_threshold {params.ngd_threshold} \
                              --num_core {params.num_core} \
                              --biolink_version {params.biolink_version} 
        """

rule step5_generate_tp_and_tn_pairs:
    input:
        script = ancient(os.path.join(CURRENT_PATH, "scripts", "generate_tp_tn_pairs.py")),
        secret_configfile = ancient(config['RTXINFO']['SECRET_CONFIGFILE']),
        db_configfile = ancient(config['RTXINFO']['DB_CONFIGFILE']),
        graph_edges = ancient(os.path.join(CURRENT_PATH, "data", 'filtered_graph_edges.txt')),
        training_data_unused = ancient(os.path.join(CURRENT_PATH, "data", config['ZENODOINFO']['TRAINING_DATA'])),
    output:
        os.path.join(CURRENT_PATH, "data", 'tp_pairs.txt'),
        os.path.join(CURRENT_PATH, "data", 'tn_pairs.txt'),
        os.path.join(CURRENT_PATH, "data", 'all_known_tps.txt'),
    params:
        mychem_tp = ancient(os.path.join(CURRENT_PATH, "data", config['TRAINING_DATA']['TP']['MYCHEM'])),
        semmed_tp = ancient(os.path.join(CURRENT_PATH, "data", config['TRAINING_DATA']['TP']['SEMMED'])),
        ndf_tp = ancient(os.path.join(CURRENT_PATH, "data", config['TRAINING_DATA']['TP']['NDF'])),
        repoDB_tp = ancient(os.path.join(CURRENT_PATH, "data", config['TRAINING_DATA']['TP']['REPODB'])),
        mychem_tn = ancient(os.path.join(CURRENT_PATH, "data", config['TRAINING_DATA']['TN']['MYCHEM'])),
        semmed_tn = ancient(os.path.join(CURRENT_PATH, "data", config['TRAINING_DATA']['TN']['SEMMED'])),
        ndf_tn = ancient(os.path.join(CURRENT_PATH, "data", config['TRAINING_DATA']['TN']['NDF'])),
        repoDB_tn = ancient(os.path.join(CURRENT_PATH, "data", config['TRAINING_DATA']['TN']['REPODB'])),
        cutoff = config['KG2INFO']['PUBLICATION_CUTOFF'],
        ngdcutoff = config['KG2INFO']['NGD_CUTOFF']
    shell:
        """
        python {input.script} --db_config_path {input.db_configfile} \
                              --secret_config_path {input.secret_configfile} \
                              --graph {input.graph_edges} \
                              --tp {params.mychem_tp} {params.semmed_tp} {params.ndf_tp} {params.repoDB_tp} \
                              --tn {params.mychem_tn} {params.semmed_tn} {params.ndf_tn} {params.repoDB_tn} \
                              --tncutoff {params.cutoff} \
                              --tpcutoff {params.cutoff} \
                              --ngdcutoff {params.ngdcutoff}        
        """

rule step6_preprocess_data:
    input:
        script = ancient(os.path.join(CURRENT_PATH, "scripts", "preprocess_data.py")),
        graph_nodes = ancient(os.path.join(CURRENT_PATH, "data", "filtered_graph_nodes_info.txt")),
        graph_edges = ancient(os.path.join(CURRENT_PATH, "data", 'filtered_graph_edges.txt'))
    output:
        os.path.join(CURRENT_PATH, "data", 'entity2freq.txt'),
        os.path.join(CURRENT_PATH, "data", 'relation2freq.txt'),
        os.path.join(CURRENT_PATH, "data", 'type2freq.txt'),
        os.path.join(CURRENT_PATH, "data", 'adj_list.pkl'),
        os.path.join(CURRENT_PATH, "data", 'entity2typeid.pkl'),
        os.path.join(CURRENT_PATH, "data", 'kg.pgrk')
    shell:
        """
        python {input.script} --graph_nodes {input.graph_nodes} --graph_edges {input.graph_edges}
        """

## to run this step, you need to first download 'drugbank.xml' from DrugBank website 'https://go.drugbank.com/releases/latest' and put it in data/ folder
rule step7_process_drugbank_action_desc:
    input:
        script = ancient(os.path.join(CURRENT_PATH, "scripts", "process_drugbank_action_desc.py")),
        secret_configfile = ancient(config['RTXINFO']['SECRET_CONFIGFILE']),
        db_configfile = ancient(config['RTXINFO']['DB_CONFIGFILE']),
        drugbank_xml = ancient(os.path.join(CURRENT_PATH, "data", config['TRAINING_DATA']['DRUGBANK_XML']))
    output:
        os.path.join(CURRENT_PATH, "data", 'expert_path_files', 'drugbank_dict.pkl'),
        os.path.join(CURRENT_PATH, "data", 'expert_path_files', 'drugbank_mapping.txt'),
        os.path.join(CURRENT_PATH, "data", 'expert_path_files', 'p_expert_paths.txt')
    shell:
        """
        python {input.script} --db_config_path {input.db_configfile} --secret_config_path {input.secret_configfile} --drugbankxml {input.drugbank_xml}
        """

## thi setp takes a long time to run because it depends on the speed of the molepro API
rule step8_integrate_drugbank_and_molepro_data:
    input:
        script = ancient(os.path.join(CURRENT_PATH, "scripts", "integrate_drugbank_and_molepro_data.py")),
        secret_configfile = ancient(config['RTXINFO']['SECRET_CONFIGFILE']),
        db_configfile = ancient(config['RTXINFO']['DB_CONFIGFILE']),
        drugbank_export_paths = ancient(os.path.join(CURRENT_PATH, "data", 'expert_path_files', 'p_expert_paths.txt'))
    output:
        os.path.join(CURRENT_PATH, "data", 'expert_path_files', 'all_drugs.txt'),
        os.path.join(CURRENT_PATH, "data", 'expert_path_files', 'p_expert_paths_combined.txt')
    params:
        molepro_api_link = config['TRAINING_DATA']['MOLEPRO_API_LINK']
    shell:
        """
        python {input.script} --db_config_path {input.db_configfile} \
                              --secret_config_path {input.secret_configfile} \
                              --drugbank_export_paths {input.drugbank_export_paths}
        """

rule step9_check_reachable:
    input:
        script = ancient(os.path.join(CURRENT_PATH, "scripts", "check_reachable.py")),
        true_pairs = ancient(os.path.join(CURRENT_PATH, "data", 'tp_pairs.txt')),
        entity2freq = ancient(os.path.join(CURRENT_PATH, "data", 'entity2freq.txt')),
        relation2freq = ancient(os.path.join(CURRENT_PATH, "data", 'relation2freq.txt')),
        adj_list = ancient(os.path.join(CURRENT_PATH, "data", 'adj_list.pkl')),
        kg_pgrk = ancient(os.path.join(CURRENT_PATH, "data", 'kg.pgrk')),
        combined_expert_paths = ancient(os.path.join(CURRENT_PATH, "data", 'expert_path_files', 'p_expert_paths_combined.txt'))
    output:
        os.path.join(CURRENT_PATH, "data", 'expert_path_files', f"reachable_expert_paths_max{config['MODELINFO']['PARAMS']['MAX_PATH']}.txt"),
        os.path.join(CURRENT_PATH, "data", 'expert_path_files', f"reachable_tp_pairs_max{config['MODELINFO']['PARAMS']['MAX_PATH']}.txt"),
        os.path.join(CURRENT_PATH, "data", 'expert_path_files', f"unreachable_tp_pairs_max{config['MODELINFO']['PARAMS']['MAX_PATH']}.txt")
    params:
        bandwidth = config['MODELINFO']['PARAMS']['BANDWIDTH'],
        max_path = config['MODELINFO']['PARAMS']['MAX_PATH']
    shell:
        """
        python {input.script} --bandwidth {params.bandwidth} \
                              --tp_pairs {input.true_pairs} \
                              --max_path {params.max_path} \
                              --combined_expert_paths {input.combined_expert_paths}
        """

rule step10_generate_expert_paths:
    input:
        script = ancient(os.path.join(CURRENT_PATH, "scripts", "generate_expert_paths.py")),
        reachable_expert_paths = ancient(os.path.join(CURRENT_PATH, "data", 'expert_path_files', f"reachable_expert_paths_max{config['MODELINFO']['PARAMS']['MAX_PATH']}.txt")),
        db_configfile = ancient(os.path.join(CURRENT_PATH, config['RTXINFO']['DB_CONFIGFILE']))
    output:
        os.path.join(CURRENT_PATH, "data", 'expert_path_files', f"expert_demonstration_paths_max{config['MODELINFO']['PARAMS']['MAX_PATH']}_raw.pkl"),
        os.path.join(CURRENT_PATH, "data", 'expert_path_files', f"expert_demonstration_paths_max{config['MODELINFO']['PARAMS']['MAX_PATH']}_filtered.pkl"),
        os.path.join(CURRENT_PATH, "data", 'expert_path_files', f"expert_demonstration_paths_translate_max{config['MODELINFO']['PARAMS']['MAX_PATH']}_filtered.pkl"),
        os.path.join(CURRENT_PATH, "data", 'expert_path_files', f"expert_demonstration_relation_entity_max{config['MODELINFO']['PARAMS']['MAX_PATH']}_filtered.pkl")
    params:
        ngd_threshold = config['KG2INFO']['NGD_CUTOFF'],
        bandwidth = config['MODELINFO']['PARAMS']['BANDWIDTH'],
        max_path = config['MODELINFO']['PARAMS']['MAX_PATH'],
        process = 180,
        batch_size = 500,
        biolink_version = config['KG2INFO']['BIOLINK_VERSION']
    shell:
        """
        python {input.script} --db_config_path {input.db_configfile} \
                              --biolink_version {params.biolink_version} \
                              --reachable_expert_paths {input.reachable_expert_paths} \
                              --bandwidth {params.bandwidth} \
                              --batch_size {params.batch_size} \
                              --process {params.process} \
                              --max_path {params.max_path} 
        """

rule step11_split_data_train_val_test:
    input:
        script = ancient(os.path.join(CURRENT_PATH, "scripts", "split_data_train_val_test.py")),
        graph_edges = ancient(os.path.join(CURRENT_PATH, "data", 'filtered_graph_edges.txt')),
        tp_pairs = ancient(os.path.join(CURRENT_PATH, "data", 'tp_pairs.txt')),
        tn_pairs = ancient(os.path.join(CURRENT_PATH, "data", 'tn_pairs.txt')),
        entity2freq = ancient(os.path.join(CURRENT_PATH, "data", 'entity2freq.txt')),
        type2freq = ancient(os.path.join(CURRENT_PATH, "data", 'type2freq.txt')),
        entity2typeid = ancient(os.path.join(CURRENT_PATH, "data", 'entity2typeid.pkl')),
        all_known_tps = ancient(os.path.join(CURRENT_PATH, "data", 'all_known_tps.txt')),
        filtered_expert_paths = ancient(os.path.join(CURRENT_PATH, "data", 'expert_path_files', f"expert_demonstration_paths_max{config['MODELINFO']['PARAMS']['MAX_PATH']}_filtered.pkl")),
        filtered_path_relation_entity = ancient(os.path.join(CURRENT_PATH, "data", 'expert_path_files', f"expert_demonstration_relation_entity_max{config['MODELINFO']['PARAMS']['MAX_PATH']}_filtered.pkl"))
    output:
        os.path.join(CURRENT_PATH, "data", "pretrain_reward_shaping_model_train_val_test_random_data_3class", "train_pairs.txt"),
        os.path.join(CURRENT_PATH, "data", "pretrain_reward_shaping_model_train_val_test_random_data_3class", "val_pairs.txt"),
        os.path.join(CURRENT_PATH, "data", "pretrain_reward_shaping_model_train_val_test_random_data_3class", "test_pairs.txt"),
        os.path.join(CURRENT_PATH, "data", "pretrain_reward_shaping_model_train_val_test_random_data_3class", "random_pairs.txt"),
        os.path.join(CURRENT_PATH, "data", 'expert_path_files', f"train_expert_demonstration_relation_entity_max{config['MODELINFO']['PARAMS']['MAX_PATH']}_filtered.pkl"),
        os.path.join(CURRENT_PATH, "data", 'expert_path_files', f"val_expert_demonstration_relation_entity_max{config['MODELINFO']['PARAMS']['MAX_PATH']}_filtered.pkl"),
        os.path.join(CURRENT_PATH, "data", 'expert_path_files', f"test_expert_demonstration_relation_entity_max{config['MODELINFO']['PARAMS']['MAX_PATH']}_filtered.pkl")
    params:
        n_random_test_mrr_hk = 500,
        train_val_test_size = "[0.8, 0.1, 0.1]",
        seed = config['MODELINFO']['PARAMS']['SEED'],
        max_path = config['MODELINFO']['PARAMS']['MAX_PATH']
    shell:
        """
        python {input.script} --graph_edges {input.graph_edges} \
                              --tp_pairs {input.tp_pairs} \
                              --tn_pairs {input.tn_pairs} \
                              --entity2freq {input.entity2freq} \
                              --type2freq {input.type2freq} \
                              --entity2typeid {input.entity2typeid} \
                              --all_known_tps {input.all_known_tps} \
                              --filtered_expert_paths {input.filtered_expert_paths} \
                              --filtered_path_relation_entity {input.filtered_path_relation_entity} \
                              --n_random_test_mrr_hk {params.n_random_test_mrr_hk} \
                              --train_val_test_size '{params.train_val_test_size}' \
                              --seed {params.seed} \
                              --max_path {params.max_path}
        """

rule step12_calculate_attribute_embedding:
    input:
        script = ancient(os.path.join(CURRENT_PATH, "scripts", "calculate_attribute_embedding.py")),
        node_info = ancient(os.path.join(CURRENT_PATH, "data", 'filtered_graph_nodes_info.txt'))
    output:
        os.path.join(CURRENT_PATH, "data", "text_embedding", "embedding_biobert_namecat.pkl")
    params:
        gpu = config['MODELINFO']['PARAMS']['GPU'],
        seed = config['MODELINFO']['PARAMS']['SEED'],
        batch_size = 64,
        pca_components = 80
    shell:
        """
        python {input.script} --node_info {input.node_info}  \
                              --gpu {params.gpu} \
                              --use_gpu \
                              --seed {params.seed} \
                              --pca_components {params.pca_components} \
                              --batch_size {params.batch_size}
        """

rule step13_graphsage_data_generation:
    input:
        script = ancient(os.path.join(CURRENT_PATH, "scripts", "graphsage_data_generation.py")),
        graph_nodes = ancient(os.path.join(CURRENT_PATH, "data", "filtered_graph_nodes_info.txt")),
        graph_edges = ancient(os.path.join(CURRENT_PATH, "data", "filtered_graph_edges.txt")),
        entity2freq = ancient(os.path.join(CURRENT_PATH, "data", "entity2freq.txt")),
        type2freq = ancient(os.path.join(CURRENT_PATH, "data", "type2freq.txt")),
        emb_file = ancient(os.path.join(CURRENT_PATH, "data", "text_embedding", "embedding_biobert_namecat.pkl"))
    output:
        os.path.join(CURRENT_PATH, "data", "graphsage_input", "id_map.txt"),
        os.path.join(CURRENT_PATH, "data", "graphsage_input", "category_map.txt"),
        os.path.join(CURRENT_PATH, "data", "graphsage_input", "data-G.json"),
        os.path.join(CURRENT_PATH, "data", "graphsage_input", "data-class_map.json"),
        os.path.join(CURRENT_PATH, "data", "graphsage_input", "data-id_map.json"),
        os.path.join(CURRENT_PATH, "data", "graphsage_input", "data-feats.npy")
    params:
        seed = config['MODELINFO']['PARAMS']['SEED'],
        feature_dim = 256,
        validation_percent = 0.3
    shell:
        """
        python {input.script} --graph_nodes {input.graph_nodes}  \
                              --graph_edges {input.graph_edges} \
                              --entity2freq {input.entity2freq} \
                              --type2freq {input.type2freq} \
                              --emb_file {input.emb_file} \
                              --seed {params.seed} \
                              --feature_dim {params.feature_dim} \
                              --validation_percent {params.validation_percent}
        """

rule step14_generate_random_walk:
    input:
        script = ancient(os.path.join(CURRENT_PATH, "scripts", "generate_random_walk.py")),
        Gjson = ancient(os.path.join(CURRENT_PATH, "data", "graphsage_input", "data-G.json"))
    output:
        os.path.join(CURRENT_PATH, "data", "graphsage_input", "data-walks.txt")
    params:
        walk_length = 30,
        number_of_walks = 10,
        batch_size = 200000,
        # process = 200
    shell:
        """
        python {input.script} --Gjson {input.Gjson}  \
                              --walk_length {params.walk_length} \
                              --number_of_walks {params.number_of_walks} \
                              --batch_size {params.batch_size}
        """

rule step15_generate_graphsage_embedding:
    input:
        id_map = ancient(os.path.join(CURRENT_PATH, "data", "graphsage_input", "id_map.txt")),
        data_category_map = ancient(os.path.join(CURRENT_PATH, "data", "graphsage_input", "category_map.txt")),
        data_Gjson = ancient(os.path.join(CURRENT_PATH, "data", "graphsage_input", "data-G.json")),
        data_class_map = ancient(os.path.join(CURRENT_PATH, "data", "graphsage_input", "data-class_map.json")),
        data_id_map = ancient(os.path.join(CURRENT_PATH, "data", "graphsage_input", "data-id_map.json")),
        data_feats = ancient(os.path.join(CURRENT_PATH, "data", "graphsage_input", "data-feats.npy")),
        data_walk = ancient(os.path.join(CURRENT_PATH, "data", "graphsage_input", "data-walks.txt"))
    output:
        os.path.join(CURRENT_PATH, "unsup-graphsage_input", "graphsage_mean_big_0.001000", "val.npy"),
        os.path.join(CURRENT_PATH, "unsup-graphsage_input", "graphsage_mean_big_0.001000", "val.txt")
    params:
        python27_path = "~/miniconda3/envs/graphsage_p2.7env/bin/python",
        train_prefix = os.path.join(CURRENT_PATH, "data", "graphsage_input", "data"),
        model_size = "big",
        learning_rate = 0.001,
        sample_size = 25,
        dim_size = 128,
        model_type = "graphsage_mean",
        max_total_steps = 100000,
        validate_iter = 1000,
        batch_size = 512,
        max_degree = 25
    shell:
        """
        {params.python27_path} -m graphsage.unsupervised_train --train_prefix {params.train_prefix} \
                                              --model_size {params.model_size} \
                                              --learning_rate {params.learning_rate} \
                                              --samples_1 {params.sample_size} \
                                              --samples_2 {params.sample_size} \
                                              --dim_1 {params.dim_size} \
                                              --dim_2 {params.dim_size} \
                                              --model {params.model_type} \
                                              --max_total_steps {params.max_total_steps} \
                                              --validate_iter {params.validate_iter} \
                                              --batch_size {params.batch_size} \
                                              --max_degree {params.max_degree}

        """

rule step16_transform_format:
    input:
        script = ancient(os.path.join(CURRENT_PATH, "scripts", "transform_format.py")),
        val_npy = ancient(os.path.join(CURRENT_PATH, "unsup-graphsage_input", "graphsage_mean_big_0.001000", "val.npy")),
        val_txt = ancient(os.path.join(CURRENT_PATH, "unsup-graphsage_input", "graphsage_mean_big_0.001000", "val.txt")),
        data_dir = ancient(os.path.join(CURRENT_PATH, "data"))
    output:
        os.path.join(CURRENT_PATH, "data", "graphsage_output", "unsuprvised_graphsage_entity_embeddings.pkl")
    params:
        graphsage_result = ancient(os.path.join(CURRENT_PATH, "unsup-graphsage_input", "graphsage_mean_big_0.001000"))
    shell:
        """
        python {input.script} --data_dir {input.data_dir} \
                              --input {params.graphsage_result}
        """


rule step17_pretrain_RF_model:
    input:
        script = ancient(os.path.join(CURRENT_PATH, "scripts", "run_RF_model_3class.py")),
        train_pairs = ancient(os.path.join(CURRENT_PATH, "data", "pretrain_reward_shaping_model_train_val_test_random_data_3class", "train_pairs.txt")),
        val_pairs = ancient(os.path.join(CURRENT_PATH, "data", "pretrain_reward_shaping_model_train_val_test_random_data_3class", "val_pairs.txt")),
        test_pairs = ancient(os.path.join(CURRENT_PATH, "data", "pretrain_reward_shaping_model_train_val_test_random_data_3class", "test_pairs.txt")),
        random_pairs = ancient(os.path.join(CURRENT_PATH, "data", "pretrain_reward_shaping_model_train_val_test_random_data_3class", "random_pairs.txt")),
        unsuprvised_graphsage_entity_embeddings = ancient(os.path.join(CURRENT_PATH, "data", "graphsage_output", "unsuprvised_graphsage_entity_embeddings.pkl")),
        data_dir = ancient(os.path.join(CURRENT_PATH, "data"))
    output:
        os.path.join(CURRENT_PATH, "models", "RF_model_3class", "RF_model.pt")
    params:
        pair_emb_method = 'concatenate',
        output_folder = os.path.join(CURRENT_PATH, "models"),
        seed = config['MODELINFO']['PARAMS']['SEED']
    shell:
        """
        python {input.script} --data_dir {input.data_dir} \
                              --pair_emb {params.pair_emb_method} \
                              --seed {params.seed} \
                              --output_folder {params.output_folder}
        """

rule step18_generate_expert_path_transition:
    input:
        script = ancient(os.path.join(CURRENT_PATH, "scripts", "generate_expert_path_transition.py")),
        path_file = ancient(os.path.join(CURRENT_PATH, "data", "expert_path_files", f"train_expert_demonstration_relation_entity_max{config['MODELINFO']['PARAMS']['MAX_PATH']}_filtered.pkl",)),
        data_dir = ancient(os.path.join(CURRENT_PATH, "data"))
    output:
        os.path.join(CURRENT_PATH, "data", "expert_path_files", f"train_expert_transitions_history{config['MODELINFO']['PARAMS']['STATE_HISTORY']}.pkl")
    params:
        path_file_name = f"train_expert_demonstration_relation_entity_max{config['MODELINFO']['PARAMS']['MAX_PATH']}_filtered.pkl",
        max_path = config['MODELINFO']['PARAMS']['MAX_PATH'],
        state_history = config['MODELINFO']['PARAMS']['STATE_HISTORY'],
        expert_trains_file_name = f"train_expert_transitions_history{config['MODELINFO']['PARAMS']['STATE_HISTORY']}.pkl"
    shell:
        """
        python {input.script} --data_dir {input.data_dir} \
                              --path_file_name {params.path_file_name} \
                              --max_path {params.max_path} \
                              --state_history {params.state_history} \
                              --expert_trains_file_name {params.expert_trains_file_name}
        """

rule step19_pretrain_ac_model:
    input:
        script = ancient(os.path.join(CURRENT_PATH, "scripts", "run_pretrain_ac_model.py")),
        data_dir = ancient(os.path.join(CURRENT_PATH, "data")),
        pretrained_RF_model = ancient(os.path.join(CURRENT_PATH, "models", "RF_model_3class", "RF_model.pt")),
        path_file = ancient(os.path.join(CURRENT_PATH, "data", "expert_path_files", f"train_expert_demonstration_relation_entity_max{config['MODELINFO']['PARAMS']['MAX_PATH']}_filtered.pkl",)),
        text_emb_file = ancient(os.path.join(CURRENT_PATH, "data", "text_embedding", "embedding_biobert_namecat.pkl"))
    output:
        os.path.join(CURRENT_PATH, "models", "pretrain_AC_model", "pretrained_ac_model.pt")
    params:
        path_file_name = f"train_expert_demonstration_relation_entity_max{config['MODELINFO']['PARAMS']['MAX_PATH']}_filtered.pkl",
        text_emb_file_name = "embedding_biobert_namecat.pkl",
        output_folder = os.path.join(CURRENT_PATH, "models"),
        max_path = config['MODELINFO']['PARAMS']['MAX_PATH'],
        max_pre_path = 10000000,
        bandwidth = config['MODELINFO']['PARAMS']['BANDWIDTH'],
        bucket_interval = config['MODELINFO']['PARAMS']['BUCKET_INTERVAL'],
        state_history = config['MODELINFO']['PARAMS']['STATE_HISTORY'],
        seed = config['MODELINFO']['PARAMS']['SEED'],
        gpu = config['MODELINFO']['PARAMS']['GPU'],
        batch_size = 1024,
        epochs = 20,
        pre_actor_epoch = 10,
        lr = config["MODELINFO"]['PARAMS']['LEARNING_RATE']
    shell:
        """
        python {input.script} --data_dir {input.data_dir} \
                              --path_file_name {params.path_file_name} \
                              --text_emb_file_name {params.text_emb_file_name} \
                              --output_folder {params.output_folder} \
                              --max_path {params.max_path} \
                              --max_pre_path {params.max_pre_path} \
                              --bandwidth {params.bandwidth} \
                              --bucket_interval {params.bucket_interval} \
                              --state_history {params.state_history} \
                              --pretrain_model_path {input.pretrained_RF_model} \
                              --seed {params.seed} \
                              --use_gpu \
                              --gpu {params.gpu} \
                              --batch_size {params.batch_size} \
                              --epochs {params.epochs} \
                              --pre_actor_epoch {params.pre_actor_epoch} \
                              --lr {params.lr}
        """


rule step20_train_adac_model:
    input:
        script = ancient(os.path.join(CURRENT_PATH, "scripts", "run_adac_model.py")),
        data_dir = ancient(os.path.join(CURRENT_PATH, "data")),
        pretrained_RF_model = ancient(os.path.join(CURRENT_PATH, "models", "RF_model_3class", "RF_model.pt")),
        pre_ac_file = ancient(os.path.join(CURRENT_PATH, "models", "pretrain_AC_model", "pretrained_ac_model.pt")),
        path_file = ancient(os.path.join(CURRENT_PATH, "data", "expert_path_files", f"train_expert_demonstration_relation_entity_max{config['MODELINFO']['PARAMS']['MAX_PATH']}_filtered.pkl",)),
        text_emb_file = ancient(os.path.join(CURRENT_PATH, "data", "text_embedding", "embedding_biobert_namecat.pkl")),
        path_trans_file = ancient(os.path.join(CURRENT_PATH, "data", "expert_path_files", f"train_expert_transitions_history{config['MODELINFO']['PARAMS']['STATE_HISTORY']}.pkl"))
    output:
        os.path.join(CURRENT_PATH, "models", "ADAC_model", "policy_net", "policy_model_epoch51.pt")
    params:
        path_file_name = f"train_expert_demonstration_relation_entity_max{config['MODELINFO']['PARAMS']['MAX_PATH']}_filtered.pkl",
        text_emb_file_name = "embedding_biobert_namecat.pkl",
        path_trans_file_name = f"train_expert_transitions_history{config['MODELINFO']['PARAMS']['STATE_HISTORY']}.pkl",
        output_folder = os.path.join(CURRENT_PATH, "models"),
        max_path = config['MODELINFO']['PARAMS']['MAX_PATH'],
        bandwidth = config['MODELINFO']['PARAMS']['BANDWIDTH'],
        bucket_interval = config['MODELINFO']['PARAMS']['BUCKET_INTERVAL'],
        gpu = config['MODELINFO']['PARAMS']['GPU'],
        epochs = 100,
        train_batch_size = 1120,
        state_history = config['MODELINFO']['PARAMS']['STATE_HISTORY'],
        ac_update_delay = 50,
        entropy_weight=0.005,
        disc_alpha=0.006,
        metadisc_alpha=0.012,
        num_rollouts=35,
        act_dropout=0.5,
        ac_lr= config["MODELINFO"]['PARAMS']['LEARNING_RATE'],
        disc_lr=config["MODELINFO"]['PARAMS']['LEARNING_RATE'],
        metadisc_lr=config["MODELINFO"]['PARAMS']['LEARNING_RATE']
    shell:
        """
        python {input.script} --data_dir {input.data_dir} \
                              --path_file_name {params.path_file_name} \
                              --text_emb_file_name {params.text_emb_file_name} \
                              --path_trans_file_name {params.path_trans_file_name} \
                              --output_folder {params.output_folder} \
                              --max_path {params.max_path} \
                              --bandwidth {params.bandwidth} \
                              --bucket_interval {params.bucket_interval} \
                              --pretrain_model_path {input.pretrained_RF_model} \
                              --use_gpu \
                              --gpu {params.gpu} \
                              --train_batch_size {params.train_batch_size} \
                              --warmup \
                              --pre_ac_file {input.pre_ac_file} \
                              --epochs {params.epochs} \
                              --state_history {params.state_history} \
                              --ac_update_delay {params.ac_update_delay} \
                              --ent_weight {params.entropy_weight} \
                              --disc_alpha {params.disc_alpha} \
                              --metadisc_alpha {params.metadisc_alpha} \
                                --num_rollouts {params.num_rollouts} \
                                --act_dropout {params.act_dropout} \
                                --ac_lr {params.ac_lr} \
                                --disc_lr {params.disc_lr} \
                                --metadisc_lr {params.metadisc_lr}                              
        """

rule step21_select_best_model:
    input:
        script = ancient(os.path.join(CURRENT_PATH, "scripts", "select_best_moa_model.py")),
        data_dir = ancient(os.path.join(CURRENT_PATH, "data")),
        policy_net_folder_check = ancient(os.path.join(CURRENT_PATH, "models", "ADAC_model", "policy_net", "policy_model_epoch51.pt")),
        pretrained_RF_model = ancient(os.path.join(CURRENT_PATH, "models", "RF_model_3class", "RF_model.pt")),
    output:
        os.path.join(CURRENT_PATH, "models", "ADAC_model", "policy_net", "best_moa_model.pt")
    params:
        policy_net_folder = ancient(os.path.join(CURRENT_PATH, "models", "ADAC_model", "policy_net")),
        max_path = config['MODELINFO']['PARAMS']['MAX_PATH'],
        bandwidth = config['MODELINFO']['PARAMS']['BANDWIDTH'],
        bucket_interval = config['MODELINFO']['PARAMS']['BUCKET_INTERVAL'],
        state_history = config['MODELINFO']['PARAMS']['STATE_HISTORY'],
        act_dropout=0.5,
        seed = config['MODELINFO']['PARAMS']['SEED'],
        factor = 0.9,
        topk = 50,
        eval_batch_size = 5,
        gpu = config['MODELINFO']['PARAMS']['GPU']
    shell:
        """
        python {input.script} --data_dir {input.data_dir} \
                              --policy_net_folder {params.policy_net_folder} \
                              --max_path {params.max_path} \
                              --bandwidth {params.bandwidth} \
                              --bucket_interval {params.bucket_interval} \
                              --state_history {params.state_history} \
                              --act_dropout {params.act_dropout} \
                              --seed {params.seed} \
                              --factor {params.factor} \
                              --topk {params.topk} \
                              --eval_batch_size {params.eval_batch_size} \
                              --pretrain_model_path {input.pretrained_RF_model} \
                              --use_gpu \
                              --gpu {params.gpu} \
                              --save_pred_paths
        """

rule step22_split_disease_into_K_pieces:
    input:
        script = ancient(os.path.join(CURRENT_PATH, "scripts", "split_disease_into_K_pieces.py")),
        data_dir = ancient(os.path.join(CURRENT_PATH, "data")),
        entity2freq = ancient(os.path.join(CURRENT_PATH, "data", 'entity2freq.txt')),
        relation2freq = ancient(os.path.join(CURRENT_PATH, "data", 'relation2freq.txt')),
        type2freq = ancient(os.path.join(CURRENT_PATH, "data", 'type2freq.txt')),
        entity2typeid = ancient(os.path.join(CURRENT_PATH, "data", 'entity2typeid.pkl')),
        all_node_info = ancient(os.path.join(CURRENT_PATH, "data", 'all_graph_nodes_info.txt')),
        db_configfile = ancient(os.path.join(CURRENT_PATH, config['RTXINFO']['DB_CONFIGFILE']))
    output:
        os.path.join(CURRENT_PATH, "data", "disease_sets", "disease_set1.txt"),
        os.path.join(CURRENT_PATH, "data", "filtered_drug_nodes_for_precomputation.pkl"),
    params:
        K = config['PARALLEL_PRECOMPUTE']['K'],
        out_dir = os.path.join(CURRENT_PATH, "data", "disease_sets")
    shell:
        """
        python {input.script} --data_dir {input.data_dir} \
                              --K {params.K} \
                              --db_config_path {input.db_configfile}
        """

rule step23_precompute_all_drug_disease_pairs_in_parallel:
    input:
        script = ancient(os.path.join(CURRENT_PATH, "scripts", "run_xDTD.py")),
        data_dir = ancient(os.path.join(CURRENT_PATH, "data")),
        ddp_model = ancient(os.path.join(CURRENT_PATH, "models", "RF_model_3class", "RF_model.pt")),
        moa_model = ancient(os.path.join(CURRENT_PATH, "models", "ADAC_model", "policy_net", "best_moa_model.pt")),
        disease_set1 = ancient(os.path.join(CURRENT_PATH, "data", "disease_sets", "disease_set1.txt")),
        disease_set2 = ancient(os.path.join(CURRENT_PATH, "data", "filtered_drug_nodes_for_precomputation.pkl")),
        model_dir = ancient(os.path.join(CURRENT_PATH, "models"))
    output:
        # os.path.join(CURRENT_PATH, "results", "path_results"),
        # os.path.join(CURRENT_PATH, "results", "prediction_scores"),
        touch(os.path.join(CURRENT_PATH, "results", "step23_done.txt"))
    params:
        out_dir = os.path.join(CURRENT_PATH, 'results'),
        K = config['PARALLEL_PRECOMPUTE']['K'],
        N_drugs = config['PARALLEL_PRECOMPUTE']['N_drugs'],
        N_paths = config['PARALLEL_PRECOMPUTE']['N_paths'],
        batch_size = config['PARALLEL_PRECOMPUTE']['BATCH_SIZE'],
        max_path = config['MODELINFO']['PARAMS']['MAX_PATH'],
        bandwidth = config['MODELINFO']['PARAMS']['BANDWIDTH'],
        bucket_interval = config['MODELINFO']['PARAMS']['BUCKET_INTERVAL'],
        state_history = config['MODELINFO']['PARAMS']['STATE_HISTORY'],
        threshold = 0.3
    run:
        for index in range(int(params.K)):
            shell(f"nohup python {input.script} --log_name run_xDTD_{index+1}.log \
                              --data_path {input.data_dir} \
                              --model_path {input.model_dir} \
                              --disease_set {input.data_dir}/disease_sets/disease_set{index+1}.txt \
                              --out_dir {params.out_dir} \
                              --N_drugs {params.N_drugs} \
                              --N_paths {params.N_paths} \
                              --batch_size {params.batch_size} \
                              --max_path {params.max_path} \
                              --bandwidth {params.bandwidth} \
                              --bucket_interval {params.bucket_interval} \
                              --state_history {params.state_history} \
                              --threshold {params.threshold} &")

# rule step24_build_sql_database:
#     input:
#         script = ancient(os.path.join(CURRENT_PATH, "scripts", "build_sql_database.py")),
#         unused_file = ancient(os.path.join(CURRENT_PATH, "results", "step23_done.txt"))
#     output:
#         os.path.join(CURRENT_PATH, config['DATABASE']['DATABASE_NAME'])
#     params:
#         path_to_score_results = os.path.join(CURRENT_PATH, "results", "path_results"),
#         path_to_path_results = os.path.join(CURRENT_PATH, "results", "prediction_scores"),
#         database_name = config['DATABASE']['DATABASE_NAME'],
#         outdir = CURRENT_PATH
#     shell:
#         """
#         python {input.script} --build \
#                               --path_to_score_results {params.path_to_score_results} \
#                               --path_to_path_results {params.path_to_path_results} \
#                               --database_name {params.database_name} \
#                               --outdir {params.outdir}
#         """

# rule step26_build_mapping_database:
#     input:
#         script = ancient(os.path.join(CURRENT_PATH, "scripts", "build_mapping_db.py")),
#         db_configfile = ancient(config['RTXINFO']['DB_CONFIGFILE']),
#         kgml_xdtd_data_entity2freq_unused = ancient(os.path.join(CURRENT_PATH, "data", "entity2freq.txt")),
#         kgml_xdtd_data_graph_edges_unused = ancient(os.path.join(CURRENT_PATH, "data", "graph_edges.txt")),
#         unused_file = ancient(os.path.join(CURRENT_PATH, "results", "step23_done.txt")),
#         database_name = ancient(os.path.join(CURRENT_PATH, config['DATABASE']['DATABASE_NAME']))
#     output:
#         touch(os.path.join(CURRENT_PATH, "results", "step25_done.txt"))
#     params:
#         outdir = CURRENT_PATH,
#         tsv_path = ancient(os.path.join(CURRENT_PATH, "kg2c-tsv")),
#         kgml_xdtd_data_path = ancient(os.path.join(CURRENT_PATH, "data"))
#     shell:
#         """
#         python {input.script} --build \
#                               --db_config_path {input.db_configfile} \
#                               --tsv_path ${params.tsv_path} \
#                               --kgml_xdtd_data_path ${params.kgml_xdtd_data_path} \
#                               --database_name ${input.database_name} \
#                               --outdir {params.outdir}
#         """
