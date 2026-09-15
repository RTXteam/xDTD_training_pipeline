"""
This file is a SnakeMake Script to automate the xDTD model training

Pipeline steps (22 total):
  Steps 1-6:   Data download, KG processing, drug/disease lists, GT, preprocessing
  Steps 7-10:  DrugBank expert paths, reachability, expert demonstration paths
  Step  11:    Train/test split (90/10)
  Step  12:    Node2Vec embedding generation
  Step  13:    XGBoost ensemble training
  Steps 14-17: RL pipeline (expert transitions, pretrain AC, ADAC, best model)
  Steps 18-22: Precompute predictions, build SQL database

Usage:
    snakemake --cores 16 -s Run_Pipeline.smk targets
"""
## Import Config Files
configfile: "./config.yaml"

## Import Python standard libraries
import os, sys

## Define Global Variables
CURRENT_PATH = os.getcwd()
_MAX_PATH = str(config['MODELINFO']['PARAMS']['MAX_PATH'])
_STATE_HISTORY = str(config['MODELINFO']['PARAMS']['STATE_HISTORY'])

## Create Required Folders
for _d in ["data", "data/ground_truth_pairs", "data/drug_disease_list",
           "data/node2vec_output", "data/text_embedding",
           "data/kg_init_embeddings", "data/expert_path_files",
           "data/pretrain_reward_shaping_model_train_val_test_data_3class",
           "log_folder", "models", "results", "reports", "reports/figures"]:
    os.makedirs(os.path.join(CURRENT_PATH, _d), exist_ok=True)


## ═══════════════════════════════════════════════════════════════════════════════
## Final target
## ═══════════════════════════════════════════════════════════════════════════════
rule targets:
    input:
        # ── Data downloads ──────────────────────────────────────
        ancient(os.path.join(CURRENT_PATH, "data", config['DRUGMECHDBINFO']['DRUGMECHDB_PATH'])),
        ancient(os.path.join(CURRENT_PATH, "data", config['EXTERNAL_DATA']['DRUGBANK_XML'])),
        ancient(os.path.join(CURRENT_PATH, "data", config['TRANSLATOR_KG']['NODES_JSONL'])),
        ancient(os.path.join(CURRENT_PATH, "data", config['TRANSLATOR_KG']['EDGES_JSONL'])),
        ancient(os.path.join(CURRENT_PATH, "data", config['DRUG_DISEASE_LIST']['DIR'], config['DRUG_DISEASE_LIST']['DRUG_FILE'])),
        ancient(os.path.join(CURRENT_PATH, "data", config['DRUG_DISEASE_LIST']['DIR'], config['DRUG_DISEASE_LIST']['DISEASE_FILE'])),
        # ── KG processing ──────────────────────────────────────
        ancient(os.path.join(CURRENT_PATH, "data", 'graph_edges.txt')),
        ancient(os.path.join(CURRENT_PATH, "data", 'all_graph_nodes_info.txt')),
        ancient(os.path.join(CURRENT_PATH, "data", 'filtered_graph_edges.txt')),
        ancient(os.path.join(CURRENT_PATH, "data", 'filtered_graph_nodes_info.txt')),
        # ── Drug/disease lists & ground truth ──────────────────
        ancient(os.path.join(CURRENT_PATH, "data", config['DRUG_DISEASE_LIST']['DIR'], 'drug_list.txt')),
        ancient(os.path.join(CURRENT_PATH, "data", config['DRUG_DISEASE_LIST']['DIR'], 'disease_list.txt')),
        ancient(os.path.join(CURRENT_PATH, "data", config['GROUND_TRUTH_PAIRS']['DIR'], 'tp_pairs.txt')),
        ancient(os.path.join(CURRENT_PATH, "data", config['GROUND_TRUTH_PAIRS']['DIR'], 'tn_pairs.txt')),
        # ── Preprocessing ──────────────────────────────────────
        ancient(os.path.join(CURRENT_PATH, "data", 'entity2freq.txt')),
        ancient(os.path.join(CURRENT_PATH, "data", 'relation2freq.txt')),
        ancient(os.path.join(CURRENT_PATH, "data", 'type2freq.txt')),
        ancient(os.path.join(CURRENT_PATH, "data", 'adj_list.pkl')),
        ancient(os.path.join(CURRENT_PATH, "data", 'entity2typeid.pkl')),
        ancient(os.path.join(CURRENT_PATH, "data", 'kg.pgrk')),
        # ── Expert paths ───────────────────────────────────────
        ancient(os.path.join(CURRENT_PATH, "data", 'expert_path_files', 'drugbank_dict.pkl')),
        ancient(os.path.join(CURRENT_PATH, "data", 'expert_path_files', 'drugbank_mapping.txt')),
        ancient(os.path.join(CURRENT_PATH, "data", 'expert_path_files', 'p_expert_paths.txt')),
        ancient(os.path.join(CURRENT_PATH, "data", 'expert_path_files', 'all_drugs.txt')),
        ancient(os.path.join(CURRENT_PATH, "data", 'expert_path_files', 'p_expert_paths_combined.txt')),
        ancient(os.path.join(CURRENT_PATH, "data", 'expert_path_files', "reachable_expert_paths_max" + _MAX_PATH + ".txt")),
        ancient(os.path.join(CURRENT_PATH, "data", 'expert_path_files', "reachable_tp_pairs_max" + _MAX_PATH + ".txt")),
        ancient(os.path.join(CURRENT_PATH, "data", 'expert_path_files', "unreachable_tp_pairs_max" + _MAX_PATH + ".txt")),
        ancient(os.path.join(CURRENT_PATH, "data", 'expert_path_files', "expert_demonstration_paths_max" + _MAX_PATH + "_raw.pkl")),
        ancient(os.path.join(CURRENT_PATH, "data", 'expert_path_files', "expert_demonstration_paths_max" + _MAX_PATH + "_filtered.pkl")),
        ancient(os.path.join(CURRENT_PATH, "data", 'expert_path_files', "expert_demonstration_paths_translate_max" + _MAX_PATH + "_filtered.pkl")),
        ancient(os.path.join(CURRENT_PATH, "data", 'expert_path_files', "expert_demonstration_relation_entity_max" + _MAX_PATH + "_filtered.pkl")),
        # ── Train/test split ───────────────────────────────────
        ancient(os.path.join(CURRENT_PATH, "data", "pretrain_reward_shaping_model_train_val_test_data_3class", "train_pairs.txt")),
        ancient(os.path.join(CURRENT_PATH, "data", "pretrain_reward_shaping_model_train_val_test_data_3class", "test_pairs.txt")),
        ancient(os.path.join(CURRENT_PATH, "data", 'expert_path_files', "train_expert_demonstration_relation_entity_max" + _MAX_PATH + "_filtered.pkl")),
        ancient(os.path.join(CURRENT_PATH, "data", 'expert_path_files', "test_expert_demonstration_relation_entity_max" + _MAX_PATH + "_filtered.pkl")),
        # ── Node2Vec embeddings ────────────────────────────────
        ancient(os.path.join(CURRENT_PATH, "data", "node2vec_output", "node2vec_entity_embeddings.pkl")),
        # ── XGBoost ensemble model ─────────────────────────────
        ancient(os.path.join(CURRENT_PATH, "models", "xgboost_model_3class", "xgboost_model.pt")),
        # ── RL pipeline ────────────────────────────────────────
        ancient(os.path.join(CURRENT_PATH, "data", "expert_path_files", "train_expert_transitions_history" + _STATE_HISTORY + ".pkl")),
        ancient(os.path.join(CURRENT_PATH, "models", "pretrain_AC_model", "pretrained_ac_model.pt")),
        ancient(os.path.join(CURRENT_PATH, "models", "ADAC_model", "policy_net", "step17_training_done.flag")),
        ancient(os.path.join(CURRENT_PATH, "models", "ADAC_model", "policy_net", "best_moa_model.pt")),
        # ── Precompute & database ──────────────────────────────
        ancient(os.path.join(CURRENT_PATH, "data", "disease_sets", "disease_set1.txt")),
        ancient(os.path.join(CURRENT_PATH, "data", "filtered_drug_nodes_for_precomputation.pkl")),
        ancient(os.path.join(CURRENT_PATH, "results", "step19_done.txt")),
        ancient(os.path.join(CURRENT_PATH, config['DATABASE']['DATABASE_NAME'])),
        ancient(os.path.join(CURRENT_PATH, "results", "step22_done.txt"))


## ═══════════════════════════════════════════════════════════════════════════════
## Steps 1-6: Data download, KG processing, GT, preprocessing
## ═══════════════════════════════════════════════════════════════════════════════

rule step1_download_data:
    """Download DrugMechDB, Translator KG, and drug/disease lists."""
    output:
        os.path.join(CURRENT_PATH, "data", config['DRUGMECHDBINFO']['DRUGMECHDB_PATH']),
        os.path.join(CURRENT_PATH, "data", config['TRANSLATOR_KG']['NODES_JSONL']),
        os.path.join(CURRENT_PATH, "data", config['TRANSLATOR_KG']['EDGES_JSONL']),
        os.path.join(CURRENT_PATH, "data", config['DRUG_DISEASE_LIST']['DIR'], config['DRUG_DISEASE_LIST']['DRUG_FILE']),
        os.path.join(CURRENT_PATH, "data", config['DRUG_DISEASE_LIST']['DIR'], config['DRUG_DISEASE_LIST']['DISEASE_FILE'])
    params:
        drugmechdb_link = config['DRUGMECHDBINFO']['LINK'],
        drugmechdb_path = config['DRUGMECHDBINFO']['DRUGMECHDB_PATH'],
        translator_kg_url = config['TRANSLATOR_KG']['DOWNLOAD_URL'],
        dd_dir = config['DRUG_DISEASE_LIST']['DIR'],
        drug_ds = config['DRUG_DISEASE_LIST']['DRUG_DATASET'],
        drug_file = config['DRUG_DISEASE_LIST']['DRUG_FILE'],
        disease_ds = config['DRUG_DISEASE_LIST']['DISEASE_DATASET'],
        disease_file = config['DRUG_DISEASE_LIST']['DISEASE_FILE']
    run:
        # ── Download DrugMechDB ───────────────────────────────────────
        shell("curl {params.drugmechdb_link}/{params.drugmechdb_path} -o ./data/{params.drugmechdb_path}"),
        # ── Download Translator KG ────────────────────────────────────
        shell("mkdir -p ./data/translator_kg"),
        shell("wget -O ./data/translator_kg.tar.zst {params.translator_kg_url}"),
        shell("tar -I zstd -xf ./data/translator_kg.tar.zst -C ./data/translator_kg/"),
        shell("rm -f ./data/translator_kg.tar.zst"),
        # ── Download Drug-Disease Lists from HuggingFace ──────────────
        import subprocess, sys
        dd_dir = os.path.join("data", config['DRUG_DISEASE_LIST']['DIR'])
        os.makedirs(dd_dir, exist_ok=True)
        _dl_script = "import sys; from datasets import load_dataset; load_dataset(sys.argv[1])['train'].to_csv(sys.argv[2], sep='\\t', index=False)"
        subprocess.run([sys.executable, "-c", _dl_script, config['DRUG_DISEASE_LIST']['DRUG_DATASET'], os.path.join(dd_dir, config['DRUG_DISEASE_LIST']['DRUG_FILE'])], check=True)
        subprocess.run([sys.executable, "-c", _dl_script, config['DRUG_DISEASE_LIST']['DISEASE_DATASET'], os.path.join(dd_dir, config['DRUG_DISEASE_LIST']['DISEASE_FILE'])], check=True)

rule step2_process_translator_kg:
    """Parse raw KG JSONL → graph_edges.txt, all_graph_nodes_info.txt."""
    input:
        script = ancient(os.path.join(CURRENT_PATH, "scripts", "process_translator_kg.py")),
        nodes_jsonl = ancient(os.path.join(CURRENT_PATH, "data", config['TRANSLATOR_KG']['NODES_JSONL'])),
        edges_jsonl = ancient(os.path.join(CURRENT_PATH, "data", config['TRANSLATOR_KG']['EDGES_JSONL']))
    output:
        os.path.join(CURRENT_PATH, "data", 'graph_edges.txt'),
        os.path.join(CURRENT_PATH, "data", 'all_graph_nodes_info.txt'),
        os.path.join(CURRENT_PATH, "data", "biolink_version.txt")
    shell:
        """
        python {input.script} --nodes_jsonl {input.nodes_jsonl} \
                              --edges_jsonl {input.edges_jsonl}
        """

rule step3_filter_graph_nodes_and_edges:
    """Filter graph by category and optionally remove edges from specified knowledge sources."""
    input:
        script = ancient(os.path.join(CURRENT_PATH, "scripts", "filter_kg2_nodes_and_edges.py")),
        graph_edges = ancient(os.path.join(CURRENT_PATH, "data", 'graph_edges.txt')),
        all_node_info = ancient(os.path.join(CURRENT_PATH, "data", 'all_graph_nodes_info.txt')),
        biolink_version = ancient(os.path.join(CURRENT_PATH, "data", "biolink_version.txt"))
    output:
        os.path.join(CURRENT_PATH, "data", 'filtered_graph_edges.txt'),
        os.path.join(CURRENT_PATH, "data", 'filtered_graph_nodes_info.txt')
    params:
        remove_sources = config['KGINFO'].get('REMOVE_KNOWLEDGE_SOURCES', [])
    shell:
        """
        biolink_ver=$(cat {CURRENT_PATH}/data/biolink_version.txt) && python {input.script} --graph_nodes {input.all_node_info} \
                              --graph_edges {input.graph_edges} \
                              --biolink_version $biolink_ver \
                              --remove_knowledge_sources {params.remove_sources}
        """

rule step4_process_drug_disease_list:
    """Filter drug/disease lists to entities present in the KG."""
    input:
        script = ancient(os.path.join(CURRENT_PATH, "scripts", "process_drug_disease_list.py")),
        drug_list = ancient(os.path.join(CURRENT_PATH, "data", config['DRUG_DISEASE_LIST']['DIR'], config['DRUG_DISEASE_LIST']['DRUG_FILE'])),
        disease_list = ancient(os.path.join(CURRENT_PATH, "data", config['DRUG_DISEASE_LIST']['DIR'], config['DRUG_DISEASE_LIST']['DISEASE_FILE'])),
        graph_nodes = ancient(os.path.join(CURRENT_PATH, "data", 'filtered_graph_nodes_info.txt')),
        biolink_version = ancient(os.path.join(CURRENT_PATH, "data", "biolink_version.txt"))
    output:
        os.path.join(CURRENT_PATH, "data", config['DRUG_DISEASE_LIST']['DIR'], 'drug_list.txt'),
        os.path.join(CURRENT_PATH, "data", config['DRUG_DISEASE_LIST']['DIR'], 'disease_list.txt')
    shell:
        """
        biolink_ver=$(cat {CURRENT_PATH}/data/biolink_version.txt) && python {input.script} --drug_list {input.drug_list} \
                              --disease_list {input.disease_list} \
                              --graph_nodes {input.graph_nodes} \
                              --biolink_version $biolink_ver
        """

rule step5_convert_ground_truth:
    """Convert EC gt_pairs_raw.tsv → tp_pairs.txt / tn_pairs.txt via Node Norm normalization."""
    input:
        script = ancient(os.path.join(CURRENT_PATH, "scripts", "convert_ec_ground_truth.py")),
        ec_gt_file = ancient(os.path.join(CURRENT_PATH, "data", config['GROUND_TRUTH_PAIRS']['GT_RAW_FILE'])),
        drug_list = ancient(os.path.join(CURRENT_PATH, "data", config['DRUG_DISEASE_LIST']['DIR'], 'drug_list.txt')),
        disease_list = ancient(os.path.join(CURRENT_PATH, "data", config['DRUG_DISEASE_LIST']['DIR'], 'disease_list.txt')),
        graph_nodes = ancient(os.path.join(CURRENT_PATH, "data", 'filtered_graph_nodes_info.txt')),
        biolink_version = ancient(os.path.join(CURRENT_PATH, "data", "biolink_version.txt"))
    output:
        os.path.join(CURRENT_PATH, "data", config['GROUND_TRUTH_PAIRS']['DIR'], 'tp_pairs.txt'),
        os.path.join(CURRENT_PATH, "data", config['GROUND_TRUTH_PAIRS']['DIR'], 'tn_pairs.txt')
    shell:
        """
        biolink_ver=$(cat {CURRENT_PATH}/data/biolink_version.txt) && python {input.script} --ec_gt_file {input.ec_gt_file} \
                              --drug_list {input.drug_list} \
                              --disease_list {input.disease_list} \
                              --graph_nodes {input.graph_nodes} \
                              --biolink_version $biolink_ver \
                              --output_folder {CURRENT_PATH}/data/{config[GROUND_TRUTH_PAIRS][DIR]}
        """

rule step6_preprocess_data:
    """Generate entity/relation/type frequencies, adjacency list, PageRank scores."""
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


## ═══════════════════════════════════════════════════════════════════════════════
## Steps 7-10: DrugBank expert paths (needed for RL pipeline)
## ═══════════════════════════════════════════════════════════════════════════════

rule step7_process_drugbank_action_desc:
    """Parse drugbank.xml for drug-gene-action expert paths."""
    input:
        script = ancient(os.path.join(CURRENT_PATH, "scripts", "process_drugbank_action_desc.py")),
        nodes_jsonl = ancient(os.path.join(CURRENT_PATH, "data", config['TRANSLATOR_KG']['NODES_JSONL'])),
        drugbank_xml = ancient(os.path.join(CURRENT_PATH, "data", config['EXTERNAL_DATA']['DRUGBANK_XML']))
    output:
        os.path.join(CURRENT_PATH, "data", 'expert_path_files', 'drugbank_dict.pkl'),
        os.path.join(CURRENT_PATH, "data", 'expert_path_files', 'drugbank_mapping.txt'),
        os.path.join(CURRENT_PATH, "data", 'expert_path_files', 'p_expert_paths.txt')
    shell:
        """
        python {input.script} --nodes_jsonl {input.nodes_jsonl} --drugbankxml {input.drugbank_xml}
        """

rule step8_integrate_drugbank_and_molepro_data:
    """Combine DrugBank + MolePro expert paths."""
    input:
        script = ancient(os.path.join(CURRENT_PATH, "scripts", "integrate_drugbank_and_molepro_data.py")),
        nodes_jsonl = ancient(os.path.join(CURRENT_PATH, "data", config['TRANSLATOR_KG']['NODES_JSONL'])),
        drugbank_export_paths = ancient(os.path.join(CURRENT_PATH, "data", 'expert_path_files', 'p_expert_paths.txt'))
    output:
        os.path.join(CURRENT_PATH, "data", 'expert_path_files', 'all_drugs.txt'),
        os.path.join(CURRENT_PATH, "data", 'expert_path_files', 'p_expert_paths_combined.txt')
    shell:
        """
        python {input.script} --nodes_jsonl {input.nodes_jsonl} \
                              --drugbank_export_paths {input.drugbank_export_paths}
        """

rule step9_check_reachable:
    """Check 3-hop reachability of expert paths in the KG."""
    input:
        script = ancient(os.path.join(CURRENT_PATH, "scripts", "check_reachable.py")),
        true_pairs = ancient(os.path.join(CURRENT_PATH, "data", config['GROUND_TRUTH_PAIRS']['DIR'], 'tp_pairs.txt')),
        entity2freq = ancient(os.path.join(CURRENT_PATH, "data", 'entity2freq.txt')),
        relation2freq = ancient(os.path.join(CURRENT_PATH, "data", 'relation2freq.txt')),
        adj_list = ancient(os.path.join(CURRENT_PATH, "data", 'adj_list.pkl')),
        kg_pgrk = ancient(os.path.join(CURRENT_PATH, "data", 'kg.pgrk')),
        combined_expert_paths = ancient(os.path.join(CURRENT_PATH, "data", 'expert_path_files', 'p_expert_paths_combined.txt'))
    output:
        os.path.join(CURRENT_PATH, "data", 'expert_path_files', "reachable_expert_paths_max" + _MAX_PATH + ".txt"),
        os.path.join(CURRENT_PATH, "data", 'expert_path_files', "reachable_tp_pairs_max" + _MAX_PATH + ".txt"),
        os.path.join(CURRENT_PATH, "data", 'expert_path_files', "unreachable_tp_pairs_max" + _MAX_PATH + ".txt")
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
    """Generate expert demonstration paths for RL training."""
    input:
        script = ancient(os.path.join(CURRENT_PATH, "scripts", "generate_expert_paths.py")),
        reachable_expert_paths = ancient(os.path.join(CURRENT_PATH, "data", 'expert_path_files', "reachable_expert_paths_max" + _MAX_PATH + ".txt")),
        biolink_version = ancient(os.path.join(CURRENT_PATH, "data", "biolink_version.txt"))
    output:
        os.path.join(CURRENT_PATH, "data", 'expert_path_files', "expert_demonstration_paths_max" + _MAX_PATH + "_raw.pkl"),
        os.path.join(CURRENT_PATH, "data", 'expert_path_files', "expert_demonstration_paths_max" + _MAX_PATH + "_filtered.pkl"),
        os.path.join(CURRENT_PATH, "data", 'expert_path_files', "expert_demonstration_paths_translate_max" + _MAX_PATH + "_filtered.pkl"),
        os.path.join(CURRENT_PATH, "data", 'expert_path_files', "expert_demonstration_relation_entity_max" + _MAX_PATH + "_filtered.pkl")
    params:
        bandwidth = config['MODELINFO']['PARAMS']['BANDWIDTH'],
        max_path = config['MODELINFO']['PARAMS']['MAX_PATH'],
        process = config['MODELINFO']['PARAMS']['EXPERT_PATH_PROCESS'],
        batch_size = config['MODELINFO']['PARAMS']['EXPERT_PATH_BATCH_SIZE']
    shell:
        """
        biolink_ver=$(cat {CURRENT_PATH}/data/biolink_version.txt) && python {input.script} --biolink_version $biolink_ver \
                              --reachable_expert_paths {input.reachable_expert_paths} \
                              --bandwidth {params.bandwidth} \
                              --batch_size {params.batch_size} \
                              --process {params.process} \
                              --max_path {params.max_path}
        """


## ═══════════════════════════════════════════════════════════════════════════════
## Step 11: Train/test split (single 90/10 fold)
## ═══════════════════════════════════════════════════════════════════════════════

rule step11_split_data_train_test:
    """Split TP/TN pairs into train (90%) and test (10%), single fold.
    Also splits expert demonstration paths by train/test for RL pipeline."""
    input:
        script = ancient(os.path.join(CURRENT_PATH, "scripts", "split_data_train_test.py")),
        graph_edges = ancient(os.path.join(CURRENT_PATH, "data", 'filtered_graph_edges.txt')),
        tp_pairs = ancient(os.path.join(CURRENT_PATH, "data", config['GROUND_TRUTH_PAIRS']['DIR'], 'tp_pairs.txt')),
        tn_pairs = ancient(os.path.join(CURRENT_PATH, "data", config['GROUND_TRUTH_PAIRS']['DIR'], 'tn_pairs.txt')),
        entity2freq = ancient(os.path.join(CURRENT_PATH, "data", 'entity2freq.txt')),
        type2freq = ancient(os.path.join(CURRENT_PATH, "data", 'type2freq.txt')),
        entity2typeid = ancient(os.path.join(CURRENT_PATH, "data", 'entity2typeid.pkl')),
        filtered_expert_paths = ancient(os.path.join(CURRENT_PATH, "data", 'expert_path_files', "expert_demonstration_paths_max" + _MAX_PATH + "_filtered.pkl")),
        filtered_path_relation_entity = ancient(os.path.join(CURRENT_PATH, "data", 'expert_path_files', "expert_demonstration_relation_entity_max" + _MAX_PATH + "_filtered.pkl"))
    output:
        os.path.join(CURRENT_PATH, "data", "pretrain_reward_shaping_model_train_val_test_data_3class", "train_pairs.txt"),
        os.path.join(CURRENT_PATH, "data", "pretrain_reward_shaping_model_train_val_test_data_3class", "test_pairs.txt"),
        os.path.join(CURRENT_PATH, "data", 'expert_path_files', "train_expert_demonstration_relation_entity_max" + _MAX_PATH + "_filtered.pkl"),
        os.path.join(CURRENT_PATH, "data", 'expert_path_files', "test_expert_demonstration_relation_entity_max" + _MAX_PATH + "_filtered.pkl")
    params:
        test_size = config['SPLIT']['TEST_SIZE'],
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
                              --filtered_expert_paths {input.filtered_expert_paths} \
                              --filtered_path_relation_entity {input.filtered_path_relation_entity} \
                              --max_path {params.max_path} \
                              --test_size {params.test_size} \
                              --seed {params.seed} \
                              --output_folder {CURRENT_PATH}/data
        """


## ═══════════════════════════════════════════════════════════════════════════════
## Step 12: Node2Vec embedding generation
## ═══════════════════════════════════════════════════════════════════════════════

rule step12_generate_node2vec_embedding:
    """Generate Node2Vec entity embeddings (drug-disease edges removed to prevent leakage)."""
    input:
        script = ancient(os.path.join(CURRENT_PATH, "scripts", "generate_node2vec_embedding.py")),
        graph_nodes = ancient(os.path.join(CURRENT_PATH, "data", "filtered_graph_nodes_info.txt")),
        graph_edges = ancient(os.path.join(CURRENT_PATH, "data", "filtered_graph_edges.txt")),
        tp_pairs = ancient(os.path.join(CURRENT_PATH, "data", config['GROUND_TRUTH_PAIRS']['DIR'], 'tp_pairs.txt')),
        tn_pairs = ancient(os.path.join(CURRENT_PATH, "data", config['GROUND_TRUTH_PAIRS']['DIR'], 'tn_pairs.txt'))
    output:
        os.path.join(CURRENT_PATH, "data", "node2vec_output", "node2vec_entity_embeddings.pkl")
    params:
        embedding_dim = config['NODE2VEC']['EMBEDDING_DIM'],
        walk_length = config['NODE2VEC']['WALK_LENGTH'],
        walks_per_node = config['NODE2VEC']['WALKS_PER_NODE'],
        p = config['NODE2VEC']['P'],
        q = config['NODE2VEC']['Q'],
        iterations = config['NODE2VEC']['ITERATIONS'],
        window_size = config['NODE2VEC']['WINDOW_SIZE'],
        workers = config['NODE2VEC']['WORKERS'],
        seed = config['NODE2VEC']['SEED']
    shell:
        """
        python {input.script} --graph_nodes {input.graph_nodes} \
                              --graph_edges {input.graph_edges} \
                              --tp_pairs {input.tp_pairs} \
                              --tn_pairs {input.tn_pairs} \
                              --embedding_dim {params.embedding_dim} \
                              --walk_length {params.walk_length} \
                              --walks_per_node {params.walks_per_node} \
                              --p {params.p} \
                              --q {params.q} \
                              --iterations {params.iterations} \
                              --window_size {params.window_size} \
                              --workers {params.workers} \
                              --seed {params.seed} \
                              --output_folder {CURRENT_PATH}/data/node2vec_output
        """


## ═══════════════════════════════════════════════════════════════════════════════
## Step 13: XGBoost ensemble training
## ═══════════════════════════════════════════════════════════════════════════════

rule step13_train_xgboost_ensemble:
    """Train 3-shard XGBoost ensemble with replacement negatives and skopt GP HPO."""
    input:
        script = ancient(os.path.join(CURRENT_PATH, "scripts", "run_xgboost_ensemble_3class.py")),
        train_pairs = ancient(os.path.join(CURRENT_PATH, "data", "pretrain_reward_shaping_model_train_val_test_data_3class", "train_pairs.txt")),
        test_pairs = ancient(os.path.join(CURRENT_PATH, "data", "pretrain_reward_shaping_model_train_val_test_data_3class", "test_pairs.txt")),
        embeddings = ancient(os.path.join(CURRENT_PATH, "data", "node2vec_output", "node2vec_entity_embeddings.pkl")),
        drug_list = ancient(os.path.join(CURRENT_PATH, "data", config['DRUG_DISEASE_LIST']['DIR'], 'drug_list.txt')),
        disease_list = ancient(os.path.join(CURRENT_PATH, "data", config['DRUG_DISEASE_LIST']['DIR'], 'disease_list.txt'))
    output:
        os.path.join(CURRENT_PATH, "models", "xgboost_model_3class", "xgboost_model.pt")
    params:
        data_dir = os.path.join(CURRENT_PATH, "data"),
        splits_dir = os.path.join(CURRENT_PATH, "data", "pretrain_reward_shaping_model_train_val_test_data_3class"),
        pair_emb_method = 'concatenate',
        output_folder = os.path.join(CURRENT_PATH, "models"),
        seed = config['MODELINFO']['PARAMS']['SEED'],
        n_shards = config['ENSEMBLE']['N_SHARDS'],
        n_replacements = config['ENSEMBLE']['N_REPLACEMENTS'],
        n_calls = config['ENSEMBLE']['SKOPT_N_CALLS'],
        n_random_starts = config['ENSEMBLE']['SKOPT_N_RANDOM_STARTS'],
        hpo_inner_test_size = config['ENSEMBLE']['HPO_INNER_TEST_SIZE'],
        device = config['ENSEMBLE']['DEVICE'],
        early_stopping_rounds = config['ENSEMBLE']['EARLY_STOPPING_ROUNDS'],
        gpu_id = config['ENSEMBLE']['GPU_IDS'][0]
    shell:
        """
        export CUDA_VISIBLE_DEVICES={params.gpu_id}
        python {input.script} --data_dir {params.data_dir} \
                              --splits_dir {params.splits_dir} \
                              --pair_emb {params.pair_emb_method} \
                              --seed {params.seed} \
                              --n_shards {params.n_shards} \
                              --n_replacements {params.n_replacements} \
                              --n_calls {params.n_calls} \
                              --n_random_starts {params.n_random_starts} \
                              --hpo_inner_test_size {params.hpo_inner_test_size} \
                              --device {params.device} \
                              --early_stopping_rounds {params.early_stopping_rounds} \
                              --drug_list {input.drug_list} \
                              --disease_list {input.disease_list} \
                              --output_folder {params.output_folder}
        """


## ═══════════════════════════════════════════════════════════════════════════════
## Steps 14-17: RL pipeline (ADAC model training)
## ═══════════════════════════════════════════════════════════════════════════════

rule step14_generate_expert_path_transition:
    """Convert expert paths into state-action transitions for RL training."""
    input:
        script = ancient(os.path.join(CURRENT_PATH, "scripts", "generate_expert_path_transition.py")),
        path_file = ancient(os.path.join(CURRENT_PATH, "data", "expert_path_files", "train_expert_demonstration_relation_entity_max" + _MAX_PATH + "_filtered.pkl")),
        data_dir = ancient(os.path.join(CURRENT_PATH, "data"))
    output:
        os.path.join(CURRENT_PATH, "data", "expert_path_files", "train_expert_transitions_history" + _STATE_HISTORY + ".pkl")
    params:
        path_file_name = "train_expert_demonstration_relation_entity_max" + _MAX_PATH + "_filtered.pkl",
        max_path = config['MODELINFO']['PARAMS']['MAX_PATH'],
        state_history = config['MODELINFO']['PARAMS']['STATE_HISTORY'],
        expert_trains_file_name = "train_expert_transitions_history" + _STATE_HISTORY + ".pkl"
    shell:
        """
        python {input.script} --data_dir {input.data_dir} \
                              --path_file_name {params.path_file_name} \
                              --max_path {params.max_path} \
                              --state_history {params.state_history} \
                              --expert_trains_file_name {params.expert_trains_file_name}
        """

rule step15_pretrain_ac_model:
    """Pre-train Actor-Critic model using expert demonstrations and XGBoost ensemble
    for reward shaping. Node2Vec embeddings are used for entity initialization."""
    input:
        script = ancient(os.path.join(CURRENT_PATH, "scripts", "run_pretrain_ac_model.py")),
        data_dir = ancient(os.path.join(CURRENT_PATH, "data")),
        pretrained_model = ancient(os.path.join(CURRENT_PATH, "models", "xgboost_model_3class", "xgboost_model.pt")),
        path_file = ancient(os.path.join(CURRENT_PATH, "data", "expert_path_files", "train_expert_demonstration_relation_entity_max" + _MAX_PATH + "_filtered.pkl")),
        node2vec_emb = ancient(os.path.join(CURRENT_PATH, "data", "node2vec_output", "node2vec_entity_embeddings.pkl"))
    output:
        os.path.join(CURRENT_PATH, "models", "pretrain_AC_model", "pretrained_ac_model.pt")
    params:
        path_file_name = "train_expert_demonstration_relation_entity_max" + _MAX_PATH + "_filtered.pkl",
        text_emb_file_name = "node2vec_entity_embeddings.pkl",
        text_emb_dir = "node2vec_output",
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
                              --text_emb_dir {params.text_emb_dir} \
                              --output_folder {params.output_folder} \
                              --max_path {params.max_path} \
                              --max_pre_path {params.max_pre_path} \
                              --bandwidth {params.bandwidth} \
                              --bucket_interval {params.bucket_interval} \
                              --state_history {params.state_history} \
                              --pretrain_model_path {input.pretrained_model} \
                              --seed {params.seed} \
                              --use_gpu \
                              --gpu {params.gpu} \
                              --batch_size {params.batch_size} \
                              --epochs {params.epochs} \
                              --pre_actor_epoch {params.pre_actor_epoch} \
                              --lr {params.lr}
        """

rule step16_train_adac_model:
    """Train the Adversarial Actor-Critic (ADAC) model with warm-start from pretrained AC."""
    input:
        script = ancient(os.path.join(CURRENT_PATH, "scripts", "run_adac_model.py")),
        data_dir = ancient(os.path.join(CURRENT_PATH, "data")),
        pretrained_model = ancient(os.path.join(CURRENT_PATH, "models", "xgboost_model_3class", "xgboost_model.pt")),
        pre_ac_file = ancient(os.path.join(CURRENT_PATH, "models", "pretrain_AC_model", "pretrained_ac_model.pt")),
        path_file = ancient(os.path.join(CURRENT_PATH, "data", "expert_path_files", "train_expert_demonstration_relation_entity_max" + _MAX_PATH + "_filtered.pkl")),
        node2vec_emb = ancient(os.path.join(CURRENT_PATH, "data", "node2vec_output", "node2vec_entity_embeddings.pkl")),
        path_trans_file = ancient(os.path.join(CURRENT_PATH, "data", "expert_path_files", "train_expert_transitions_history" + _STATE_HISTORY + ".pkl"))
    output:
        os.path.join(CURRENT_PATH, "models", "ADAC_model", "policy_net", "step17_training_done.flag")
    params:
        path_file_name = "train_expert_demonstration_relation_entity_max" + _MAX_PATH + "_filtered.pkl",
        text_emb_file_name = "node2vec_entity_embeddings.pkl",
        text_emb_dir = "node2vec_output",
        path_trans_file_name = "train_expert_transitions_history" + _STATE_HISTORY + ".pkl",
        output_folder = os.path.join(CURRENT_PATH, "models"),
        max_path = config['MODELINFO']['PARAMS']['MAX_PATH'],
        bandwidth = config['MODELINFO']['PARAMS']['BANDWIDTH'],
        bucket_interval = config['MODELINFO']['PARAMS']['BUCKET_INTERVAL'],
        gpu = config['MODELINFO']['PARAMS']['GPU'],
        epochs = 100,
        train_batch_size = 1120,
        state_history = config['MODELINFO']['PARAMS']['STATE_HISTORY'],
        ac_update_delay = 50,
        entropy_weight = 0.005,
        disc_alpha = 0.006,
        metadisc_alpha = 0.012,
        num_rollouts = 35,
        act_dropout = 0.5,
        ac_lr = config["MODELINFO"]['PARAMS']['LEARNING_RATE'],
        disc_lr = config["MODELINFO"]['PARAMS']['LEARNING_RATE'],
        metadisc_lr = config["MODELINFO"]['PARAMS']['LEARNING_RATE']
    shell:
        """
        python {input.script} --data_dir {input.data_dir} \
                              --path_file_name {params.path_file_name} \
                              --text_emb_file_name {params.text_emb_file_name} \
                              --text_emb_dir {params.text_emb_dir} \
                              --path_trans_file_name {params.path_trans_file_name} \
                              --output_folder {params.output_folder} \
                              --max_path {params.max_path} \
                              --bandwidth {params.bandwidth} \
                              --bucket_interval {params.bucket_interval} \
                              --pretrain_model_path {input.pretrained_model} \
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
                                --metadisc_lr {params.metadisc_lr} && \
        touch {output}
        """

rule step17_select_best_model:
    """Evaluate ADAC model checkpoints and select the best MoA model."""
    input:
        script = ancient(os.path.join(CURRENT_PATH, "scripts", "select_best_moa_model.py")),
        data_dir = ancient(os.path.join(CURRENT_PATH, "data")),
        policy_net_folder_check = ancient(os.path.join(CURRENT_PATH, "models", "ADAC_model", "policy_net", "step17_training_done.flag")),
        pretrained_model = ancient(os.path.join(CURRENT_PATH, "models", "xgboost_model_3class", "xgboost_model.pt"))
    output:
        os.path.join(CURRENT_PATH, "models", "ADAC_model", "policy_net", "best_moa_model.pt")
    params:
        policy_net_folder = ancient(os.path.join(CURRENT_PATH, "models", "ADAC_model", "policy_net")),
        max_path = config['MODELINFO']['PARAMS']['MAX_PATH'],
        bandwidth = config['MODELINFO']['PARAMS']['BANDWIDTH'],
        bucket_interval = config['MODELINFO']['PARAMS']['BUCKET_INTERVAL'],
        state_history = config['MODELINFO']['PARAMS']['STATE_HISTORY'],
        act_dropout = 0.5,
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
                              --pretrain_model_path {input.pretrained_model} \
                              --use_gpu \
                              --gpu {params.gpu} \
                              --save_pred_paths
        """


## ═══════════════════════════════════════════════════════════════════════════════
## Steps 18-22: Precompute predictions & build SQL database
## ═══════════════════════════════════════════════════════════════════════════════

rule step18_split_disease_into_K_pieces:
    """Split disease list into K chunks for parallel precomputation."""
    input:
        script = ancient(os.path.join(CURRENT_PATH, "scripts", "split_disease_into_K_pieces.py")),
        data_dir = ancient(os.path.join(CURRENT_PATH, "data")),
        entity2freq = ancient(os.path.join(CURRENT_PATH, "data", 'entity2freq.txt')),
        relation2freq = ancient(os.path.join(CURRENT_PATH, "data", 'relation2freq.txt')),
        type2freq = ancient(os.path.join(CURRENT_PATH, "data", 'type2freq.txt')),
        entity2typeid = ancient(os.path.join(CURRENT_PATH, "data", 'entity2typeid.pkl')),
        all_node_info = ancient(os.path.join(CURRENT_PATH, "data", 'all_graph_nodes_info.txt'))
    output:
        os.path.join(CURRENT_PATH, "data", "disease_sets", "disease_set1.txt"),
        os.path.join(CURRENT_PATH, "data", "filtered_drug_nodes_for_precomputation.pkl"),
    params:
        K = config['PARALLEL_PRECOMPUTE']['K'],
        out_dir = os.path.join(CURRENT_PATH, "data", "disease_sets")
    shell:
        """
        python {input.script} --data_dir {input.data_dir} \
                              --K {params.K}
        """

rule step19_precompute_all_drug_disease_pairs_in_parallel:
    """Launch K parallel processes to precompute prediction scores and paths."""
    input:
        script = ancient(os.path.join(CURRENT_PATH, "scripts", "run_xDTD.py")),
        data_dir = ancient(os.path.join(CURRENT_PATH, "data")),
        ddp_model = ancient(os.path.join(CURRENT_PATH, "models", "xgboost_model_3class", "xgboost_model.pt")),
        moa_model = ancient(os.path.join(CURRENT_PATH, "models", "ADAC_model", "policy_net", "best_moa_model.pt")),
        disease_set1 = ancient(os.path.join(CURRENT_PATH, "data", "disease_sets", "disease_set1.txt")),
        disease_set2 = ancient(os.path.join(CURRENT_PATH, "data", "filtered_drug_nodes_for_precomputation.pkl")),
        model_dir = ancient(os.path.join(CURRENT_PATH, "models"))
    output:
        touch(os.path.join(CURRENT_PATH, "results", "step19_done.txt"))
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
        import subprocess
        script = str(input.script)
        data_dir = str(input.data_dir)
        model_dir = str(input.model_dir)
        out_dir = str(params.out_dir)
        log_dir = os.path.join(out_dir, "process_logs")
        os.makedirs(log_dir, exist_ok=True)
        for index in range(int(params.K)):
            idx = index + 1
            cmd = [
                "python", script,
                "--log_name", "run_xDTD_" + str(idx) + ".log",
                "--data_path", data_dir,
                "--model_path", model_dir,
                "--disease_set", os.path.join(data_dir, "disease_sets", "disease_set" + str(idx) + ".txt"),
                "--out_dir", out_dir,
                "--N_drugs", str(params.N_drugs),
                "--N_paths", str(params.N_paths),
                "--batch_size", str(params.batch_size),
                "--max_path", str(params.max_path),
                "--bandwidth", str(params.bandwidth),
                "--bucket_interval", str(params.bucket_interval),
                "--state_history", str(params.state_history),
                "--threshold", str(params.threshold),
            ]
            stdout_f = open(os.path.join(log_dir, "run_xDTD_" + str(idx) + ".stdout"), "w")
            stderr_f = open(os.path.join(log_dir, "run_xDTD_" + str(idx) + ".stderr"), "w")
            subprocess.Popen(cmd, start_new_session=True, stdout=stdout_f, stderr=stderr_f)

rule step20_build_sql_database:
    """Build the prediction score + path result SQLite database."""
    input:
        script = ancient(os.path.join(CURRENT_PATH, "scripts", "build_sql_database.py")),
        unused_file = ancient(os.path.join(CURRENT_PATH, "results", "step19_done.txt"))
    output:
        os.path.join(CURRENT_PATH, config['DATABASE']['DATABASE_NAME'])
    params:
        path_to_score_results = os.path.join(CURRENT_PATH, "results", "prediction_scores"),
        path_to_path_results = os.path.join(CURRENT_PATH, "results", "path_results"),
        database_name = config['DATABASE']['DATABASE_NAME'],
        outdir = CURRENT_PATH
    shell:
        """
        python {input.script} --build \
                              --path_to_score_results {params.path_to_score_results} \
                              --path_to_path_results {params.path_to_path_results} \
                              --database_name {params.database_name} \
                              --outdir {params.outdir}
        """

rule step21_build_mapping_database:
    """Add node/edge mapping tables to the SQLite database."""
    input:
        script = ancient(os.path.join(CURRENT_PATH, "scripts", "build_mapping_db.py")),
        nodes_jsonl = ancient(os.path.join(CURRENT_PATH, "data", config['TRANSLATOR_KG']['NODES_JSONL'])),
        edges_jsonl = ancient(os.path.join(CURRENT_PATH, "data", config['TRANSLATOR_KG']['EDGES_JSONL'])),
        unused_file = ancient(os.path.join(CURRENT_PATH, "results", "step19_done.txt")),
        database_name = ancient(os.path.join(CURRENT_PATH, config['DATABASE']['DATABASE_NAME']))
    output:
        touch(os.path.join(CURRENT_PATH, "results", "step22_done.txt"))
    params:
        outdir = CURRENT_PATH
    shell:
        """
        python {input.script} --build \
                              --nodes_jsonl {input.nodes_jsonl} \
                              --edges_jsonl {input.edges_jsonl} \
                              --database_name {input.database_name} \
                              --outdir {params.outdir}
        """
