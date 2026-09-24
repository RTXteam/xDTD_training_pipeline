# Automatic Pipeline for KGML-xDTD Model Training

This repository provides an automated [Snakemake](https://snakemake.readthedocs.io/en/stable/)-based pipeline for training the [KGML-xDTD](https://github.com/chunyuma/KGML-xDTD) (Knowledge Graph-based Machine Learning for Explainable Drug Treatment Discovery) model for the [Translator](https://ncats.nih.gov/research/research-activities/biomedical-data-translator) knowledge graph.

The pipeline automates the full workflow from data acquisition to deployment-ready database generation. It downloads and processes knowledge graph data, curated ground-truth drug-disease pairs, generates demonstration/expert paths, trains a suite of models (XGBoost ensemble for prediction, [Node2Vec](https://arxiv.org/abs/1607.00653) for node embeddings, and an Adversarial Actor-Critic (ADAC) model for explainable path reasoning), pre-computes predictions for all drug-disease combinations, and builds a final SQLite database containing prediction scores, explanation paths, and KG metadata mapping tables.

The associated publication can be found here: https://academic.oup.com/gigascience/article/doi/10.1093/gigascience/giad057/7246583

Please cite via:

> Ma, C., Zhou, C., Wang, H., & Koslicki, D. (2023). KGML-xDTD: a knowledge graph-based machine learning framework for drug treatment prediction and mechanism description. *GigaScience*, 12, giad057. [https://doi.org/10.1093/gigascience/giad057](https://academic.oup.com/gigascience/article/doi/10.1093/gigascience/giad057/7246583)

&nbsp;

## Table of Contents

- [Installation](#installation)
- [Configuration](#configuration)
- [Prerequisites](#prerequisites)
- [Running the Pipeline](#running-the-pipeline)
- [Pipeline Steps](#pipeline-steps)
  - [Step 1 — Download Data](#step-1--download-data)
  - [Step 2 — Process Translator KG](#step-2--process-translator-kg)
  - [Step 3 — Filter Graph Nodes and Edges](#step-3--filter-graph-nodes-and-edges)
  - [Step 4 — Process Drug-Disease Lists](#step-4--process-drug-disease-lists)
  - [Step 5 — Convert Ground Truth Pairs](#step-5--convert-ground-truth-pairs)
  - [Step 6 — Preprocess Data](#step-6--preprocess-data)
  - [Step 7 — Process DrugBank Action Descriptions](#step-7--process-drugbank-action-descriptions)
  - [Step 8 — Integrate DrugBank Data](#step-8--integrate-drugbank-data)
  - [Step 9 — Check Reachable Paths](#step-9--check-reachable-paths)
  - [Step 10 — Generate Expert Paths](#step-10--generate-expert-paths)
  - [Step 11 — Split Train / Test](#step-11--split-train--test)
  - [Step 12 — Generate Node2Vec Embeddings](#step-12--generate-node2vec-embeddings)
  - [Step 13 — Train XGBoost Ensemble](#step-13--train-xgboost-ensemble)
  - [Step 14 — Generate Expert Path Transitions](#step-14--generate-expert-path-transitions)
  - [Step 15 — Pre-train Actor-Critic Model](#step-15--pre-train-actor-critic-model)
  - [Step 16 — Train ADAC Model](#step-16--train-adac-model)
  - [Step 17 — Select Best Model](#step-17--select-best-model)
  - [Step 18 — Split Diseases into K Pieces](#step-18--split-diseases-into-k-pieces)
  - [Step 19 — Pre-compute All Drug-Disease Pairs](#step-19--pre-compute-all-drug-disease-pairs)
  - [Step 20 — Build SQL Database](#step-20--build-sql-database)
  - [Step 21 — Build Mapping Database](#step-21--build-mapping-database)
- [Output Database](#output-database)
- [Contact](#contact)

---

## Installation

1. Install [conda](https://conda.io/projects/conda/en/latest/user-guide/install/index.html), then create the required environment:

```bash
conda env create -f envs/xDTD_training_pipeline_env.yml
```

2. Activate the main environment:

```bash
conda activate xDTD_training_pipeline
```

---

## Configuration

Edit `config.yaml` before running the pipeline. Key parameters you may need to adjust:

| Section | Parameter | Description |
|---------|-----------|-------------|
| `TRANSLATOR_KG` | `DOWNLOAD_URL` | URL to download the translator KG archive (`.tar.zst`) |
| `KGINFO` | `REMOVE_KNOWLEDGE_SOURCES` | List of knowledge sources to remove (e.g. `["infores:semmeddb"]`); empty = keep all |
| `MODELINFO.PARAMS` | `GPU` | GPU device index (set to `0` if you have a single GPU) |
| `NODE2VEC` | `EMBEDDING_DIM` | Node2Vec embedding dimension (default: 512) |
| `NODE2VEC` | `WORKERS` | Parallel workers for Node2Vec training (default: 64) |
| `ENSEMBLE` | `N_SHARDS` | Number of XGBoost ensemble shards (default: 3) |
| `ENSEMBLE` | `DEVICE` | XGBoost device: `cuda` for GPU, `cpu` for CPU-only |
| `ENSEMBLE` | `GPU_IDS` | GPU IDs for ensemble training |
| `SPLIT` | `TEST_SIZE` | Fraction of data reserved for test set (default: 0.1) |
| `PARALLEL_PRECOMPUTE` | `K` | Number of disease-set chunks for parallel pre-computation |
| `DATABASE` | `DATABASE_NAME` | Output SQLite database filename |

> **Note:** The Biolink model version is **auto-detected** from the KGX archive's `content_metadata.json` during Step 2 and propagated to all downstream steps. No manual configuration is needed.

---

## Prerequisites

### Ground Truth Data

For Koslicki Lab internal use, you can find the ground truth file `gt_pairs_raw.tsv` under `/scratch/backup/xDTD_training_pipeline_files/`. Place the file in the `data/` folder before running the pipeline.
This file contains drug-disease pairs with indication/contraindication labels.

### DrugBank XML

You need a DrugBank account to download `drugbank.xml` from [DrugBank Releases](https://go.drugbank.com/releases/latest). For Koslicki Lab internal use, you can find it under `/scratch/backup/xDTD_training_pipeline_files`. Place the file in the `data/` folder before running the pipeline.

---

## Running the Pipeline

Run all steps up to pre-computation:

```bash
nohup snakemake --cores 16 -s Run_Pipeline.smk targets &
```

> **Note:** Step 19 (pre-computation) runs in the background. Once it finishes, run the final two database-building steps separately:

```bash
nohup snakemake --cores 16 -s Run_Pipeline.smk step20_build_sql_database &
nohup snakemake --cores 16 -s Run_Pipeline.smk step21_build_mapping_database &
```

---

## Pipeline Steps

### Step 1 &mdash; Download Data

Downloads all required external datasets:
- **DrugMechDB**: `indication_paths.yaml` from [DrugMechDB](https://github.com/SuLab/DrugMechDB) (curated drug mechanism paths)
- **Translator KG**: `nodes.jsonl` and `edges.jsonl` from the Translator knowledge graph archive
- **Drug/Disease Lists**: drug and disease entity lists from [EveryCure datasets](https://huggingface.co/everycure) (`everycure/drug-list`, `everycure/disease-list`)

> **Note:** `gt_pairs_raw.tsv` and `drugbank.xml` must be manually placed in `data/` — see [Prerequisites](#prerequisites).

### Step 2 &mdash; Process Translator KG

Parses the raw translator KG JSONL files (`nodes.jsonl`, `edges.jsonl`) and converts them into tab-separated graph files:
- `graph_edges.txt` &mdash; all edges with subject, object, predicate
- `all_graph_nodes_info.txt` &mdash; node metadata (id, name, category)
- `biolink_version.txt` &mdash; auto-detected Biolink model version from KGX `content_metadata.json` (used by downstream steps)

### Step 3 &mdash; Filter Graph Nodes and Edges

Filters the full graph to remove:
- Nodes with categories not relevant to drug treatment prediction (e.g. `biolink:Cell`, `biolink:AnatomicalEntity`, etc.)
- Edges whose knowledge sources are entirely within the `REMOVE_KNOWLEDGE_SOURCES` list (default: keep all)
- Redundant edges based on Biolink predicate hierarchy (more specific predicates subsume ancestors)

Produces `filtered_graph_edges.txt` and `filtered_graph_nodes_info.txt`.

### Step 4 &mdash; Process Drug-Disease Lists

Processes the raw drug and disease entity lists, filtering to only include entities present in the filtered graph. Outputs `drug_list.txt` and `disease_list.txt`.

### Step 5 &mdash; Convert Ground Truth Pairs

Converts the raw ground truth file (`gt_pairs_raw.tsv`) into the pipeline's standard format by:
- Normalizing CURIEs via the [Node Normalization API](https://nodenormalization-sri.renci.org/)
- Separating indications (y=1) as TP pairs and contraindications (y=0) as TN pairs
- Filtering to entities present in the KG
- Augmenting drug/disease lists with new entries from ground truth

Outputs: `tp_pairs.txt` (true positives) and `tn_pairs.txt` (true negatives).

### Step 6 &mdash; Preprocess Data

Generates core data structures for model training:
- `entity2freq.txt`, `relation2freq.txt`, `type2freq.txt` &mdash; frequency mappings
- `adj_list.pkl` &mdash; adjacency list representation of the graph
- `entity2typeid.pkl` &mdash; entity-to-type mapping
- `kg.pgrk` &mdash; PageRank scores for all nodes

### Step 7 &mdash; Process DrugBank Action Descriptions

Parses `drugbank.xml` and the translator KG `nodes.jsonl` to extract drug-gene-action relationships. Outputs:
- `drugbank_dict.pkl` &mdash; DrugBank drug-target dictionary
- `drugbank_mapping.txt` &mdash; identifier mapping between DrugBank and the KG
- `p_expert_paths.txt` &mdash; expert paths derived from DrugBank

**Requires** `drugbank.xml` to be in the `data/` folder (see [Prerequisites](#prerequisites)).

### Step 8 &mdash; Integrate DrugBank Data with MolePro Data

Combines the DrugBank-derived expert paths with additional drug-gene data from MolePro to produce:
- `all_drugs.txt` &mdash; consolidated drug list
- `p_expert_paths_combined.txt` &mdash; merged expert paths from all sources

### Step 9 &mdash; Check Reachable Paths

Checks whether 3-hop reachable paths exist between each true positive drug-disease pair through intermediate genes. Produces:
- `reachable_expert_paths_max3.txt` &mdash; expert paths that are reachable in the graph
- `reachable_tp_pairs_max3.txt` / `unreachable_tp_pairs_max3.txt` &mdash; reachable/unreachable pair splits

### Step 10 &mdash; Generate Expert Paths

Generates expert demonstration paths for reinforcement learning training:
- Raw, filtered, translated, and relation-entity formats of expert paths
- Used as demonstrations for the Actor-Critic pre-training

### Step 11 &mdash; Split Train / Test

Splits the drug-disease pairs into training (90%) and test (10%) sets using drug-stratified splitting:
- TP pairs separated into "in expert" and "not in expert" groups, each split independently to preserve expert-path proportionality
- Sources (drugs) that appear only once are forced into the training set
- Stratified splitting ensures **every drug in the test set also appears in the training set**
- Expert demonstration paths are also split by train/test for RL compatibility
- Logs verification that all test drugs have been seen in training

### Step 12 &mdash; Generate Node2Vec Embeddings

Generates 512-dimensional Node2Vec entity embeddings using [GRAPE/ensmallen](https://github.com/AnacletoLAB/ensmallen) (Rust) for fast random walk generation and [gensim](https://radimrehurek.com/gensim/) Word2Vec for training:
- Drug-disease edges are removed to prevent label leakage
- Default: walk_length=30, walks_per_node=10, p=1.0, q=1.0

### Step 13 &mdash; Train XGBoost Ensemble

Trains a 3-shard XGBoost ensemble for 3-class drug-disease prediction (true positive, true negative, unknown):
- Each shard uses different replacement-based synthetic negatives (2 drug + 2 disease replacements per positive pair)
- Hyperparameter optimization via [scikit-optimize](https://scikit-optimize.github.io/) Gaussian Process (20 calls per shard)
- Inner StratifiedShuffleSplit (10%) for HPO evaluation
- Best hyperparameters retrained on full training data per shard
- Final prediction = mean of predict_proba across all shards
- Also saves `entity_embeddings.npy` for downstream RL pipeline compatibility

After training, generates an evaluation report (`reports/evaluation_report.md`) with:
- Classification metrics (3-class and renormalized 2-class)
- Ranking metrics (MRR, Hit@K, Recall@N)
- Visualization plots (treat score distribution, precision-recall curve, ranking performance)

### Step 14 &mdash; Generate Expert Path Transitions

Converts expert demonstration paths into state-action transition sequences with configurable history length, used as training signal for the Actor-Critic model.

### Step 15 &mdash; Pre-train Actor-Critic Model

Pre-trains the Actor-Critic (AC) model on expert demonstration paths. The actor learns to follow expert trajectories while the critic evaluates state values using the pre-trained XGBoost ensemble for reward shaping. Node2Vec embeddings are used for entity initialization.

### Step 16 &mdash; Train ADAC Model

Trains the Adversarial Actor-Critic (ADAC) model with:
- Warm-start from pre-trained AC weights
- Discriminator and meta-discriminator for adversarial imitation learning
- Configurable entropy weight, learning rates, and rollout count

### Step 17 &mdash; Select Best Model

Evaluates each saved policy model checkpoint, scoring them on mechanism-of-action (MOA) path quality. Selects and saves the best model as `best_moa_model.pt`.

### Step 18 &mdash; Split Diseases into K Pieces

Splits the disease list into K chunks for parallel pre-computation, and identifies the set of drug nodes to evaluate.

### Step 19 &mdash; Pre-compute All Drug-Disease Pairs

Launches K parallel processes to pre-compute prediction scores and explanation paths for all drug-disease pair combinations. Each process handles one disease chunk. **This step runs in the background.**

### Step 20 &mdash; Build SQL Database

Reads the pre-computed results and builds the SQLite database with two tables:

| Table | Key Columns | Description |
|-------|-------------|-------------|
| `PREDICTION_SCORE_TABLE` | `drug_id`, `disease_id` | Drug-disease prediction scores (`tn_score`, `tp_score`, `unknown_score`) |
| `PATH_RESULT_TABLE` | `drug_id`, `disease_id` | Predicted explanation paths with path scores |

### Step 21 &mdash; Build Mapping Database

Reads the translator KG JSONL files and adds two mapping tables to the existing SQLite database:

| Table | Key | Columns |
|-------|-----|---------|
| `NODE_MAPPING_TABLE` | `id` | `name`, `category`, `extra_attributes` (JSON) |
| `EDGE_MAPPING_TABLE` | (`subject`, `predicate`, `object`) | `id`, `category`, `extra_attributes` (JSON) |

These tables enable looking up KG node/edge metadata when interpreting predicted paths.

---

## Output Database

The final database (e.g. `ExplainableDTD_v1.0-tier0-20260621-all_with_paths.db`) contains four tables:

| Table | Records | Purpose |
|-------|---------|---------|
| `PREDICTION_SCORE_TABLE` | ~millions | Drug-disease prediction scores |
| `PATH_RESULT_TABLE` | ~millions | Explanation paths for predictions |
| `NODE_MAPPING_TABLE` | ~1.7M | Translator KG node metadata |
| `EDGE_MAPPING_TABLE` | ~29.4M | Translator KG edge metadata |

---

## Contact

If you have any questions or need help, please contact @chunyuma.
