# Automatic Pipeline for KGML-xDTD Model Training

This repository provides an automated [Snakemake](https://snakemake.readthedocs.io/en/stable/)-based pipeline for training the [KGML-xDTD](https://github.com/chunyuma/KGML-xDTD) (Knowledge Graph-based Machine Learning for Explainable Drug Treatment Discovery) model for the [Translator](https://ncats.nih.gov/research/research-activities/biomedical-data-translator) knowledge graph.

The pipeline automates the full workflow from data acquisition to deployment-ready database generation. It downloads and processes knowledge graph, curated ground-truth drug-disease pairs, generates demonstration/expert paths, trains a suite of models (XGBoost for prediction, [GraphSAGE](https://snap.stanford.edu/graphsage/) for node embeddings, and an Adversarial Actor-Critic (ADAC) model for explainable path reasoning), pre-computes predictions for all drug-disease combinations, and builds a final SQLite database containing prediction scores, explanation paths, and KG metadata mapping tables.

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
  - [Step 5 — Process Ground Truth Pairs](#step-5--process-ground-truth-pairs)
  - [Step 6 — Preprocess Data](#step-6--preprocess-data)
  - [Step 7 — Process DrugBank Action Descriptions](#step-7--process-drugbank-action-descriptions)
  - [Step 8 — Integrate DrugBank Data](#step-8--integrate-drugbank-data)
  - [Step 9 — Check Reachable Paths](#step-9--check-reachable-paths)
  - [Step 10 — Generate Expert Paths](#step-10--generate-expert-paths)
  - [Step 11 — Split Train / Val / Test](#step-11--split-train--val--test)
  - [Step 12 — Calculate Attribute Embeddings](#step-12--calculate-attribute-embeddings)
  - [Step 13 — GraphSAGE Data Generation](#step-13--graphsage-data-generation)
  - [Step 14 — Generate Random Walk](#step-14--generate-random-walk)
  - [Step 15 — Generate GraphSAGE Embeddings](#step-15--generate-graphsage-embeddings)
  - [Step 16 — Transform Format](#step-16--transform-format)
  - [Step 17 — Pre-train XGBoost Model](#step-17--pre-train-xgboost-model)
  - [Step 18 — Generate Expert Path Transitions](#step-18--generate-expert-path-transitions)
  - [Step 19 — Pre-train Actor-Critic Model](#step-19--pre-train-actor-critic-model)
  - [Step 20 — Train ADAC Model](#step-20--train-adac-model)
  - [Step 21 — Select Best Model](#step-21--select-best-model)
  - [Step 22 — Split Diseases into K Pieces](#step-22--split-diseases-into-k-pieces)
  - [Step 23 — Pre-compute All Drug-Disease Pairs](#step-23--pre-compute-all-drug-disease-pairs)
  - [Step 23 — Build SQL Database](#step-23--build-sql-database)
  - [Step 24 — Build Mapping Database](#step-24--build-mapping-database)
- [Output Database](#output-database)
- [Contact](#contact)

---

## Installation

1. Install [conda](https://conda.io/projects/conda/en/latest/user-guide/install/index.html), then create the required environments:

```bash
conda env create -f envs/graphsage_p2.7env.yml
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
| `KGINFO` | `BIOLINK_VERSION` | Biolink model version used by the translator KG (e.g. `4.3.6`) |
| `KGINFO` | `PUBLICATION_CUTOFF` | Minimum publication count threshold for edge filtering |
| `MODELINFO.PARAMS` | `GPU` | GPU device index (set to `0` if you have a single GPU) |
| `PARALLEL_PRECOMPUTE` | `K` | Number of disease-set chunks for parallel pre-computation (depends on available RAM) |
| `DATABASE` | `DATABASE_NAME` | Output SQLite database filename |

---

## Prerequisites

### DrugBank XML

You need a DrugBank account to download `drugbank.xml` from [DrugBank Releases](https://go.drugbank.com/releases/latest). Place the file in the `data/` folder before running the pipeline.

---

## Running the Pipeline

Run all steps up to pre-computation:

```bash
nohup snakemake --cores 16 -s Run_Pipeline.smk targets &
```

> **Note:** Step 23 (pre-computation) runs in the background. Once it finishes, run the final two database-building steps separately:

```bash
nohup snakemake --cores 16 -s Run_Pipeline.smk step23_build_sql_database &
nohup snakemake --cores 16 -s Run_Pipeline.smk step24_build_mapping_database &
```

---

## Pipeline Steps

### Step 1 &mdash; Download Data

Downloads all required external datasets:
- **DrugMechDB**: `indication_paths.yaml` from [DrugMechDB](https://github.com/SuLab/DrugMechDB) (curated drug mechanism paths)
- **Translator KG**: `nodes.jsonl` and `edges.jsonl` from the Translator knowledge graph archive
- **Ground Truth Pairs**: indication and contraindication lists from [EveryCure](https://github.com/everycure-org/matrix-indication-list)
- **Drug/Disease Lists**: drug and disease entity lists from [EveryCure datasets](https://huggingface.co/everycure) (`everycure/drug-list`, `everycure/disease-list`)

### Step 2 &mdash; Process Translator KG

Parses the raw translator KG JSONL files (`nodes.jsonl`, `edges.jsonl`) and converts them into tab-separated graph files:
- `graph_edges.txt` &mdash; all edges with subject, object, predicate
- `all_graph_nodes_info.txt` &mdash; node metadata (id, name, category)

Uses the Biolink model version specified in `config.yaml` to standardize node types and predicates.

### Step 3 &mdash; Filter Graph Nodes and Edges

Filters the full graph to remove:
- Nodes with categories not relevant to drug treatment prediction
- SemMedDB Edges that do not meet the publication count threshold (`PUBLICATION_CUTOFF`)

Produces `filtered_graph_edges.txt` and `filtered_graph_nodes_info.txt`.

### Step 4 &mdash; Process Drug-Disease Lists

Processes the raw drug and disease entity lists, filtering to only include entities present in the filtered graph. Outputs `drug_list.txt` and `disease_list.txt`.

### Step 5 &mdash; Process Ground Truth Pairs

Generates high-quality training pairs by cross-referencing indication/contraindication data with the filtered graph:
- `tp_pairs.txt` &mdash; true positive drug-disease pairs (indication pairs)
- `tn_pairs.txt` &mdash; true negative drug-disease pairs (contraindications pairs)

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

### Step 11 &mdash; Split Train / Val / Test

Splits the drug-disease pairs and corresponding expert paths into training (80%), validation (10%), and test (10%) sets.

### Step 12 &mdash; Calculate Attribute Embeddings

Computes text-based attribute embeddings for all graph nodes using [PubMedBERT](https://huggingface.co/microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract), followed by PCA dimensionality reduction.

### Step 13 &mdash; GraphSAGE Data Generation

Prepares input data for [GraphSAGE](https://snap.stanford.edu/graphsage/) unsupervised node embedding:
- Graph JSON structure, class/ID maps, feature matrix (`data-feats.npy`)
- Combines text embeddings with graph topology features

### Step 14 &mdash; Generate Random Walk

Produces random walk sequences on the graph for GraphSAGE training (walk length: 30, 10 walks per node).

### Step 15 &mdash; Generate GraphSAGE Embeddings

Trains an unsupervised GraphSAGE model to generate structural node embeddings. Requires the Python 2.7 GraphSAGE environment.

### Step 16 &mdash; Transform Format

Converts GraphSAGE output embeddings into a pickle format (`unsuprvised_graphsage_entity_embeddings.pkl`) for downstream model consumption.

### Step 17 &mdash; Pre-train XGBoost Model

Trains an XGBoost classifier for 3-class drug-disease prediction (true positive, true negative, unknown). Uses [Optuna](https://optuna.org/) for hyperparameter optimization with configurable trial counts and early stopping.

### Step 18 &mdash; Generate Expert Path Transitions

Converts expert demonstration paths into state-action transition sequences with configurable history length, used as training signal for the Actor-Critic model.

### Step 19 &mdash; Pre-train Actor-Critic Model

Pre-trains the Actor-Critic (AC) model on expert demonstration paths. The actor learns to follow expert trajectories while the critic evaluates state values using the pre-trained XGBoost model for reward shaping.

### Step 20 &mdash; Train ADAC Model

Formally trains the Adversarial Actor-Critic (ADAC) model with:
- Warm-start from pre-trained AC weights
- Discriminator and meta-discriminator for adversarial imitation learning
- Configurable entropy weight, learning rates, and rollout count

### Step 21 &mdash; Select Best Model

Evaluates each saved policy model checkpoint, scoring them on mechanism-of-action (MOA) path quality. Selects and saves the best model as `best_moa_model.pt`.

### Step 22 &mdash; Split Diseases into K Pieces

Splits the disease list into K chunks for parallel pre-computation, and identifies the set of drug nodes to evaluate.

### Step 23 &mdash; Pre-compute All Drug-Disease Pairs

Launches K parallel processes to pre-compute prediction scores and explanation paths for all drug-disease pair combinations. Each process handles one disease chunk. **This step runs in the background.**

### Step 23 &mdash; Build SQL Database

Reads the pre-computed results and builds the SQLite database with two tables:

| Table | Key Columns | Description |
|-------|-------------|-------------|
| `PREDICTION_SCORE_TABLE` | `drug_id`, `disease_id` | Drug-disease prediction scores (`tn_score`, `tp_score`, `unknown_score`) |
| `PATH_RESULT_TABLE` | `drug_id`, `disease_id` | Predicted explanation paths with path scores |

### Step 24 &mdash; Build Mapping Database

Reads the translator KG JSONL files and adds two mapping tables to the existing SQLite database:

| Table | Key | Columns |
|-------|-----|---------|
| `NODE_MAPPING_TABLE` | `id` | `name`, `category`, `equivalent_identifiers`, `description`, `synonym`, `xref`, `chembl_natural_product`, `chembl_availability_type`, `chembl_black_box_warning` |
| `EDGE_MAPPING_TABLE` | (`subject`, `predicate`, `object`) | `id`, `category`, `qualifier`, `publications`, `sources`, `resource_id`, `resource_role`, `knowledge_level`, `agent_type`, `stage_qualifier`, `original_subject`, `original_object`, `extra_attributes` |

These tables enable looking up KG node/edge metadata when interpreting predicted paths.

---

## Output Database

The final database (e.g. `ExplainableDTD_v1.0-tier0-20260408-all_with_paths.db`) contains four tables:

| Table | Records | Purpose |
|-------|---------|---------|
| `PREDICTION_SCORE_TABLE` | ~millions | Drug-disease prediction scores |
| `PATH_RESULT_TABLE` | ~millions | Explanation paths for predictions |
| `NODE_MAPPING_TABLE` | ~1.7M | Translator KG node metadata |
| `EDGE_MAPPING_TABLE` | ~29.4M | Translator KG edge metadata |

---

## Contact

If you have any questions or need help, please contact @chunyuma.
