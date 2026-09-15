"""Generate Node2Vec entity embeddings from a filtered knowledge graph.

Node2Vec settings:
  - embedding_dim=512, walk_length=30, walks_per_node=10
  - iterations=10, window_size=10, random_seed=42
  - Drug-disease edges removed to prevent label leakage

Uses GRAPE/ensmallen (Rust) for fast random walk generation and
gensim Word2Vec for embedding training.

"""

import argparse
import json
import os
import pickle
import sys

import numpy as np
import polars as pl

pathlist = os.getcwd().split(os.path.sep)
ROOTindex = pathlist.index("xDTD_training_pipeline")
ROOTPath = os.path.sep.join([*pathlist[:(ROOTindex + 1)]])
sys.path.append(os.path.join(ROOTPath, 'scripts'))
import utils

DRUG_CATEGORIES = {'biolink:Drug', 'biolink:SmallMolecule', 'biolink:ChemicalEntity'}
DISEASE_CATEGORIES = {'biolink:Disease', 'biolink:PhenotypicFeature'}


def _parse_categories(cat_str):
    """Parse a JSON-encoded list of categories from a string."""
    if cat_str is None or cat_str == '':
        return set()
    try:
        cats = json.loads(cat_str)
        if isinstance(cats, list):
            return set(cats)
    except (json.JSONDecodeError, TypeError):
        pass
    return {cat_str}


def load_node_categories(graph_nodes_path):
    """Load node info and return dict: node_id -> set of categories."""
    df = pl.read_csv(graph_nodes_path, separator='\t')
    node_cats = {}
    for row in df.iter_rows(named=True):
        node_id = row['id']
        raw_cats = row.get('all_categories', row.get('primary_category', ''))
        node_cats[node_id] = _parse_categories(raw_cats)
    return node_cats


def identify_drug_disease_nodes(node_cats):
    """Identify drug and disease node sets based on their categories."""
    drug_nodes = set()
    disease_nodes = set()
    for node_id, cats in node_cats.items():
        if cats & DRUG_CATEGORIES:
            drug_nodes.add(node_id)
        if cats & DISEASE_CATEGORIES:
            disease_nodes.add(node_id)
    return drug_nodes, disease_nodes


def build_edgelist_without_drug_disease(graph_edges_path, drug_nodes, disease_nodes,
                                        logger=None):
    """Load edges and remove all edges between drug-type and disease-type nodes.

    An edge (u, v) is removed if:
      - u is a drug AND v is a disease, OR
      - u is a disease AND v is a drug
    """
    df = pl.read_csv(graph_edges_path, separator='\t')
    n_total = df.height

    sources = df['source'].to_list()
    targets = df['target'].to_list()

    keep_mask = []
    n_removed = 0
    for s, t in zip(sources, targets):
        is_drug_disease = (s in drug_nodes and t in disease_nodes)
        is_disease_drug = (s in disease_nodes and t in drug_nodes)
        if is_drug_disease or is_disease_drug:
            keep_mask.append(False)
            n_removed += 1
        else:
            keep_mask.append(True)

    filtered_df = df.filter(pl.Series(keep_mask))

    if logger:
        logger.info(f"Edges: {n_total} total, {n_removed} drug-disease edges removed, "
                     f"{filtered_df.height} edges retained")

    return filtered_df


def write_edgelist_file(edges_df, output_path):
    """Write edge list to a TSV file for ensmallen Graph.from_csv()."""
    with open(output_path, 'w') as f:
        for row in edges_df.iter_rows(named=True):
            f.write(f"{row['source']}\t{row['target']}\n")


def run_node2vec(edgelist_path, all_node_ids, embedding_dim=512, walk_length=30,
                 walks_per_node=10, p=1.0, q=1.0, iterations=10,
                 window_size=10, workers=8, seed=42, logger=None):
    """Run Node2Vec using GRAPE/ensmallen (Rust) for random walks + gensim Word2Vec.

    """
    import time
    from ensmallen import Graph
    from gensim.models import Word2Vec

    if logger:
        logger.info(f"Running Node2Vec: dim={embedding_dim}, walk_length={walk_length}, "
                     f"walks_per_node={walks_per_node}, p={p}, q={q}, "
                     f"iterations={iterations}, window={window_size}")

    # Load graph with ensmallen (Rust-based, very fast)
    t0 = time.time()
    g = Graph.from_csv(
        edge_path=edgelist_path,
        edge_list_separator='\t',
        sources_column_number=0,
        destinations_column_number=1,
        edge_list_header=False,
        directed=False,
        name='kg',
    )
    t1 = time.time()
    if logger:
        logger.info(f"Graph loaded in {t1-t0:.1f}s: "
                     f"{g.get_number_of_nodes()} nodes, {g.get_number_of_edges()} edges")

    # Generate random walks with ensmallen
    t0 = time.time()
    walks = g.complete_walks(
        walk_length=walk_length,
        iterations=walks_per_node,
        return_weight=1.0 / p,
        explore_weight=1.0 / q,
        random_state=seed,
    )
    t1 = time.time()
    n_walks = g.get_number_of_nodes() * walks_per_node
    if logger:
        logger.info(f"Random walks generated in {t1-t0:.1f}s "
                     f"({n_walks:,} walks, {n_walks/(t1-t0):,.0f} walks/sec)")

    # Convert walk node IDs to string labels for Word2Vec
    t0 = time.time()
    node_names_arr = np.array(g.get_node_names())
    walks_np = np.array(walks, dtype=np.int64)
    str_walks = node_names_arr[walks_np].tolist()
    t1 = time.time()
    if logger:
        logger.info(f"Walk string conversion completed in {t1-t0:.1f}s")

    # Train Word2Vec on the walks
    t0 = time.time()
    model = Word2Vec(
        str_walks,
        vector_size=embedding_dim,
        window=window_size,
        min_count=0,
        sg=1,
        workers=workers,
        epochs=iterations,
        seed=seed,
    )
    t1 = time.time()
    if logger:
        logger.info(f"Word2Vec training completed in {t1-t0:.1f}s")

    embeddings = {}
    nodes_with_emb = 0
    nodes_without_emb = 0
    for node_id in all_node_ids:
        if node_id in model.wv:
            embeddings[node_id] = model.wv[node_id].astype(np.float32)
            nodes_with_emb += 1
        else:
            embeddings[node_id] = np.zeros(embedding_dim, dtype=np.float32)
            nodes_without_emb += 1

    if logger:
        logger.info(f"Embeddings: {nodes_with_emb} nodes with embeddings, "
                     f"{nodes_without_emb} nodes with zero vectors (isolated)")

    return embeddings


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate Node2Vec embeddings from KG")
    parser.add_argument("--graph_nodes", type=str, required=True,
                        help="Path to filtered_graph_nodes_info.txt")
    parser.add_argument("--graph_edges", type=str, required=True,
                        help="Path to filtered_graph_edges.txt")
    parser.add_argument("--tp_pairs", type=str, default=None,
                        help="Path to tp_pairs.txt (for logging stats)")
    parser.add_argument("--tn_pairs", type=str, default=None,
                        help="Path to tn_pairs.txt (for logging stats)")
    parser.add_argument("--embedding_dim", type=int, default=512)
    parser.add_argument("--walk_length", type=int, default=30)
    parser.add_argument("--walks_per_node", type=int, default=10)
    parser.add_argument("--p", type=float, default=1.0,
                        help="Node2Vec return parameter (default 1.0 = DeepWalk)")
    parser.add_argument("--q", type=float, default=1.0,
                        help="Node2Vec in-out parameter (default 1.0 = DeepWalk)")
    parser.add_argument("--iterations", type=int, default=10,
                        help="Word2Vec training epochs")
    parser.add_argument("--window_size", type=int, default=10)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_folder", type=str, required=True)
    parser.add_argument("--log_dir", type=str, default=os.path.join(ROOTPath, "log_folder"))
    parser.add_argument("--log_name", type=str, default="generate_node2vec_embedding.log")
    args = parser.parse_args()

    logger = utils.get_logger(os.path.join(args.log_dir, args.log_name))
    logger.info(args)

    os.makedirs(args.output_folder, exist_ok=True)

    # 1. Load node categories
    logger.info("Loading node categories...")
    node_cats = load_node_categories(args.graph_nodes)
    all_node_ids = list(node_cats.keys())
    logger.info(f"Loaded {len(all_node_ids)} nodes")

    # 2. Identify drug and disease nodes
    drug_nodes, disease_nodes = identify_drug_disease_nodes(node_cats)
    logger.info(f"Drug nodes: {len(drug_nodes)}, Disease nodes: {len(disease_nodes)}")

    # 3. Build edge list without drug-disease edges
    logger.info("Building edge list without drug-disease edges...")
    filtered_edges = build_edgelist_without_drug_disease(
        args.graph_edges, drug_nodes, disease_nodes, logger=logger
    )

    # 4. Write edge list file for ensmallen Graph.from_csv()
    edgelist_path = os.path.join(args.output_folder, 'edgelist_no_drug_disease.tsv')
    write_edgelist_file(filtered_edges, edgelist_path)
    logger.info(f"Edge list written to {edgelist_path}")

    # 5. Run Node2Vec
    logger.info("Starting Node2Vec training...")
    embeddings = run_node2vec(
        edgelist_path=edgelist_path,
        all_node_ids=all_node_ids,
        embedding_dim=args.embedding_dim,
        walk_length=args.walk_length,
        walks_per_node=args.walks_per_node,
        p=args.p,
        q=args.q,
        iterations=args.iterations,
        window_size=args.window_size,
        workers=args.workers,
        seed=args.seed,
        logger=logger,
    )

    # 6. Save embeddings
    emb_path = os.path.join(args.output_folder, 'node2vec_entity_embeddings.pkl')
    with open(emb_path, 'wb') as f:
        pickle.dump(embeddings, f, protocol=pickle.HIGHEST_PROTOCOL)
    logger.info(f"Embeddings saved to {emb_path} ({len(embeddings)} entities, "
                f"dim={args.embedding_dim})")

    # 7. Log stats about ground truth coverage
    if args.tp_pairs:
        tp = pl.read_csv(args.tp_pairs, separator='\t')
        tp_drugs = set(tp['drug_id'].unique().to_list())
        tp_diseases = set(tp['disease_id'].unique().to_list())
        tp_drugs_covered = len(tp_drugs & set(embeddings.keys()))
        tp_diseases_covered = len(tp_diseases & set(embeddings.keys()))
        logger.info(f"TP coverage: {tp_drugs_covered}/{len(tp_drugs)} drugs, "
                     f"{tp_diseases_covered}/{len(tp_diseases)} diseases")

    if args.tn_pairs:
        tn = pl.read_csv(args.tn_pairs, separator='\t')
        tn_drugs = set(tn['drug_id'].unique().to_list())
        tn_diseases = set(tn['disease_id'].unique().to_list())
        tn_drugs_covered = len(tn_drugs & set(embeddings.keys()))
        tn_diseases_covered = len(tn_diseases & set(embeddings.keys()))
        logger.info(f"TN coverage: {tn_drugs_covered}/{len(tn_drugs)} drugs, "
                     f"{tn_diseases_covered}/{len(tn_diseases)} diseases")

    logger.info("Node2Vec embedding generation complete.")
