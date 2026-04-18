## This script generates required input files for GraphSAGE
## (G.json, id_map.json, class_map.json, etc.)
## See https://github.com/williamleif/GraphSAGE

import json
import numpy as np
import polars as pl
import random
import os
import sys
import argparse
import pickle
from tqdm import tqdm

## Import Personal Packages
pathlist = os.getcwd().split(os.path.sep)
ROOTindex = pathlist.index("xDTD_training_pipeline")
ROOTPath = os.path.sep.join([*pathlist[:(ROOTindex + 1)]])
sys.path.append(os.path.join(ROOTPath, 'scripts'))
import utils


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", type=str, help="The path of logfile folder", default=os.path.join(ROOTPath, "log_folder"))
    parser.add_argument("--log_name", type=str, help="log file name", default="step13_graphsage_data_generation.log")
    parser.add_argument("--graph_nodes", type=str, help="Filtered graph node file", default=os.path.join(ROOTPath, "data", "filtered_graph_nodes_info.txt"))
    parser.add_argument("--graph_edges", type=str, help="Filtered graph edge file", default=os.path.join(ROOTPath, "data", "filtered_graph_edges.txt"))
    parser.add_argument('--entity2freq', type=str, help='Path to a file containing entity with frequency', default=os.path.join(ROOTPath, "data", "entity2freq.txt"))
    parser.add_argument('--type2freq', type=str, help='Path to a file containing entity type with frequency', default=os.path.join(ROOTPath, "data", "type2freq.txt"))
    parser.add_argument("--seed", type=int, help="Random seed (default: 1023)", default=1023)
    parser.add_argument("--emb_file", type=str, help="The full path of initial embedding file", default=None)
    parser.add_argument("--feature_dim", type=int, help="The node feature dimension", default=256)
    parser.add_argument("--validation_percent", type=float, help="The percentage of validation data (default: 0.3)", default=0.3)
    parser.add_argument("--output_folder", type=str, help="The path of output folder", default=os.path.join(ROOTPath, "data", "graphsage_input"))
    args = parser.parse_args()

    logger = utils.get_logger(os.path.join(args.log_dir, args.log_name))
    logger.info(args)

    os.makedirs(args.output_folder, exist_ok=True)

    ## ── Read edges and deduplicate ──────────────────────────────────────
    logger.info("Reading graph edges")
    graph_edges = pl.read_csv(args.graph_edges, separator='\t').select(['source', 'target']).unique(maintain_order=True)

    ## ── Read optional text embedding ────────────────────────────────────
    logger.info("Reading embedding file")
    if args.emb_file is not None:
        with open(args.emb_file, 'rb') as f:
            emb_file = pickle.load(f)
    else:
        emb_file = None

    ## ── Load entity / type vocabularies ─────────────────────────────────
    logger.info("Loading entity and type vocabularies")
    graph_nodes = pl.read_csv(args.graph_nodes, separator='\t')
    id_to_type = dict(zip(graph_nodes['id'].to_list(), graph_nodes['primary_category'].to_list()))

    type2id, id2type = utils.load_index(args.type2freq)
    entity2id, id2entity = utils.load_index(args.entity2freq)
    del entity2id['DUMMY_ENTITY']

    n_types = len(type2id)
    
    ## ── Build one-hot label vector per entity ───────────────────────────
    logger.info("Generating node label vectors")
    entity_label_vec = {}
    for curie, node_type in id_to_type.items():
        one_hot = np.zeros(n_types).tolist()
        one_hot[type2id[node_type]] = 1
        entity_label_vec[curie] = (node_type, one_hot)

    ## ── Write id_map.txt ────────────────────────────────────────────────
    ## index - 1 ensures alignment when external initial embeddings are included
    logger.info("Writing id_map.txt")
    id_map_df = pl.DataFrame(
        [(curie, idx - 1) for curie, idx in entity2id.items()],
        schema=['curie', 'id'],
        orient='row',
    )
    id_map_df.write_csv(os.path.join(args.output_folder, 'id_map.txt'), separator='\t')

    ## ── Write category_map.txt ──────────────────────────────────────────
    logger.info("Writing category_map.txt")
    seen_categories = {}
    for node_type, one_hot in entity_label_vec.values():
        if node_type not in seen_categories:
            seen_categories[node_type] = one_hot

    cat_map_df = pl.DataFrame({
        'category': list(seen_categories.keys()),
        'category_vec': [json.dumps(vec) for vec in seen_categories.values()],
    })
    cat_map_df.write_csv(os.path.join(args.output_folder, 'category_map.txt'), separator='\t')

    ## ── Map edge CURIEs to integer IDs ──────────────────────────────────
    curie_to_int = dict(zip(id_map_df['curie'].to_list(), id_map_df['id'].to_list()))

    graph_edges = graph_edges.with_columns(
        pl.col('source').replace(curie_to_int).cast(pl.Int64).alias('source_id'),
        pl.col('target').replace(curie_to_int).cast(pl.Int64).alias('target_id'),
    ).sort(['source_id', 'target_id'])

    ## ── Select validation set ───────────────────────────────────────────
    logger.info("Selecting validation set")
    n_entities = len(entity2id)
    all_ids = [idx - 1 for idx in entity2id.values()]
    random.seed(args.seed)
    random.shuffle(all_ids)
    valid_set = set(all_ids[:int(n_entities * args.validation_percent)])

    ## ── Build data-G.json ───────────────────────────────────────────────
    logger.info("Generating data-G.json")
    nodes = [
        {'test': False, 'id': i, 'feature': [], 'label': [], 'val': i in valid_set}
        for i in range(n_entities)
    ]

    links = [
        {'test_removed': False, 'train_removed': False, 'source': row[0], 'target': row[1]}
        for row in graph_edges.select(['source_id', 'target_id']).iter_rows()
    ]

    graph_json = {
        'directed': False,
        'graph': {'name': 'disjoint_union(,)'},
        'nodes': nodes,
        'links': links,
        'multigraph': False,
    }
    with open(os.path.join(args.output_folder, 'data-G.json'), 'w') as f:
        json.dump(graph_json, f)

    ## ── Write data-class_map.json ───────────────────────────────────────
    logger.info("Generating data-class_map.json")
    class_map = {
        str(idx - 1): entity_label_vec[id2entity[idx]][1]
        for idx in tqdm(entity2id.values(), desc="class_map")
    }
    with open(os.path.join(args.output_folder, 'data-class_map.json'), 'w') as f:
        json.dump(class_map, f)

    ## ── Write data-id_map.json ──────────────────────────────────────────
    logger.info("Generating data-id_map.json")
    id_map_json = {str(idx - 1): idx - 1 for idx in entity2id.values()}
    with open(os.path.join(args.output_folder, 'data-id_map.json'), 'w') as f:
        json.dump(id_map_json, f)

    ## ── Write data-feats.npy ────────────────────────────────────────────
    logger.info("Generating data-feats.npy")
    if emb_file is not None:
        feats = np.array([emb_file[id2entity[idx]] for idx in tqdm(entity2id.values(), desc="feats")])
        np.save(os.path.join(args.output_folder, 'data-feats.npy'), feats)
