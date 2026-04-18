## Import Standard Packages
import graph_tool.all as gt
import os, sys
import argparse
import pickle
import itertools
import torch
import polars as pl
import sqlite3
from tqdm import tqdm
import multiprocessing
import json

## Import Personal Packages
pathlist = os.getcwd().split(os.path.sep)
ROOTindex = pathlist.index("xDTD_training_pipeline")
ROOTPath = os.path.sep.join([*pathlist[:(ROOTindex + 1)]])
sys.path.append(os.path.join(ROOTPath, 'scripts'))
import utils
from utils import KnowledgeGraph, build_graph_tool_graph


# def load_curie_to_pmids(db_path):
#     conn = sqlite3.connect(db_path)
#     try:
#         df = pl.read_database("SELECT * FROM curie_to_pmids", conn)
#     finally:
#         conn.close()
#     return df


def score_path(path):
    """Return (path, ngd_score) if both intermediate nodes have PMID data, else None."""
    return (path, None)
    # if len(path) == 2:
    #     return (path, None)
    # if path[1] in ngd_mapping_dict and path[2] in ngd_mapping_dict:
    #     ngd = utils.calculate_ngd([ngd_mapping_dict[path[1]], ngd_mapping_dict[path[2]]])
    #     return (path, ngd)
    # return None


def extract_paths_for_pair(params):
    """Find all paths between (n2, target) and score them. Used by multiprocessing."""
    source, target, source_id, target_id = params
    if source_id is None or target_id is None:
        return [(source, target), []]
    scored_paths = []
    for path in gt.all_paths(G, source_id, target_id, cutoff=args.max_path - 1):
        result = score_path(list(path))
        if result is not None:
            scored_paths.append(result)
    return [(source, target), scored_paths]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", type=str, help="The path of logfile folder", default=os.path.join(ROOTPath, "log_folder"))
    parser.add_argument("--log_name", type=str, help="log file name", default="step10_generate_expert_paths.log")
    # parser.add_argument("--db_config_path", type=str, help="path to database config file", default="../config_dbs.json")
    # parser.add_argument('--ngd_threshold', type=float, help='NGD threshold to filter paths for expert paths', default=0.6)
    parser.add_argument('--reachable_expert_paths', type=str, help='Path to reachable expert paths file')
    parser.add_argument('--batch_size', type=int, help='Number of batch size for parallel running', default=500)
    parser.add_argument('--process', type=int, help='Number of processes used for parallel running', default=100)
    parser.add_argument('--biolink_version', type=str, help='Biolink model version', default='4.2.0')
    parser.add_argument('--bandwidth', type=int, help='Maximum number of neighbors', default=3000)
    parser.add_argument('--max_path', type=int, help='Maximum length of path', default=3)
    parser.add_argument("--output_folder", type=str, help="The path of output folder", default=os.path.join(ROOTPath, "data"))
    args = parser.parse_args()

    logger = utils.get_logger(os.path.join(args.log_dir, args.log_name))
    logger.info(args)

    ## ── Build pruned knowledge graph ────────────────────────────────────
    kg = KnowledgeGraph(args.output_folder, bandwidth=args.bandwidth, logger=logger)

    ## ── Get predicate depth in biolink model relation hierarchy ──────────
    predicate_list = [
        p for p in kg.relation2id
        if p not in ('DUMMY_RELATION', 'SELF_LOOP_RELATION', 'biolink:has_effect', 'biolink:has_no_effect')
    ]
    predicate_depth = utils.get_depth_of_predicate(predicate_list, biolink_version=args.biolink_version)

    ## ── Load NGD mapping (curie_id → pmid list) ─────────────────────────
    # with open(args.db_config_path, 'r') as f:
    #     config_dbs = json.load(f)
    # curie_to_pmids_name = config_dbs["database_downloads"]["curie_to_pmids"].split('/')[-1]
    # ngd_db_path = os.path.join(ROOTPath, "data", curie_to_pmids_name)
    #
    # ngd_mapping_path = os.path.join(args.output_folder, "expert_path_files", "ngd_mapping.pkl")
    # if not os.path.exists(ngd_mapping_path):
    #     if not os.path.exists(ngd_db_path):
    #         logger.error(f"No file found at {ngd_db_path}")
    #         sys.exit(1)
    #     ngd_df = load_curie_to_pmids(ngd_db_path)
    #     ngd_mapping_dict = {
    #         kg.entity2id[row['curie']]: json.loads(row['pmids'])
    #         for row in ngd_df.iter_rows(named=True)
    #         if row['curie'] in kg.entity2id
    #     }
    #     with open(ngd_mapping_path, 'wb') as f:
    #         pickle.dump(ngd_mapping_dict, f)
    # else:
    #     with open(ngd_mapping_path, 'rb') as f:
    #         ngd_mapping_dict = pickle.load(f)

    ## ── Build graph-tool Graph ──────────────────────────────────────────
    G = build_graph_tool_graph(kg.graph)
    etype = G.edge_properties['edge_type']

    ## ── Read reachable expert paths ────────────────────────────────────
    checked_expert_paths = pl.read_csv(args.reachable_expert_paths, separator='\t')
    reachable_paths = checked_expert_paths.filter(pl.col('reachable')).select(['source', 'n2', 'target'])

    # drug_target_pairs = reachable_paths.select(['source', 'n2']).unique()
    # ngd_scores = [
    #     utils.calculate_ngd([
    #         ngd_mapping_dict.get(kg.entity2id.get(src)),
    #         ngd_mapping_dict.get(kg.entity2id.get(n2)),
    #     ])
    #     for src, n2 in drug_target_pairs.iter_rows()
    # ]
    # drug_target_pairs = (
    #     drug_target_pairs
    #     .with_columns(pl.Series('ngd_score', ngd_scores))
    #     .drop_nulls()
    #     .filter(pl.col('ngd_score') <= args.ngd_threshold)
    # )
    # reachable_paths = reachable_paths.join(drug_target_pairs, on=['source', 'n2'], how='inner').drop('ngd_score')

    ## ── Extract all paths (n2→target) in parallel batches ───────────────
    n2_target_items = [
        [n2, target, kg.resolve_curie(n2)[1], kg.resolve_curie(target)[1]]
        for n2, target in reachable_paths.select(['n2', 'target']).unique().iter_rows()
    ]

    batch_starts = list(range(0, len(n2_target_items), args.batch_size))
    logger.info(f'Total batches for extracting paths: {len(batch_starts)}')

    path_dict = {}
    pool_size = None if args.process == -1 else args.process
    for batch_idx, start in enumerate(tqdm(batch_starts, desc="Extracting paths")):
        end = min(start + args.batch_size, len(n2_target_items))
        logger.info(f'Batch {batch_idx + 1}: items {start}-{end}')

        with multiprocessing.Pool(processes=pool_size) as pool:
            results = pool.map(extract_paths_for_pair, n2_target_items[start:end])
        path_dict.update({r[0]: r[1] for r in results})

        with open(os.path.join(args.output_folder, "expert_path_files", "temp_dict_backup.pkl"), 'wb') as f:
            pickle.dump(path_dict, f)

    ## ── Assemble expert demonstration paths (source→n2→…→target) ────────
    expert_paths_raw = {}
    for row in tqdm(reachable_paths.iter_rows(named=True), total=reachable_paths.height):
        source, n2, target = row['source'], row['n2'], row['target']
        sub_paths = path_dict.get((n2, target), [])
        if not sub_paths:
            continue
        source_id = kg.resolve_curie(source)[1]
        extended = [([ source_id] + p[0], p[1]) for p in sub_paths]
        if (source, target) in expert_paths_raw:
            expert_paths_raw[(source, target)] += extended
        else:
            expert_paths_raw[(source, target)] = extended

    with open(os.path.join(args.output_folder, "expert_path_files", f"expert_demonstration_paths_max{args.max_path}_raw.pkl"), 'wb') as f:
        pickle.dump(expert_paths_raw, f)

    total_paths = sum(len(v) for v in expert_paths_raw.values())
    logger.info(
        f"{total_paths} expert demonstration paths (max length {args.max_path}) "
        f"found from {len(expert_paths_raw)} true positive drug-disease pairs"
    )

    ## ── Filter paths by NGD threshold ───────────────────────────────────
    # pairs_to_remove = []
    # for pair, paths in expert_paths_raw.items():
    #     filtered = [(p, ngd) for p, ngd in paths if ngd is not None and ngd <= args.ngd_threshold]
    #     if filtered:
    #         expert_paths_raw[pair] = filtered
    #     else:
    #         pairs_to_remove.append(pair)
    # for pair in pairs_to_remove:
    #     del expert_paths_raw[pair]
    #
    # total_filtered = sum(len(v) for v in expert_paths_raw.values())
    # logger.info(
    #     f"After NGD <= {args.ngd_threshold} filtering: {total_filtered} paths "
    #     f"from {len(expert_paths_raw)} pairs"
    # )

    ## ── Add relation IDs into paths ─────────────────────────────────────
    UNINFORMATIVE_PREDICATES = {'biolink:coexists_with', 'biolink:related_to', 'biolink:part_of'}

    for pair, paths in expert_paths_raw.items():
        with_relations = []
        for node_path, ngd in paths:
            path_with_rels = [node_path[0]]
            for i in range(len(node_path) - 1):
                edge = G.edge(node_path[i], node_path[i + 1])
                path_with_rels.append(list(etype[edge]))
                path_with_rels.append(node_path[i + 1])
            with_relations.append((path_with_rels, ngd))
        expert_paths_raw[pair] = with_relations

    ## ── Translate to entity names / filter uninformative predicates ──────
    expert_paths_translated = {}
    for pair, paths in list(expert_paths_raw.items()):
        translated = []
        keep_indices = []
        for idx, (path, ngd) in enumerate(paths):
            translated_path = []
            valid = True
            for element in path:
                if isinstance(element, list):
                    relations = [
                        kg.id2relation[rid] for rid in element
                        if kg.id2relation[rid] not in UNINFORMATIVE_PREDICATES
                    ]
                    if not relations:
                        valid = False
                        break
                    max_depth = max(predicate_depth[r] for r in relations)
                    relations = [r for r in relations if predicate_depth[r] == max_depth]
                    translated_path.append(relations)
                else:
                    translated_path.append(kg.id2entity[element])
            if valid:
                keep_indices.append(idx)
                translated.append((translated_path, ngd))

        if translated:
            expert_paths_raw[pair] = [paths[i] for i in keep_indices]
            expert_paths_translated[pair] = translated
        else:
            del expert_paths_raw[pair]

    logger.info(
        f"After removing uninformative predicates and keeping max-depth predicates: "
        f"{len(expert_paths_translated)} pairs with at least one path (max length {args.max_path})"
    )
    total_translated = sum(len(v) for v in expert_paths_translated.values())
    logger.info(
        f"{total_translated} expert demonstration paths (nodes only) from "
        f"{len(expert_paths_translated)} pairs"
    )

    with open(os.path.join(args.output_folder, "expert_path_files", f"expert_demonstration_paths_max{args.max_path}_filtered.pkl"), 'wb') as f:
        pickle.dump(expert_paths_raw, f)

    with open(os.path.join(args.output_folder, "expert_path_files", f"expert_demonstration_paths_translate_max{args.max_path}_filtered.pkl"), 'wb') as f:
        pickle.dump(expert_paths_translated, f)

    ## ── Flatten into entity/relation tensors for training ────────────────
    entity_expert_paths = []
    relation_expert_paths = []
    self_loop = kg.relation2id['SELF_LOOP_RELATION']

    for pair, paths in expert_paths_translated.items():
        for path, ngd in paths:
            flattened_variants = list(itertools.product(
                *([x] if isinstance(x, str) else x for x in path)
            ))
            for flat in flattened_variants:
                entities = [kg.entity2id[flat[i]] for i in range(0, len(flat), 2)]
                relations = [kg.relation2id[flat[i]] for i in range(1, len(flat), 2)]

                if len(flat) == 7:
                    relation_expert_paths.append([self_loop] + relations)
                    entity_expert_paths.append(entities)
                elif len(flat) == 5:
                    relation_expert_paths.append([self_loop] + relations + [self_loop])
                    entity_expert_paths.append(entities + [entities[-1]])
                else:
                    logger.info(f"Unexpected path length: {flat}")

    expert_path_tensors = [torch.tensor(relation_expert_paths), torch.tensor(entity_expert_paths)]
    logger.info(
        f"{expert_path_tensors[1].shape[0]} expert demonstration paths (nodes+edges) "
        f"from {len(expert_paths_translated)} pairs"
    )

    with open(os.path.join(args.output_folder, "expert_path_files", f"expert_demonstration_relation_entity_max{args.max_path}_filtered.pkl"), 'wb') as f:
        pickle.dump(expert_path_tensors, f)
