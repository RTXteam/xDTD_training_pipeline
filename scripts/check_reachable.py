## Import Standard Packages
import os, sys
import graph_tool.all as gt
import argparse
import polars as pl

## Import Personal Packages
pathlist = os.getcwd().split(os.path.sep)
ROOTindex = pathlist.index("xDTD_training_pipeline")
ROOTPath = os.path.sep.join([*pathlist[:(ROOTindex + 1)]])
sys.path.append(os.path.join(ROOTPath, 'scripts'))
import utils
from utils import KnowledgeGraph, build_graph_tool_graph


def is_directly_connected(source, target):
    """True if source and target share a direct edge in G."""
    _, src_id = kg.resolve_curie(source)
    _, tgt_id = kg.resolve_curie(target)
    if src_id is None or tgt_id is None:
        return False
    return any(True for _ in gt.all_paths(G, src_id, tgt_id, cutoff=1))


def is_reachable(source, target, cutoff):
    """True if any path of length <= cutoff exists between source and target."""
    _, src_id = kg.resolve_curie(source)
    _, tgt_id = kg.resolve_curie(target)
    if src_id is None or tgt_id is None:
        return False
    for _ in gt.all_paths(G, src_id, tgt_id, cutoff=cutoff):
        return True
    return False


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", type=str, help="The path of logfile folder", default=os.path.join(ROOTPath, "log_folder"))
    parser.add_argument("--log_name", type=str, help="Log file name", default="step9_check_reachable.log")
    parser.add_argument('--tp_pairs', type=str, help='Path to true positive pairs', default=os.path.join(ROOTPath, "data", "ground_truth_pairs", "tp_pairs.txt"))
    parser.add_argument('--bandwidth', type=int, help='Maximum number of neighbors', default=3000)
    parser.add_argument('--max_path', type=int, help='Maximum length of path', default=3)
    parser.add_argument('--combined_expert_paths', type=str, help='Path to drugbank+molepro combined paths', default=os.path.join(ROOTPath, "data", "expert_path_files", "p_expert_paths_combined.txt"))
    parser.add_argument("--output_folder", type=str, help="The path of output folder", default=os.path.join(ROOTPath, "data"))
    args = parser.parse_args()

    logger = utils.get_logger(os.path.join(args.log_dir, args.log_name))
    logger.info(args)

    ## ── Build pruned knowledge graph and graph-tool Graph ───────────────
    kg = KnowledgeGraph(args.output_folder, bandwidth=args.bandwidth, logger=logger)
    G = build_graph_tool_graph(kg.graph)

    ## ── Read and canonicalize expert path pairs ─────────────────────────
    combined_df = pl.read_csv(args.combined_expert_paths, separator='\t')
    all_curies = combined_df['subject'].to_list() + combined_df['object'].to_list()
    logger.info(f"Batch-normalizing {len(set(all_curies))} unique CURIEs via Node Norm API")
    utils.batch_normalize_curies(list(set(all_curies)))
    canon_pairs = list({
        (utils.get_canonical_curie(s), utils.get_canonical_curie(o))
        for s, o in zip(combined_df['subject'].to_list(), combined_df['object'].to_list())
    })
    pair_df = pl.DataFrame(canon_pairs, schema=['source', 'n2'], orient='row')

    ## ── Keep only pairs with a direct edge in the pruned KG ─────────────
    direct_flags = [
        is_directly_connected(row[0], row[1])
        for row in pair_df.iter_rows()
    ]
    pair_df = pair_df.with_columns(pl.Series('direct', direct_flags))
    expert_pairs = pair_df.filter(pl.col('direct')).select(['source', 'n2'])

    ## ── Merge with true-positive (drug, disease) pairs ──────────────────
    tp_pairs = pl.read_csv(args.tp_pairs, separator='\t').select([
        pl.col('drug_id').alias('source'), pl.col('disease_id').alias('target'),
    ])
    expert_with_disease = expert_pairs.join(tp_pairs, on='source')

    ## ── Check n2→target reachability (path length max_path - 1) ─────────
    n2_target = expert_with_disease.select(['n2', 'target']).unique()
    logger.info(f"{n2_target.height} n2-target pairs to check for path length {args.max_path - 1}")

    n2_reachable = [
        is_reachable(row[0], row[1], args.max_path - 1)
        for row in n2_target.iter_rows()
    ]
    n2_target = n2_target.with_columns(pl.Series('reachable', n2_reachable))

    expert_with_disease = (
        expert_with_disease
        .join(n2_target, on=['n2', 'target'], how='left')
        .with_columns(pl.col('reachable').fill_null(False))
    )
    expert_with_disease.write_csv(
        os.path.join(args.output_folder, 'expert_path_files', f"reachable_expert_paths_max{args.max_path}.txt"),
        separator='\t',
    )

    ## ── Check remaining source→target pairs (full max_path) ─────────────
    reachable_tp_pairs = (
        expert_with_disease
        .filter(pl.col('reachable'))
        .select(['source', 'target'])
        .unique()
    )
    already_reachable = set(reachable_tp_pairs.iter_rows())

    remaining_pairs = [
        (s, t) for s, t in tp_pairs.select(['source', 'target']).iter_rows()
        if (s, t) not in already_reachable
    ]

    if remaining_pairs:
        remaining_df = pl.DataFrame(remaining_pairs, schema=['source', 'target'], orient='row')
        logger.info(f"{remaining_df.height} source-target pairs to check for path length {args.max_path}")

        st_reachable = [
            is_reachable(row[0], row[1], args.max_path)
            for row in remaining_df.iter_rows()
        ]
        remaining_df = remaining_df.with_columns(pl.Series('reachable', st_reachable))

        newly_reachable = remaining_df.filter(pl.col('reachable')).select(['source', 'target'])
        unreachable_tp_pairs = remaining_df.filter(~pl.col('reachable')).select(['source', 'target'])
        reachable_tp_pairs = pl.concat([reachable_tp_pairs, newly_reachable])
    else:
        unreachable_tp_pairs = pl.DataFrame(schema={'source': pl.Utf8, 'target': pl.Utf8})

    ## ── Save results ────────────────────────────────────────────────────
    reachable_tp_pairs.write_csv(
        os.path.join(args.output_folder, 'expert_path_files', f"reachable_tp_pairs_max{args.max_path}.txt"),
        separator='\t',
    )
    unreachable_tp_pairs.write_csv(
        os.path.join(args.output_folder, 'expert_path_files', f"unreachable_tp_pairs_max{args.max_path}.txt"),
        separator='\t',
    )
