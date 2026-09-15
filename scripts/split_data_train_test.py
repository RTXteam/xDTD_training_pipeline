"""Generate a single train/test split for drug-disease prediction.

  - Single 90/10 train/test split
  - Drug-stratified splitting with singleton forcing
  - TP pairs separated into 'in expert' and 'not in expert' groups,
    each split independently to preserve expert-path proportionality
  - No y=2 synthetic negatives (generated per-shard in XGBoost ensemble script)
  - Optionally splits expert demonstration paths by train/test for RL pipeline
"""
import argparse
import math
import os
import pickle
import random
import sys

import numpy as np
import polars as pl
from sklearn.model_selection import train_test_split

## Import Personal Packages
pathlist = os.getcwd().split(os.path.sep)
ROOTindex = pathlist.index("xDTD_training_pipeline")
ROOTPath = os.path.sep.join([*pathlist[:(ROOTindex + 1)]])
sys.path.append(os.path.join(ROOTPath, 'scripts'))
import utils


def _stratified_train_test_split(pairs, test_size, label, random_state):
    """Stratified train/test split of (source, target) pairs by source drug.

    Sources with only one pair are forced into the training set so
    sklearn's stratified split has at least 2 members per stratum.
    """
    pairs = pairs.select(['source', 'target']).unique(maintain_order=True)

    source_counts = pairs.group_by('source').len()
    singleton_sources = set(
        source_counts.filter(pl.col('len') == 1)['source'].to_list()
    )

    singletons = pairs.filter(pl.col('source').is_in(singleton_sources))
    rest = pairs.filter(~pl.col('source').is_in(singleton_sources))

    if rest.height == 0:
        return (
            pairs.with_columns(pl.lit(label).alias('y')),
            pl.DataFrame(schema={'source': pl.Utf8, 'target': pl.Utf8, 'y': pl.Int64}),
        )

    unique_src = rest['source'].unique().to_list()
    src_to_cluster = {s: i for i, s in enumerate(unique_src)}
    clusters = np.array([src_to_cluster[s] for s in rest['source'].to_list()])

    n_target_train = math.ceil(pairs.height * (1.0 - test_size))
    pad_size = max(1, n_target_train - singletons.height)

    if pad_size >= rest.height:
        return (
            pairs.with_columns(pl.lit(label).alias('y')),
            pl.DataFrame(schema={'source': pl.Utf8, 'target': pl.Utf8, 'y': pl.Int64}),
        )

    n_test = rest.height - pad_size
    n_unique_clusters = len(unique_src)

    use_stratify = n_test >= n_unique_clusters and n_unique_clusters >= 2
    train_idx, test_idx = train_test_split(
        np.arange(rest.height),
        train_size=pad_size,
        random_state=random_state, shuffle=True,
        stratify=clusters if use_stratify else None,
    )

    train = pl.concat([
        singletons.select(['source', 'target']),
        rest[train_idx.tolist()].select(['source', 'target']),
    ]).with_columns(pl.lit(label).alias('y'))

    test = rest[test_idx.tolist()].select(['source', 'target']).with_columns(
        pl.lit(label).alias('y')
    )

    return train, test


def _filter_expert_paths(split_pairs, entity2id, edf_idx, expert_rel_ent):
    """Select expert-demonstration rows whose (source, target) match split_pairs."""
    src_ids = [entity2id[s] for s in split_pairs['source'].to_list()]
    tgt_ids = [entity2id[t] for t in split_pairs['target'].to_list()]

    ids = []
    for sid, tid in zip(src_ids, tgt_ids):
        matches = edf_idx.filter(
            (pl.col('0') == sid) & (pl.col('3') == tid)
        )['idx'].to_list()
        ids.extend(matches)

    return [expert_rel_ent[0][ids], expert_rel_ent[1][ids]]


def _pairs_to_set(df, src='source', tgt='target'):
    return set(zip(df[src].to_list(), df[tgt].to_list()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Generate single train/test split")
    parser.add_argument("--log_dir", type=str, default=os.path.join(ROOTPath, "log_folder"))
    parser.add_argument("--log_name", type=str, default="step11_split_data.log")
    parser.add_argument("--graph_edges", type=str,
                        default=os.path.join(ROOTPath, "data", "filtered_graph_edges.txt"))
    parser.add_argument('--tp_pairs', type=str,
                        default=os.path.join(ROOTPath, "data", "ground_truth_pairs", "tp_pairs.txt"))
    parser.add_argument('--tn_pairs', type=str,
                        default=os.path.join(ROOTPath, "data", "ground_truth_pairs", "tn_pairs.txt"))
    parser.add_argument('--entity2freq', type=str,
                        default=os.path.join(ROOTPath, "data", "entity2freq.txt"))
    parser.add_argument('--type2freq', type=str,
                        default=os.path.join(ROOTPath, "data", "type2freq.txt"))
    parser.add_argument('--entity2typeid', type=str,
                        default=os.path.join(ROOTPath, "data", "entity2typeid.pkl"))
    parser.add_argument('--filtered_expert_paths', type=str, default=None,
                        help='Filtered expert paths pickle (optional, for RL pipeline)')
    parser.add_argument('--filtered_path_relation_entity', type=str, default=None,
                        help='Expert path relation/entity pickle (optional, for RL pipeline)')
    parser.add_argument('--max_path', type=int, default=3,
                        help='Maximum path length (for expert path output naming)')
    parser.add_argument('--test_size', type=float, default=0.1,
                        help="Fraction of GT pairs in test")
    parser.add_argument('--seed', type=int, default=1023)
    parser.add_argument("--output_folder", type=str,
                        default=os.path.join(ROOTPath, "data"))
    args = parser.parse_args()

    random.seed(args.seed)

    logger = utils.get_logger(os.path.join(args.log_dir, args.log_name))
    logger.info(args)

    ## ── Build unified triple set ──────────────────────────────────────────────
    graph_edges = (
        pl.read_csv(args.graph_edges, separator='\t')
        .select(['source', 'target', 'predicate'])
        .unique(maintain_order=True)
    )
    tp_triples = pl.read_csv(args.tp_pairs, separator='\t').select([
        pl.col('drug_id').alias('source'), pl.col('disease_id').alias('target'),
    ]).with_columns(pl.lit('biolink:has_effect').alias('predicate'))
    tn_triples = pl.read_csv(args.tn_pairs, separator='\t').select([
        pl.col('drug_id').alias('source'), pl.col('disease_id').alias('target'),
    ]).with_columns(pl.lit('biolink:has_no_effect').alias('predicate'))

    all_triples = pl.concat([graph_edges, tp_triples, tn_triples]).unique(maintain_order=True)
    all_triples.write_csv(os.path.join(args.output_folder, 'all_triples.txt'), separator='\t')

    ## ── Load entity/type mappings ────────────────────────────────────────────
    entity2id, id2entity = utils.load_index(args.entity2freq)
    type2id, id2type = utils.load_index(args.type2freq)
    with open(args.entity2typeid, 'rb') as f:
        entity2typeid = pickle.load(f)

    disease_type_ids = {type2id[t] for t in ['biolink:Disease', 'biolink:PhenotypicFeature']}
    drug_type_ids = {type2id[t] for t in ['biolink:Drug', 'biolink:SmallMolecule', 'biolink:ChemicalEntity']}
    disease_ids = [id2entity[i] for i, tid in enumerate(entity2typeid) if tid in disease_type_ids]
    drug_ids = [id2entity[i] for i, tid in enumerate(entity2typeid) if tid in drug_type_ids]

    logger.info(f"TP pairs: {tp_triples.height}, TN pairs: {tn_triples.height}")
    logger.info(f"Drug entities: {len(drug_ids)}, Disease entities: {len(disease_ids)}")

    ## ── Load expert paths if provided (for RL pipeline) ──────────────────────
    expert_paths = None
    expert_rel_ent = None
    if args.filtered_expert_paths and os.path.isfile(args.filtered_expert_paths):
        with open(args.filtered_expert_paths, 'rb') as f:
            expert_paths = pickle.load(f)
        logger.info(f"Loaded expert paths: {len(expert_paths)} pairs")
    if args.filtered_path_relation_entity and os.path.isfile(args.filtered_path_relation_entity):
        with open(args.filtered_path_relation_entity, 'rb') as f:
            expert_rel_ent = pickle.load(f)
        logger.info(f"Loaded expert path relation/entity data")

    ## ── Generate split(s) ────────────────────────────────────────────────────
    pretrain_dir = os.path.join(
        args.output_folder,
        'pretrain_reward_shaping_model_train_val_test_data_3class',
    )
    os.makedirs(pretrain_dir, exist_ok=True)

    rs = np.random.RandomState(args.seed)

    ## ── Split TP pairs: separate 'in expert' and 'not in expert' groups ──
    # Expert-path TP pairs and non-expert-path TP pairs are split independently to preserve
    # proportional representation across train/test.
    if expert_paths is not None:
        expert_tp_pairs = pl.DataFrame(
            list(expert_paths.keys()), schema=['source', 'target'], orient='row')

        tp_in_expert = tp_triples.join(
            expert_tp_pairs, on=['source', 'target'], how='inner')
        tp_not_in_expert = tp_triples.join(
            expert_tp_pairs, on=['source', 'target'], how='anti')

        logger.info(f"TP separation: {tp_in_expert.height} in expert, "
                    f"{tp_not_in_expert.height} not in expert")

        train_tp_ie, test_tp_ie = _stratified_train_test_split(
            tp_in_expert, args.test_size, 1, rs)
        train_tp_nie, test_tp_nie = _stratified_train_test_split(
            tp_not_in_expert, args.test_size, 1, rs)

        train_tp = pl.concat([train_tp_ie, train_tp_nie])
        test_tp = pl.concat([test_tp_ie, test_tp_nie])

        logger.info(f"TP split: train_ie={train_tp_ie.height}, test_ie={test_tp_ie.height}, "
                    f"train_nie={train_tp_nie.height}, test_nie={test_tp_nie.height}")
    else:
        expert_tp_pairs = None
        train_tp, test_tp = _stratified_train_test_split(
            tp_triples, args.test_size, 1, rs)

    ## ── Split TN pairs ──────────────────────────────────────────────────
    train_tn, test_tn = _stratified_train_test_split(
        tn_triples, args.test_size, 0, rs)

    ## ── Combine and save pretrain data ──────────────────────────────────
    train_data = pl.concat([train_tp, train_tn]).sample(
        fraction=1.0, shuffle=True, seed=args.seed)
    test_data = pl.concat([test_tp, test_tn]).sample(
        fraction=1.0, shuffle=True, seed=args.seed)

    train_data.write_csv(os.path.join(pretrain_dir, 'train_pairs.txt'), separator='\t')
    test_data.write_csv(os.path.join(pretrain_dir, 'test_pairs.txt'), separator='\t')

    logger.info(
        f"Split: train={train_data.height} "
        f"(TP={train_data.filter(pl.col('y')==1).height}, "
        f"TN={train_data.filter(pl.col('y')==0).height}), "
        f"test={test_data.height} "
        f"(TP={test_data.filter(pl.col('y')==1).height}, "
        f"TN={test_data.filter(pl.col('y')==0).height})"
    )

    ## ── Verify every drug in test has been seen in train ──────────────────
    train_drugs = set(train_data.filter(pl.col('y') == 1)['source'].unique().to_list())
    test_drugs = set(test_data.filter(pl.col('y') == 1)['source'].unique().to_list())
    unseen_drugs = test_drugs - train_drugs
    logger.info(f"Drug coverage: {len(train_drugs)} train drugs, {len(test_drugs)} test drugs")
    if unseen_drugs:
        logger.warning(
            f"⚠ {len(unseen_drugs)} test drugs NOT in train set: "
            f"{list(unseen_drugs)[:10]}{'...' if len(unseen_drugs) > 10 else ''}"
        )
    else:
        logger.info("✓ All test drugs appear in training set (singleton drugs forced to train)")

    ## ── Split expert paths for RL pipeline ───────────────────────────────
    if expert_rel_ent is not None and expert_paths is not None:
        rel_ent_np = expert_rel_ent[1].numpy()
        col_names = [str(i) for i in range(rel_ent_np.shape[1])]
        edf_idx = pl.DataFrame(rel_ent_np, schema=col_names).with_row_index('idx')

        expert_dir = os.path.join(args.output_folder, 'expert_path_files')
        for split_name, split_pairs in [('train', train_tp_ie), ('test', test_tp_ie)]:
            if split_pairs.height > 0:
                split_expert = _filter_expert_paths(split_pairs, entity2id, edf_idx, expert_rel_ent)
                out_path = os.path.join(
                    expert_dir,
                    f'{split_name}_expert_demonstration_relation_entity_max{args.max_path}_filtered.pkl')
                with open(out_path, 'wb') as f:
                    pickle.dump(split_expert, f)
                logger.info(f"Saved {split_name} expert paths: {split_expert[1].shape[0]} entries → {out_path}")

        ## ── Also save RL-specific data ──────────────────────────────────
        rl_dir = os.path.join(args.output_folder, 'RL_model_train_val_test_data')
        os.makedirs(rl_dir, exist_ok=True)
        train_tp_ie.write_csv(os.path.join(rl_dir, 'train_pairs.txt'), separator='\t')
        test_tp_ie.write_csv(os.path.join(rl_dir, 'test_pairs.txt'), separator='\t')
        pl.concat([train_tp_ie, test_tp_ie]).write_csv(
            os.path.join(rl_dir, 'all_pairs.txt'), separator='\t')
        logger.info(f"RL pairs: train={train_tp_ie.height}, test={test_tp_ie.height}")

    logger.info("Done. Split generation complete.")
