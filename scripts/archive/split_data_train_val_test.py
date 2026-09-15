import argparse
import json
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


def _stratified_split(pairs, train_val_test_size, label, random_state):
    """Stratified train/val/test split of (source, target) pairs by source.

    Sources that appear only once are forced into the training set so
    sklearn's stratified split has at least 2 members per stratum.
    """
    pairs = pairs.select(['source', 'target']).unique(maintain_order=True)

    source_counts = pairs.group_by('source').len()
    singleton_sources = set(
        source_counts.filter(pl.col('len') == 1)['source'].to_list()
    )

    singletons = pairs.filter(pl.col('source').is_in(singleton_sources))
    rest = pairs.filter(~pl.col('source').is_in(singleton_sources))

    unique_src = rest['source'].unique().to_list()
    src_to_cluster = {s: i for i, s in enumerate(unique_src)}
    clusters = np.array([src_to_cluster[s] for s in rest['source'].to_list()])

    n_target_train = math.ceil(pairs.height * train_val_test_size[0])
    pad_size = n_target_train - singletons.height
    n_val_test = pairs.height - n_target_train

    train_idx, val_test_idx = train_test_split(
        np.arange(rest.height),
        train_size=pad_size / (pad_size + n_val_test),
        random_state=random_state, shuffle=True, stratify=clusters,
    )

    train = pl.concat([singletons, rest[train_idx.tolist()].select(['source', 'target'])])
    val_test = rest[val_test_idx.tolist()].select(['source', 'target'])

    val_idx, test_idx = train_test_split(
        np.arange(val_test.height),
        train_size=train_val_test_size[1] / (train_val_test_size[1] + train_val_test_size[2]),
        random_state=random_state, shuffle=True,
    )

    return (
        train.with_columns(pl.lit(label).alias('y')),
        val_test[val_idx.tolist()].with_columns(pl.lit(label).alias('y')),
        val_test[test_idx.tolist()].with_columns(pl.lit(label).alias('y')),
    )


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


def generate_rand_data(n, pairs, disease_list, drug_list, existing_set=None):
    """Generate random drug-disease pairs that don't overlap existing pairs."""
    tp_pairs = pairs.filter(pl.col('y') == 1)
    drugs_in_pairs = tp_pairs['source'].unique().to_list()
    diseases_in_pairs = tp_pairs['target'].unique().to_list()
    drug_pool = list(drug_list)
    disease_pool = list(disease_list)
    excluded = set(existing_set) if existing_set else set()
    random_frames = []

    for drug in drugs_in_pairs:
        n_drug = n if n is not None else tp_pairs.filter(pl.col('source') == drug).height
        selected = []
        random.shuffle(disease_pool)
        for disease in disease_pool:
            if (drug, disease) not in excluded:
                selected.append((drug, disease))
            if len(selected) == n_drug:
                break
        if selected:
            random_frames.append(pl.DataFrame(selected, schema=['source', 'target'], orient='row'))

    for disease in diseases_in_pairs:
        n_disease = n if n is not None else tp_pairs.filter(pl.col('target') == disease).height
        selected = []
        random.shuffle(drug_pool)
        for drug in drug_pool:
            if (drug, disease) not in excluded:
                selected.append((drug, disease))
            if len(selected) == n_disease:
                break
        if selected:
            random_frames.append(pl.DataFrame(selected, schema=['source', 'target'], orient='row'))

    result = pl.concat(random_frames).with_columns(pl.lit(2).alias('y'))
    print(f'Number of random pairs: {result.height}', flush=True)
    return result


def split_df_into_train_val_test(df, train_val_test_size, expert_tp_pairs, data_type='tp', seed=1024):
    rs = np.random.RandomState(seed)

    if data_type == 'tp':
        tp_pairs = df.select(['source', 'target']).unique(maintain_order=True)
        in_expert = tp_pairs.join(
            expert_tp_pairs.select(['source', 'target']),
            on=['source', 'target'], how='inner',
        )
        not_in_expert = tp_pairs.join(
            expert_tp_pairs.select(['source', 'target']),
            on=['source', 'target'], how='anti',
        )

        train_ie, val_ie, test_ie = _stratified_split(in_expert, train_val_test_size, 1, rs)
        train_nie, val_nie, test_nie = _stratified_split(not_in_expert, train_val_test_size, 1, rs)

        return [
            (train_ie, train_nie),
            (val_ie, val_nie),
            (test_ie, test_nie),
        ]
    else:
        return list(_stratified_split(df, train_val_test_size, 0, rs))


def _pairs_to_set(df, src='source', tgt='target'):
    return set(zip(df[src].to_list(), df[tgt].to_list()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", type=str, help="The path of logfile folder", default=os.path.join(ROOTPath, "log_folder"))
    parser.add_argument("--log_name", type=str, help="log file name", default="step11_split_data_train_val_test.log")
    parser.add_argument("--graph_edges", type=str, help="Filtered graph edge file", default=os.path.join(ROOTPath, "data", "filtered_graph_edges.txt"))
    parser.add_argument('--tp_pairs', type=str, help='Path to true positive pairs', default=os.path.join(ROOTPath, "data", "ground_truth_pairs", "tp_pairs.txt"))
    parser.add_argument('--tn_pairs', type=str, help='Path to true negative pairs', default=os.path.join(ROOTPath, "data", "ground_truth_pairs", "tn_pairs.txt"))
    parser.add_argument('--entity2freq', type=str, help='Entity frequency file', default=os.path.join(ROOTPath, "data", "entity2freq.txt"))
    parser.add_argument('--type2freq', type=str, help='Entity type frequency file', default=os.path.join(ROOTPath, "data", "type2freq.txt"))
    parser.add_argument('--entity2typeid', type=str, help='Entity→type-id mapping', default=os.path.join(ROOTPath, "data", "entity2typeid.pkl"))
    parser.add_argument('--filtered_expert_paths', type=str, help='Filtered expert paths pickle', default=os.path.join(ROOTPath, "data", "expert_path_files", "expert_demonstration_paths_max3_filtered.pkl"))
    parser.add_argument('--filtered_path_relation_entity', type=str, help='Expert path relation/entity pickle', default=os.path.join(ROOTPath, "data", "expert_path_files", "expert_demonstration_relation_entity_max3_filtered.pkl"))
    parser.add_argument("--train_val_test_size", type=str, help="Train/val/test proportions as JSON list", default="[0.8, 0.1, 0.1]")
    parser.add_argument('--seed', type=int, help='Random seed', default=1023)
    parser.add_argument('--max_path', type=int, help='Maximum path length', default=3)
    parser.add_argument("--output_folder", type=str, help="Output folder", default=os.path.join(ROOTPath, "data"))
    args = parser.parse_args()

    random.seed(args.seed)
    train_val_test_size = json.loads(args.train_val_test_size)
    if abs(sum(train_val_test_size) - 1.0) > 1e-9:
        raise ValueError("train_val_test_size must sum to 1")

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

    ## ── Load entity/type mappings and derive drug/disease lists ───────────────
    entity2id, id2entity = utils.load_index(args.entity2freq)
    type2id, id2type = utils.load_index(args.type2freq)
    with open(args.entity2typeid, 'rb') as f:
        entity2typeid = pickle.load(f)

    disease_type_ids = {type2id[t] for t in ['biolink:Disease', 'biolink:PhenotypicFeature']}
    drug_type_ids = {type2id[t] for t in ['biolink:Drug','biolink:SmallMolecule','biolink:ChemicalEntity']}
    disease_ids = [id2entity[i] for i, tid in enumerate(entity2typeid) if tid in disease_type_ids]
    drug_ids = [id2entity[i] for i, tid in enumerate(entity2typeid) if tid in drug_type_ids]
    ## ── Split TP/TN into train/val/test ──────────────────────────────────────
    with open(args.filtered_expert_paths, 'rb') as f:
        expert_paths = pickle.load(f)
    expert_tp_pairs = pl.DataFrame(list(expert_paths.keys()), schema=['source', 'target'], orient='row')

    (train_tp_ie, train_tp_nie), (val_tp_ie, val_tp_nie), (test_tp_ie, test_tp_nie) = \
        split_df_into_train_val_test(tp_triples, train_val_test_size, expert_tp_pairs, data_type='tp', seed=args.seed)
    train_tn, val_tn, test_tn = split_df_into_train_val_test(tn_triples, train_val_test_size, None, 'tn', seed=args.seed)

    pretrain_train = pl.concat([train_tp_ie, train_tp_nie, train_tn]).sample(fraction=1.0, shuffle=True)
    pretrain_val = pl.concat([val_tp_ie, val_tp_nie, val_tn]).sample(fraction=1.0, shuffle=True)
    pretrain_test = pl.concat([test_tp_ie, test_tp_nie, test_tn]).sample(fraction=1.0, shuffle=True)

    ## ── Generate random pairs for each split ────────────────────────────────
    for split_name, split_df_holder in [('train', [pretrain_train]), ('val', [pretrain_val]), ('test', [pretrain_test])]:
        all_existing = _pairs_to_set(pl.concat([pretrain_train, pretrain_val, pretrain_test]))
        rand = generate_rand_data(None, split_df_holder[0], disease_ids, drug_ids, all_existing)
        split_df_holder[0] = pl.concat([split_df_holder[0], rand]).sample(fraction=1.0, shuffle=True)
        if split_name == 'train':
            pretrain_train = split_df_holder[0]
        elif split_name == 'val':
            pretrain_val = split_df_holder[0]
        else:
            pretrain_test = split_df_holder[0]

    ## ── Save pretrain data ───────────────────────────────────────────────────
    pretrain_dir = os.path.join(args.output_folder, 'pretrain_reward_shaping_model_train_val_test_data_3class')
    os.makedirs(pretrain_dir, exist_ok=True)
    pretrain_train.write_csv(os.path.join(pretrain_dir, 'train_pairs.txt'), separator='\t')
    pretrain_val.write_csv(os.path.join(pretrain_dir, 'val_pairs.txt'), separator='\t')
    pretrain_test.write_csv(os.path.join(pretrain_dir, 'test_pairs.txt'), separator='\t')

    ## ── Save RL model data ───────────────────────────────────────────────────
    rl_dir = os.path.join(args.output_folder, 'RL_model_train_val_test_data')
    os.makedirs(rl_dir, exist_ok=True)

    rl_train = train_tp_ie
    rl_val = val_tp_ie
    rl_test = test_tp_ie
    rl_all = pl.concat([rl_train, rl_val, rl_test])

    rl_train.write_csv(os.path.join(rl_dir, 'train_pairs.txt'), separator='\t')
    rl_val.write_csv(os.path.join(rl_dir, 'val_pairs.txt'), separator='\t')
    rl_test.write_csv(os.path.join(rl_dir, 'test_pairs.txt'), separator='\t')
    rl_all.write_csv(os.path.join(rl_dir, 'all_pairs.txt'), separator='\t')

    ## ── Split expert demonstration paths by train/val/test ───────────────────
    with open(args.filtered_path_relation_entity, 'rb') as f:
        expert_rel_ent = pickle.load(f)

    rel_ent_np = expert_rel_ent[1].numpy()
    col_names = [str(i) for i in range(rel_ent_np.shape[1])]
    edf_idx = pl.DataFrame(rel_ent_np, schema=col_names).with_row_index('idx')

    expert_dir = os.path.join(args.output_folder, 'expert_path_files')
    for split_name, split_pairs in [('train', rl_train), ('val', rl_val), ('test', rl_test)]:
        split_expert = _filter_expert_paths(split_pairs, entity2id, edf_idx, expert_rel_ent)
        out_path = os.path.join(expert_dir, f'{split_name}_expert_demonstration_relation_entity_max{args.max_path}_filtered.pkl')
        with open(out_path, 'wb') as f:
            pickle.dump(split_expert, f)
