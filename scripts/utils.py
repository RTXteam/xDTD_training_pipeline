import collections
import functools
import logging
import logging.handlers
import math
import os
import pickle
import random
import re
import sys

import graph_tool.all as gt
import joblib
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import requests
import torch
import torch.nn as nn
from biolink_helper_pkg import BiolinkHelper
from sklearn.metrics import f1_score, precision_recall_curve
from torch.autograd import Variable
from tqdm import tqdm

from models import Transition

plt.switch_backend('agg')

## ── BiolinkHelper singleton ─────────────────────────────────────────────
@functools.lru_cache(maxsize=1)
def get_biolink_helper(biolink_version='4.2.0'):
    """Return a lazily-initialized, cached BiolinkHelper instance."""
    pathlist = os.getcwd().split(os.path.sep)
    root_index = pathlist.index("xDTD_training_pipeline")
    root_path = os.path.sep.join([*pathlist[:(root_index + 1)]])
    biolink_cache = os.path.join(root_path, "data", "biolink_cache")
    os.makedirs(biolink_cache, exist_ok=True)
    return BiolinkHelper(biolink_version=biolink_version, cached_path=biolink_cache)


def get_primary_category(categories, biolink_version='4.2.0'):
    """Pick the most specific non-mixin, non-NamedThing category using BiolinkHelper."""
    if not categories:
        return 'biolink:NamedThing'
    bh = get_biolink_helper(biolink_version)
    non_mixin = bh.filter_out_mixins(categories)
    non_mixin = [c for c in non_mixin if c != 'biolink:NamedThing']
    if not non_mixin:
        return 'biolink:NamedThing'
    if len(non_mixin) == 1:
        return non_mixin[0]
    return min(non_mixin, key=lambda c: len(bh.get_descendants(c, include_mixins=False)))


def get_leaf_categories(categories, biolink_version='4.2.0'):
    """Return all leaf (most specific) non-mixin categories from a list.

    A category is a "leaf" if no other category in the set is more specific
    (i.e. it is not an ancestor of any other category in the set).
    """
    if not categories:
        return ['biolink:NamedThing']
    bh = get_biolink_helper(biolink_version)
    non_mixin = bh.filter_out_mixins(categories)
    non_mixin = [c for c in non_mixin if c != 'biolink:NamedThing']
    if not non_mixin:
        return ['biolink:NamedThing']
    all_ancestors = set()
    for cat in non_mixin:
        ancestors = set(bh.get_ancestors(cat, include_mixins=False))
        ancestors.discard(cat)
        all_ancestors.update(ancestors)
    leaves = [c for c in non_mixin if c not in all_ancestors]
    return leaves if leaves else ['biolink:NamedThing']


## ── Node Normalization API helpers ──────────────────────────────────────
NODE_NORM_URL = 'https://nodenormalization-sri.renci.org/1.5/get_normalized_nodes'
_node_norm_cache = {}


def batch_normalize_curies(curies, batch_size=1000):
    """Pre-fill the Node Norm cache for a list of CURIEs in batches."""
    uncached = [c for c in set(curies) if c is not None and c not in _node_norm_cache]
    if not uncached:
        return
    for i in range(0, len(uncached), batch_size):
        batch = uncached[i:i + batch_size]
        try:
            resp = requests.post(
                NODE_NORM_URL,
                json={"curies": batch, "conflation": True, "drug_chemical_conflation": True},
                timeout=60,
            )
            resp.raise_for_status()
            results = resp.json()
        except Exception:
            for c in batch:
                _node_norm_cache.setdefault(c, None)
            continue
        for c in batch:
            info = results.get(c)
            if info is not None:
                _node_norm_cache[c] = {
                    'preferred_curie': info['id']['identifier'],
                    'preferred_name': info['id'].get('label'),
                    'types': info.get('type', []),
                }
            else:
                _node_norm_cache[c] = None


def get_node_norm_info(curie):
    """Get full normalized info for a CURIE via Node Norm API (cached).
    Returns dict with preferred_curie, preferred_name, types; or None."""
    if curie is None:
        return None
    if curie not in _node_norm_cache:
        batch_normalize_curies([curie])
    return _node_norm_cache.get(curie)


def get_canonical_curie(curie):
    """Get the canonical/preferred CURIE via Node Norm API (cached)."""
    info = get_node_norm_info(curie)
    return info['preferred_curie'] if info else None


## ── Constants ────────────────────────────────────────────────────────────
SELF_LOOP_RELATION = 'SELF_LOOP_RELATION'
DUMMY_RELATION = 'DUMMY_RELATION'
DUMMY_ENTITY = 'DUMMY_ENTITY'

DUMMY_RELATION_ID = 0
SELF_LOOP_RELATION_ID = 1
DUMMY_ENTITY_ID = 0
EPSILON = float(np.finfo(float).eps)
HUGE_INT = 1e31
TINY_VALUE = 1e-41
NGD_normalizer = 4.0e+7 * 20  # ~40M PubMed articles x ~20 MeSH terms each (as of 2026-04-04)


## ── Data loading helpers ─────────────────────────────────────────────────
class ACDataLoader:
    def __init__(self, indexes, batch_size, permutation=True):
        self.indexes = np.array(indexes)
        self.num_paths = len(indexes)
        self.batch_size = batch_size
        self._permutation = permutation
        self.reset()

    def reset(self):
        if self._permutation:
            self._rand_perm = np.random.permutation(self.num_paths)
        else:
            self._rand_perm = np.array(range(self.num_paths))
        self._start_idx = 0
        self._has_next = True


    def has_next(self):
        return self._has_next

    def get_batch(self):
        if not self._has_next:
            return None
        # Multiple users per batch
        end_idx = min(self._start_idx + self.batch_size, self.num_paths)
        batch_idx = self._rand_perm[self._start_idx:end_idx]
        batch_indexes = self.indexes[batch_idx]
        self._has_next = self._has_next and end_idx < self.num_paths
        self._start_idx = end_idx
        return batch_indexes.tolist()


## ── Text cleanup and NGD ─────────────────────────────────────────────────
def calculate_ngd(concept_pubmed_ids):
    """Normalized Google Distance between two sets of PubMed IDs."""
    if concept_pubmed_ids[0] is None or concept_pubmed_ids[1] is None:
        return None
    marginal_counts = [len(set(pmids)) for pmids in concept_pubmed_ids]
    joint_count = len(set(concept_pubmed_ids[0]) & set(concept_pubmed_ids[1]))
    if 0 in marginal_counts or joint_count == 0:
        return None
    try:
        log_marginals = [math.log(c) for c in marginal_counts]
        return (max(log_marginals) - math.log(joint_count)) / (math.log(NGD_normalizer) - min(log_marginals))
    except ValueError:
        return None

def clean_up_desc(string):
    if isinstance(string, str):
        string = re.sub(r"UMLS Semantic Type: UMLS_STY:[a-zA-Z][0-9]{3}[;]?", "", string).strip().strip(";")
        if string == 'None':
            return ''
        elif re.match(r"^COMMENTS: ", string):
            return re.sub(r"^COMMENTS: ", "", string)
        elif "-!- FUNCTION: " in string:
            part1 = [part for part in string.split('-!-') if re.match(r"^ FUNCTION: ", part)][0].replace(' FUNCTION: ', '')
            return re.sub(r' \{ECO:.*\}.', '', re.sub(r" \(PubMed:[0-9]*,? ?(PubMed:[0-9]*,?)?\)", "", part1))
        elif re.search(r'Check for "https://www\.cancer\.gov/', string):
            return re.sub(r'Check for "https://www\.cancer\.gov/.*" active clinical trials using this agent\. \(".*NCI Thesaurus\); ', '', string)
        else:
            return string
    elif string is None:
        return ''
    else:
        raise ValueError(f'Not expected type {type(string)}')

def clean_up_name(string):
    if isinstance(string, str):
        return '' if string == 'None' else string
    elif string is None:
        return ''
    else:
        raise ValueError(f'Not expected type {type(string)}')


## ── Torch / GPU utility functions ────────────────────────────────────────
def detach_module(mdl):
    for param in mdl.parameters():
        param.requires_grad = False

def get_expert_trans(expert, idx):
    def select_idx(e):
        if isinstance(e, list):
            if not e:
                return []
            return [x[idx] for x in e]
        elif isinstance(e, torch.Tensor):
            return e[idx]
        else:
            return []
    return Transition(*tuple([select_idx(e) for e in list(expert)]))

def load_index(input_path):
    name_to_id, id_to_name = {}, {}
    with open(input_path) as f:
        for index, line in enumerate(f.readlines()):
            name, _ = line.strip().split('\t')
            name_to_id[name] = index
            id_to_name[index] = name
    return name_to_id, id_to_name


## ── Knowledge Graph and graph-tool helpers ──────────────────────────────
class KnowledgeGraph:
    """Pruned knowledge graph backed by adjacency list and page-rank scores."""

    def __init__(self, data_dir, bandwidth=3000, logger=None):
        self.bandwidth = bandwidth
        self.entity2id, self.id2entity = load_index(os.path.join(data_dir, 'entity2freq.txt'))
        self.num_entities = len(self.entity2id)
        if logger:
            logger.info(f'Total {self.num_entities} entities loaded')
        self.relation2id, self.id2relation = load_index(os.path.join(data_dir, 'relation2freq.txt'))
        if logger:
            logger.info(f'Total {len(self.relation2id)} relations loaded')

        with open(os.path.join(data_dir, 'adj_list.pkl'), 'rb') as f:
            self.adj_list = pickle.load(f)

        self.page_rank_scores = self._load_page_rank_scores(os.path.join(data_dir, 'kg.pgrk'))
        self.graph = {src: self._get_action_space(src) for src in range(self.num_entities)}

    def _load_page_rank_scores(self, input_path):
        scores = collections.defaultdict(float)
        with open(input_path) as f:
            for line in f:
                entity, score = line.strip().split('\t')
                scores[self.entity2id[entity.strip()]] = float(score)
        return scores

    def _get_action_space(self, source):
        if source not in self.adj_list:
            return []
        action_space = [
            (relation, target)
            for relation, targets in self.adj_list[source].items()
            for target in targets
        ]
        if len(action_space) + 1 >= self.bandwidth:
            action_space.sort(key=lambda x: self.page_rank_scores[x[1]], reverse=True)
            action_space = action_space[:self.bandwidth]
        return action_space

    def resolve_curie(self, curie):
        """Resolve a CURIE to (canonical_curie, entity_id) via the Node Norm API."""
        canonical = get_canonical_curie(curie)
        if canonical is None:
            return (curie, None)
        return (canonical, self.entity2id.get(canonical))


def build_graph_tool_graph(kg_graph):
    """Build a graph-tool Graph from a KnowledgeGraph action-space dict."""
    G = gt.Graph()
    edge_relations = {}
    for source, actions in kg_graph.items():
        for relation, target in actions:
            edge_relations.setdefault((source, target), set()).add(relation)
    etype = G.new_edge_property('object')
    for (source, target), relations in edge_relations.items():
        e = G.add_edge(source, target)
        etype[e] = relations
    G.edge_properties['edge_type'] = etype
    return G


def get_depth_of_predicate(predicate_list, biolink_version='4.2.0'):
    bh = get_biolink_helper(biolink_version)
    return {predicate: len(bh.get_ancestors(predicate, include_mixins=False)) for predicate in predicate_list}


## ── Embedding / model loading helpers ────────────────────────────────────
def hist_to_vocab(_dict):
    return sorted(sorted(_dict.items(), key=lambda x: x[0]), key=lambda x: x[1], reverse=True)


def entity_load_embed(args):
    path = os.path.join(args.data_dir, 'kg_init_embeddings', 'entity_embeddings.npy')
    return torch.tensor(np.load(path)).float()


def get_graphsage_embedding(args):
    path = os.path.join(os.path.dirname(args.pretrain_model_path), 'entity_embeddings.npy')
    return torch.tensor(np.load(path)).float()


def relation_load_embed(args):
    path = os.path.join(args.data_dir, 'kg_init_embeddings', 'relation_embeddings.npy')
    return torch.tensor(np.load(path)).float()


def entity_type_load_embed(args):
    path = os.path.join(args.data_dir, 'kg_init_embeddings', 'entity_type_embeddings.npy')
    return torch.tensor(np.load(path)).float()


def batch_lookup(M, idx, vector_output=True):
    batch_size, _ = M.size()
    batch_size2, sample_size = idx.size()
    assert batch_size == batch_size2
    if sample_size == 1 and vector_output:
        return torch.gather(M, 1, idx).view(-1)
    return torch.gather(M, 1, idx)


def empty_gpu_cache(args):
    with torch.cuda.device(f'cuda:{args.gpu}'):
        torch.cuda.empty_cache()


def ones_var_cuda(s, args, requires_grad=False, use_gpu=True):
    v = Variable(torch.ones(s), requires_grad=requires_grad)
    return v.to(args.device) if use_gpu else v.long()


def zeros_var_cuda(s, args, requires_grad=False, use_gpu=True):
    v = Variable(torch.zeros(s), requires_grad=requires_grad)
    return v.to(args.device) if use_gpu else v.long()


def int_var_cuda(x, args, requires_grad=False, use_gpu=True):
    v = Variable(x, requires_grad=requires_grad).long()
    return v.to(args.device) if use_gpu else v


def var_cuda(x, args, requires_grad=False, use_gpu=True):
    v = Variable(x, requires_grad=requires_grad)
    return v.to(args.device) if use_gpu else v.long()

def pad_and_cat(a, padding_value, padding_dim=1):
    max_dim_size = max([x.size()[padding_dim] for x in a])
    padded_a = []
    for x in a:
        if x.size()[padding_dim] < max_dim_size:
            res_len = max_dim_size - x.size()[1]
            pad = nn.ConstantPad1d((0, res_len), padding_value)
            padded_a.append(pad(x))
        else:
            padded_a.append(x)
    return torch.cat(padded_a, dim=0)

def rearrange_vector_list(l, offset):
    for i, v in enumerate(l):
        l[i] = v[offset]


def tile_along_beam(v, beam_size, dim=0):
    """
    Tile a tensor along a specified dimension for the specified beam size.
    :param v: Input tensor.
    :param beam_size: Beam size.
    """
    if dim == -1:
        dim = len(v.size()) - 1
    v = v.unsqueeze(dim + 1)
    v = torch.cat([v] * beam_size, dim=dim+1)
    new_size = []
    for i, d in enumerate(v.size()):
        if i == dim + 1:
            new_size[-1] *= d
        else:
            new_size.append(d)
    return v.view(new_size)


def flatten(l):
    """Flatten nested lists/tuples into a single list."""
    flat = []
    for c in l:
        if isinstance(c, (list, tuple)):
            flat.extend(flatten(c))
        else:
            flat.append(c)
    return flat


def unique_max(unique_x, x, values, args, marker_2D=None, use_gpu=True):
    unique_interval = 2
    unique_values, unique_indices = [], []
    # prevent memory explotion during decoding
    for i in range(0, len(unique_x), unique_interval):
        unique_x_b = unique_x[i:i+unique_interval]
        marker_2D = (unique_x_b.unsqueeze(1) == x.unsqueeze(0)).float()
        values_2D = marker_2D * values.unsqueeze(0) - (1 - marker_2D) * HUGE_INT
        if use_gpu:
            empty_gpu_cache(args)
        unique_values_b, unique_idx_b = values_2D.max(dim=1)
        unique_values.append(unique_values_b)
        unique_indices.append(unique_idx_b)
    unique_values = torch.cat(unique_values)
    unique_idx = torch.cat(unique_indices)
    return unique_values, unique_idx

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


## ── Logging ──────────────────────────────────────────────────────────────
def get_logger(logname):
    logger = logging.getLogger(logname)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s  [%(levelname)s]  %(message)s', datefmt="%Y-%m-%d %H:%M:%S")
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    fh = logging.handlers.RotatingFileHandler(logname, mode='w')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger


## ── Triple / metric helpers ──────────────────────────────────────────────
def load_triples(data_path, entity_index_path, relation_index_path, group_examples_by_query=False, seen_entities=None, verbose=False):
    """
    Convert triples stored on disc into indices.
    """
    entity2id, _ = load_index(entity_index_path)
    relation2id, _ = load_index(relation_index_path)

    def triple2ids(source, target, relation):
        return entity2id[source], entity2id[target], relation2id[relation]

    triples = []
    if group_examples_by_query:
        triple_dict = {}
    with open(data_path) as f:
        num_skipped = 0
        for line in f:
            source, target, relation = line.strip().split()
            if seen_entities and (not source in seen_entities or not target in seen_entities):
                num_skipped += 1
                if verbose:
                    print('Skip triple ({}) with unseen entity: {}'.format(num_skipped, line.strip())) 
                continue

            if group_examples_by_query:
                source_id, target_id, relation_id = triple2ids(source, target, relation)
                if source_id not in triple_dict:
                    triple_dict[source_id] = {}
                if relation_id not in triple_dict[source_id]:
                    triple_dict[source_id][relation_id] = set()
                triple_dict[source_id][relation_id].add(target_id)
            else:
                triples.append(triple2ids(source, target, relation))

    if group_examples_by_query:
        for source_id in triple_dict:
            for relation_id in triple_dict[source_id]:
                triples.append((source_id, list(triple_dict[source_id][relation_id]), relation_id))
    print('{} triples loaded from {}'.format(len(triples), data_path))
    return triples

def calculate_f1score(preds, labels, average='binary'):
    y_pred_tags = np.argmax(np.array(preds), axis=1)
    return f1_score(np.array(labels), y_pred_tags, average=average)


def calculate_acc(preds, labels):
    y_pred_tags = np.argmax(np.array(preds), axis=1)
    return (y_pred_tags == np.array(labels)).astype(float).mean()

## ── Full-matrix evaluation metrics ───────────────────────────────────────

def give_recall_at_n(matrix, n_lst, bool_test_col='is_known_positive',
                     score_col='treat score', perform_sort=True,
                     out_of_matrix_mode=False):
    """Proportion of ground-truth test pairs that appear in the top-n of the matrix."""
    N = matrix.filter(pl.col(bool_test_col)).height
    if N == 0:
        return [0] * len(n_lst)
    if out_of_matrix_mode:
        matrix = matrix.filter(pl.col('in_matrix') | pl.col(bool_test_col))
    if perform_sort or out_of_matrix_mode:
        matrix = matrix.sort(by=score_col, descending=True)
    ranks = matrix.with_row_index('index').filter(pl.col(bool_test_col)).select(pl.col('index')).to_series() + 1
    return [(ranks <= n).sum() / N for n in n_lst]


def give_hit_at_k(matrix, k_max, bool_test_col='is_known_positive',
                  score_col='treat score'):
    """Hit@k: proportion of test positives whose disease-specific rank <= k."""
    test_diseases = (
        matrix.group_by('target')
        .agg(pl.col(bool_test_col).sum().alias('n_pos'))
        .filter(pl.col('n_pos') > 0)
        .select('target').to_series().to_list()
    )
    matrix = matrix.filter(pl.col('target').is_in(test_diseases))
    matrix = matrix.with_columns(
        disease_rank=pl.col(score_col).rank(descending=True, method='random').over('target')
    )
    matrix = (
        matrix.filter(pl.col(bool_test_col))
        .with_columns(
            disease_rank_among_positives=pl.col(score_col).rank(descending=True, method='dense').over('target')
        )
        .with_columns(
            disease_rank_against_negatives=(
                pl.col('disease_rank').cast(pl.Int64) - pl.col('disease_rank_among_positives').cast(pl.Int64) + 1
            ).cast(pl.UInt64)
        )
    )
    ranks_agg = (
        matrix.filter(pl.col(bool_test_col))
        .group_by('disease_rank_against_negatives').len()
        .sort('disease_rank_against_negatives')
        .with_columns(pl.col('len').cum_sum().alias('cumulative_len'))
    )
    n_test = matrix.filter(pl.col(bool_test_col)).height
    df_hit = pl.DataFrame({
        'k': ranks_agg['disease_rank_against_negatives'].cast(pl.UInt32),
        'hit_at_k': ranks_agg['cumulative_len'] / n_test,
    })
    all_k = pl.DataFrame({'k': list(range(1, k_max + 1))}, schema={'k': pl.UInt32})
    df_hit = (
        all_k.join(df_hit, on='k', how='left')
        .with_columns(pl.col('hit_at_k').forward_fill().fill_null(0.0))
    )
    return df_hit


def give_disease_specific_mrr(matrix, bool_test_col='is_known_positive',
                              score_col='treat score'):
    """Mean Reciprocal Rank across disease-specific rankings of test positives."""
    test_diseases = (
        matrix.group_by('target')
        .agg(pl.col(bool_test_col).sum().alias('n_pos'))
        .filter(pl.col('n_pos') > 0)
        .select('target').to_series().to_list()
    )
    matrix = matrix.filter(pl.col('target').is_in(test_diseases))
    matrix = matrix.with_columns(
        disease_rank=pl.col(score_col).rank(descending=True, method='random').over('target')
    )
    matrix = (
        matrix.filter(pl.col(bool_test_col))
        .with_columns(
            disease_rank_among_positives=pl.col(score_col).rank(descending=True, method='dense').over('target')
        )
        .with_columns(
            disease_rank_against_negatives=(
                pl.col('disease_rank').cast(pl.Int64) - pl.col('disease_rank_among_positives').cast(pl.Int64) + 1
            ).cast(pl.UInt64)
        )
    )
    return (1 / matrix['disease_rank_against_negatives']).mean()


def give_precision_recall_curve(matrix, bool_test_col_pos='is_known_positive',
                                bool_test_col_neg='is_known_negative',
                                score_col='treat score'):
    """Precision-recall curve over known positive and negative test pairs."""
    gt = matrix.filter(pl.col(bool_test_col_pos) | pl.col(bool_test_col_neg)).select(bool_test_col_pos, score_col)
    prec, rec, _ = precision_recall_curve(gt[bool_test_col_pos], gt[score_col])
    return prec, rec


## ── Evaluation plots ─────────────────────────────────────────────────────

def plot_av_ranking_metrics(matrices_all, model_names,
                            bool_test_col='is_known_positive',
                            score_col='treat score', perform_sort=True,
                            n_min=10, n_max=100000, n_steps=1000, k_max=100,
                            sup_title=None, force_full_y_axis=True,
                            save_path=None, **_kwargs):
    """Plot Recall@n and Hit@k for one or more models (single-matrix mode)."""
    n_lst = [int(n) for n in np.linspace(n_min, n_max, n_steps)]
    matrix_length = min(len(m) for m in matrices_all)
    n_drugs = min(len(m['source'].unique()) for m in matrices_all)

    _, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

    for name, matrix in zip(model_names, matrices_all):
        ax1.plot(n_lst, give_recall_at_n(matrix, n_lst, bool_test_col=bool_test_col,
                                         score_col=score_col, perform_sort=perform_sort), label=name)
    ax1.plot([0, matrix_length], [0, 1], 'k--', label='Random classifier', alpha=0.5)
    ax1.legend(); ax1.set_xlabel('n'); ax1.set_ylabel('Recall@n')
    ax1.set_xlim(0, n_max)
    if force_full_y_axis:
        ax1.set_ylim(0, 1)
    ax1.set_title('Recall@n vs n'); ax1.grid(True)

    for name, matrix in zip(model_names, matrices_all):
        hk = give_hit_at_k(matrix, k_max, bool_test_col=bool_test_col, score_col=score_col)
        ax2.plot(hk['k'], hk['hit_at_k'], label=name)
    ax2.plot([0, n_drugs], [0, 1], 'k--', label='Random classifier', alpha=0.5)
    ax2.legend(); ax2.set_xlabel('k'); ax2.set_ylabel('Hit@k')
    ax2.set_xlim(0, k_max)
    if force_full_y_axis:
        ax2.set_ylim(0, 1)
    ax2.set_title('Disease-specific Hit@k vs k'); ax2.grid(True)

    if sup_title:
        plt.suptitle(sup_title)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.close()


def plot_negative_metrics(matrices_all, model_names,
                          bool_pos_col='is_known_positive',
                          bool_neg_col='is_known_negative',
                          score_col='treat score', perform_sort=True,
                          n_min=10, n_max=None, n_steps=1000, k_max=None,
                          sup_title=None, force_full_y_axis=False,
                          save_path=None, **_kwargs):
    """Plot Precision-Recall curve, negative Recall@n, and negative Hit@k."""
    matrix_length = min(len(m) for m in matrices_all)
    n_drugs = min(len(m['source'].unique()) for m in matrices_all)
    if n_max is None:
        n_max = matrix_length
    if k_max is None:
        k_max = n_drugs
    n_lst = [int(n) for n in np.linspace(n_min, n_max, n_steps)]

    _, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 15))

    for name, matrix in zip(model_names, matrices_all):
        prec, rec = give_precision_recall_curve(matrix, bool_pos_col, bool_neg_col, score_col)
        ax1.plot(rec, prec, label=name)
    ax1.legend(); ax1.set_xlabel('Recall'); ax1.set_ylabel('Precision')
    ax1.set_title('Precision-Recall curve'); ax1.grid(True)

    for name, matrix in zip(model_names, matrices_all):
        ax2.plot(n_lst, give_recall_at_n(matrix, n_lst, bool_test_col=bool_neg_col,
                                         score_col=score_col, perform_sort=perform_sort))
    ax2.plot([0, n_max], [0, n_max / matrix_length], 'k--', label='Random classifier', alpha=0.5)
    ax2.set_xlabel('n'); ax2.set_ylabel('Recall@n (negatives)')
    ax2.set_title('Recall@n vs n (negatives - lower is better)')
    if force_full_y_axis:
        ax2.set_ylim(0, 1)
    ax2.legend(); ax2.grid(True)

    for name, matrix in zip(model_names, matrices_all):
        hk = give_hit_at_k(matrix, k_max, bool_test_col=bool_neg_col, score_col=score_col)
        ax3.plot(hk['k'], hk['hit_at_k'], label=name)
    ax3.plot([0, k_max], [0, k_max / n_drugs], 'k--', label='Random classifier', alpha=0.5)
    ax3.set_xlabel('k'); ax3.set_ylabel('Hit@k (negatives)')
    ax3.set_title('Disease-specific Hit@k vs k (negatives - lower is better)')
    if force_full_y_axis:
        ax3.set_ylim(0, 1)
    ax3.legend(); ax3.grid(True)

    if sup_title:
        plt.suptitle(sup_title)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.close()

## ── Beam search and evaluation ───────────────────────────────────────────
def beam_search(args, source_ids, env, model):
    """
    Beam search from source.
    """
    def top_k_action(action_weighted_prob_dist, action_space, batch_size, beam_size):
        """
        Get top k actions.
        """
        full_size = len(action_weighted_prob_dist)
        assert (full_size % batch_size == 0)
        last_k = int(full_size / batch_size)
        (r_space, e_space), _ = action_space
        action_space_size = r_space.size()[1]
        action_weighted_prob_dist = action_weighted_prob_dist.view(batch_size, -1)
        beam_action_space_size = action_weighted_prob_dist.size()[1]
        k = min(beam_size, beam_action_space_size)
        action_weighted_prob, action_ind = torch.topk(action_weighted_prob_dist, k)
        next_r = batch_lookup(r_space.view(batch_size, -1), action_ind).view(-1)
        next_e = batch_lookup(e_space.view(batch_size, -1), action_ind).view(-1)
        action_weighted_prob = action_weighted_prob.view(-1)
        action_beam_offset = torch.div(action_ind, action_space_size, rounding_mode='floor')
        if args.use_gpu:
            action_batch_offset = int_var_cuda(torch.arange(batch_size) * last_k, args=args, use_gpu=True).unsqueeze(1)
        else:
            action_batch_offset = int_var_cuda(torch.arange(batch_size) * last_k, args=args, use_gpu=False).unsqueeze(1)
        action_offset = (action_batch_offset + action_beam_offset).view(-1)
        return (next_r, next_e), action_weighted_prob, action_offset

    def top_k_answer_unique(action_weighted_prob_dist, action_space, batch_size, beam_size):
        """
        Get top k unique entities
        """
        full_size = len(action_weighted_prob_dist)
        assert (full_size % batch_size == 0)
        last_k = int(full_size / batch_size)
        (r_space, e_space), _ = action_space
        action_space_size = r_space.size()[1]

        r_space = r_space.view(batch_size, -1)
        e_space = e_space.view(batch_size, -1)
        action_weighted_prob_dist = action_weighted_prob_dist.view(batch_size, -1)
        beam_action_space_size = action_weighted_prob_dist.size()[1]
        assert (beam_action_space_size % action_space_size == 0)
        k = min(beam_size, beam_action_space_size)
        next_r_list, next_e_list = [], []
        action_prob_list = []
        action_offset_list = []
        for i in range(batch_size):
            action_weighted_prob_dist_b = action_weighted_prob_dist[i]
            r_space_b = r_space[i]
            e_space_b = e_space[i]
            if args.use_gpu:
                unique_e_space_b = var_cuda(torch.unique(e_space_b.data.cpu()), args=args, use_gpu=True)
            else:
                unique_e_space_b = var_cuda(torch.unique(e_space_b.data.cpu()), args=args, use_gpu=False)
            unique_action_weighted_prob_dist, unique_idx = unique_max(unique_e_space_b, e_space_b, action_weighted_prob_dist_b, args=args, use_gpu=args.use_gpu)
            k_prime = min(len(unique_e_space_b), k)
            top_unique_action_weighted_prob_dist, top_unique_idx2 = torch.topk(unique_action_weighted_prob_dist, k_prime)
            top_unique_idx = unique_idx[top_unique_idx2]
            top_unique_beam_offset = torch.div(top_unique_idx, action_space_size, rounding_mode='floor')
            top_r = r_space_b[top_unique_idx]
            top_e = e_space_b[top_unique_idx]
            next_r_list.append(top_r.unsqueeze(0))
            next_e_list.append(top_e.unsqueeze(0))
            action_prob_list.append(top_unique_action_weighted_prob_dist.unsqueeze(0))
            top_unique_batch_offset = i * last_k
            top_unique_action_offset = top_unique_batch_offset + top_unique_beam_offset
            action_offset_list.append(top_unique_action_offset.unsqueeze(0))
        next_r = pad_and_cat(next_r_list, padding_value=env.kg.dummy_r).view(-1)
        next_e = pad_and_cat(next_e_list, padding_value=env.kg.dummy_e).view(-1)
        action_prob = pad_and_cat(action_prob_list, padding_value=0)
        action_offset = pad_and_cat(action_offset_list, padding_value=-1)
        return (next_r, next_e), action_prob.view(-1), action_offset.view(-1)

    # Initialization
    r_s = int_var_cuda(source_ids, args=args, use_gpu=False)
    batch_size = len(r_s)
    env.initialize_path(r_s)
    num_steps = args.max_path 


    if args.use_gpu:
        action_log_weighted_prob = zeros_var_cuda(batch_size, args=args, use_gpu=True)
    else:
        action_log_weighted_prob = zeros_var_cuda(batch_size, args=args, use_gpu=False)


    print(f"", flush=True)
    for t in range(num_steps):
        print(f"Here is step {t+1} for beam search", flush=True)

        state_inputs = model.process_state(model.history_len, env._batch_curr_state)
        probs, _ = model.policy_net(state_inputs.to(args.device), env._batch_curr_action_spaces)
        if args.use_gpu:
            empty_gpu_cache(args)
        weighted_prob = action_log_weighted_prob.view(-1, 1) + args.factor**t * torch.log((probs+TINY_VALUE) * torch.count_nonzero(probs, dim=1).view(-1,1))
        if t == num_steps - 1:
            action, action_log_weighted_prob, action_offset = top_k_answer_unique(weighted_prob, env._batch_curr_action_spaces, batch_size, args.topk)
        else:
            action, action_log_weighted_prob, action_offset = top_k_action(weighted_prob, env._batch_curr_action_spaces, batch_size, args.topk)
        if args.use_gpu:
            empty_gpu_cache(args)
        env.batch_step(action, offset=action_offset)
        if args.use_gpu:
            empty_gpu_cache(args)

    output_beam_size = int(action[0].size()[0] / batch_size)
    beam_search_output = dict()
    beam_search_output['paths'] = env._batch_path
    beam_search_output['pred_prob_scores'] = action_log_weighted_prob.cpu()
    beam_search_output['output_beam_size'] = output_beam_size
    env.reset()

    return beam_search_output

def evaluate(args, drug_disease_dict, env, model, all_drug_disease_dict, save_paths=False):

    model.policy_net.eval()
    eval_drugs = torch.tensor(list(drug_disease_dict.keys()))
    args.logger.info('Evaluating model')
    with torch.no_grad():
        ## predict paths for drugs
        eval_dataloader = ACDataLoader(list(range(len(eval_drugs))), args.eval_batch_size)
        pbar = tqdm(total=eval_dataloader.num_paths)
        if save_paths:
            all_paths_r, all_paths_e, all_prob_scores = [], [], []
        pred_diseases = dict()
        while eval_dataloader.has_next():
            dids_idx = eval_dataloader.get_batch()
            source_ids = eval_drugs[dids_idx]
            res = beam_search(args, source_ids, env, model)
            if args.use_gpu:
                empty_gpu_cache(args)
            if save_paths:
                all_paths_r += [res['paths'][0]]
                all_paths_e += [res['paths'][1]]
                all_prob_scores += [res['pred_prob_scores']]
            for index in range(0,res['paths'][1].shape[0],args.topk):
                drug_id = int(res['paths'][1][index][0])
                pred_list = res['paths'][1][index:(index+args.topk),-1].tolist()
                disease_list = list(set(pred_list).intersection(set(args.disease_ids)))
                pred_diseases[drug_id] = dict()
                pred_diseases[drug_id]['list'] = disease_list
                if len(disease_list) !=0:
                    pred_diseases[drug_id]['pred_score'] = env.prob([drug_id]*len(disease_list),disease_list)
                else:
                    pred_diseases[drug_id]['pred_score'] = torch.tensor([])
            pbar.update(len(source_ids))
    
        failed_pred = 0
        avg_pred_score = []
        recalls, precisions, hits = [], [], []
        all_recalls, all_precisions, all_hits = [], [], []
        hit_disease_target_pairs = dict()
        for drug_id in drug_disease_dict.keys():
            if len(pred_diseases[drug_id]['list']) == 0:
                failed_pred += 1
                continue
            pred_list, rel_set, all_rel_set = pred_diseases[drug_id]['list'], drug_disease_dict[drug_id], all_drug_disease_dict[drug_id]

            hit_num = len(set(pred_list).intersection(set(rel_set)))
            if hit_num > 0:
                hit_disease_target_pairs[drug_id] = list(set(pred_list).intersection(set(rel_set)))
            recall = hit_num / len(rel_set)
            precision = hit_num / len(pred_list)
            hit = 1.0 if hit_num > 0 else 0.0
            recalls.append(recall)
            precisions.append(precision)
            hits.append(hit)

            all_hit_num = len(set(pred_list).intersection(set(all_rel_set)))
            all_recall = all_hit_num / len(all_rel_set)
            all_precision = all_hit_num / len(pred_list)
            all_hit = 1.0 if all_hit_num > 0 else 0.0
            all_recalls.append(all_recall)
            all_precisions.append(all_precision)
            all_hits.append(all_hit)
            avg_pred_score.append(pred_diseases[drug_id]['pred_score'].mean().item())

        args.logger.info(f'{failed_pred}/{len(drug_disease_dict.keys())} from evaluation dataset have no disease prediction')
        avg_pred_score = np.mean(avg_pred_score)
        avg_recall = np.mean(recalls) * 100
        avg_precision = np.mean(precisions) * 100
        avg_hit = np.mean(hits) * 100
        args.logger.info(f'Avg prediction score={avg_pred_score:.3f}')
        args.logger.info(f'Evaluation dataset only: Recall={avg_recall:.3f} | HR={avg_hit:.3f} | Precision={avg_precision:.3f}')
        all_avg_recall = np.mean(all_recalls) * 100
        all_avg_precision = np.mean(all_precisions) * 100
        all_avg_hit = np.mean(all_hits) * 100
        args.logger.info(f'all datasets (train, val and test): Recall={all_avg_recall:.3f} | HR={all_avg_hit:.3f} | Precision={all_avg_precision:.3f}')

        if save_paths:
            all_paths_r = torch.cat(all_paths_r)
            all_paths_e = torch.cat(all_paths_e)
            all_prob_scores = torch.cat(all_prob_scores)
            return {'paths': [all_paths_r,all_paths_e], 'prob_scores': all_prob_scores, 'hit_pairs': hit_disease_target_pairs, 'recall': avg_recall, 'precision': avg_precision, 'hr': avg_hit}
        else:
            return None


## ── Device, model loading, and CURIE helpers ─────────────────────────────
def check_device(logger, use_gpu: bool = False, gpu: int = 0):
    if use_gpu and torch.cuda.is_available():
        device = torch.device(f'cuda:{gpu}')
        torch.cuda.set_device(gpu)
        return [True, device]
    if use_gpu:
        logger.info('No GPU detected. Falling back to CPU.')
    return [False, torch.device('cpu')]


def load_graphsage_unsupervised_embeddings(data_path: str):
    file_path = os.path.join(data_path, 'graphsage_output', 'unsuprvised_graphsage_entity_embeddings.pkl')
    with open(file_path, 'rb') as infile:
        entity_embeddings_dict = pickle.load(infile)
    return entity_embeddings_dict


def load_ML_model(model_path: str):
    file_path = os.path.join(model_path, 'xgboost_model_3class', 'xgboost_model.pt')
    return joblib.load(file_path)

def load_gt_kg(kg):
    """Build a graph-tool Graph from a KG object. Returns (Graph, edge_type_property)."""
    G = build_graph_tool_graph(kg.graph)
    return G, G.edge_properties['edge_type']


def check_curie_available(logger, curie: str, available_curies_dict: dict):
    info = get_node_norm_info(curie)
    if info:
        curie = info['preferred_curie']
    if curie in available_curies_dict:
        return [True, curie]
    return [False, None]


def check_curie(curie: str, entity2id):
    if curie is None:
        return (None, None)
    info = get_node_norm_info(curie)
    preferred_curie = info['preferred_curie'] if info else None
    if preferred_curie and preferred_curie in entity2id:
        return (preferred_curie, entity2id[preferred_curie])
    return (preferred_curie, None)


def id_to_name(curie: str):
    if curie is None:
        return str(None)
    info = get_node_norm_info(curie)
    return str(info['preferred_name'] if info else None)
