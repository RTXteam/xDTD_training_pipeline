"""Train and evaluate an 3-shard XGBoost ensemble for drug-disease prediction.

Key design:
  - 3 independent XGBoost shards, each with different synthetic negatives
  - Replacement-based synthetic negative generation (n_replacements=2 per positive)
  - skopt Gaussian Process HPO with inner StratifiedShuffleSplit
  - Best HPs retrained on full TRAIN data
  - Final prediction = mean of predict_proba across shards

"""

import argparse
import json
import os
import pickle
import random
import sys

import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import xgboost as xgb
from datetime import datetime
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, confusion_matrix)
from sklearn.model_selection import StratifiedShuffleSplit
from skopt import gp_minimize
from skopt.space import Integer, Real
from tqdm import tqdm

pathlist = os.getcwd().split(os.path.sep)
ROOTindex = pathlist.index("xDTD_training_pipeline")
ROOTPath = os.path.sep.join([*pathlist[:(ROOTindex + 1)]])
sys.path.append(os.path.join(ROOTPath, 'scripts'))
import utils
from utils import (give_recall_at_n, give_hit_at_k, give_disease_specific_mrr,
                   give_precision_recall_curve,
                   plot_av_ranking_metrics, plot_negative_metrics)


# ─── Replacement-based synthetic negative generation ──────────────

def generate_replacement_negatives(positive_pairs, drug_pool, disease_pool,
                                   all_existing_pairs, n_replacements=2,
                                   seed=None):
    """Generate synthetic negative pairs by replacing drug or disease in positive pairs.

    For each positive pair (drug, disease):
      - Sample n_replacements random drugs → (random_drug, disease) with y=2
      - Sample n_replacements random diseases → (drug, random_disease) with y=2

    """
    rng = random.Random(seed)
    drug_list = list(drug_pool)
    disease_list = list(disease_pool)

    synthetic_sources, synthetic_targets = [], []

    for src, tgt in zip(positive_pairs['source'].to_list(),
                        positive_pairs['target'].to_list()):
        # Drug replacements
        replacements = 0
        rng.shuffle(drug_list)
        for d in drug_list:
            if replacements >= n_replacements:
                break
            if (d, tgt) not in all_existing_pairs:
                synthetic_sources.append(d)
                synthetic_targets.append(tgt)
                replacements += 1

        # Disease replacements
        replacements = 0
        rng.shuffle(disease_list)
        for dis in disease_list:
            if replacements >= n_replacements:
                break
            if (src, dis) not in all_existing_pairs:
                synthetic_sources.append(src)
                synthetic_targets.append(dis)
                replacements += 1

    result = pl.DataFrame({
        'source': synthetic_sources,
        'target': synthetic_targets,
        'y': [2] * len(synthetic_sources),
    }).unique(subset=['source', 'target'])

    return result


def generate_X_and_y(data_df, entity_embeddings_dict, pair_emb='concatenate'):
    """Generate feature matrix and labels from data DataFrame."""
    sources = data_df['source'].to_list()
    targets = data_df['target'].to_list()
    if pair_emb == 'concatenate':
        X = np.vstack([np.hstack([entity_embeddings_dict[s], entity_embeddings_dict[t]])
                       for s, t in zip(sources, targets)])
    elif pair_emb == 'hadamard':
        X = np.vstack([entity_embeddings_dict[s] * entity_embeddings_dict[t]
                       for s, t in zip(sources, targets)])
    else:
        raise ValueError("Only 'concatenate' or 'hadamard' is acceptable")
    y = data_df['y'].to_numpy()
    return X, y


# ─── Ensemble wrapper ────────────────────────────────────────────────────────

class XGBoostEnsemble:
    """Ensemble of XGBoost models that averages predict_proba."""

    def __init__(self, models):
        self.models = models

    def predict_proba(self, X):
        probas = np.stack([m.predict_proba(X) for m in self.models], axis=0)
        return probas.mean(axis=0)

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)


# ─── skopt Gaussian Process HPO (inner split) ───────────────────────

def train_single_shard(train_X, train_y,
                       hpo_inner_test_size=0.1,
                       n_calls=20, n_random_starts=5, seed=42,
                       device='cpu', early_stopping_rounds=0,
                       logger=None):
    """Train a single XGBoost shard using skopt GP HPO with inner split.

    StratifiedShuffleSplit(n_splits=1, test_size=0.1) within TRAIN
    for HPO evaluation.  Best HPs are then retrained on the full TRAIN data.

    Hyperparameter search:
      - learning_rate: 0.01 – 0.5
      - max_depth: 3 – 10
      - lambda (L2 reg): 0 – 1
      - n_estimators: 50 – 2500
    """
    search_space = [
        Real(0.01, 0.5, name='learning_rate'),
        Integer(3, 10, name='max_depth'),
        Real(0.0, 1.0, name='reg_lambda'),
        Integer(50, 2500, name='n_estimators'),
    ]

    xgb_extra = {'tree_method': 'hist'}
    if device.startswith('cuda'):
        xgb_extra['device'] = device
    else:
        xgb_extra['n_jobs'] = -1

    use_early_stop = early_stopping_rounds > 0

    # Inner split for HPO (EC uses StratifiedShuffleSplit n_splits=1, test_size=0.1)
    sss = StratifiedShuffleSplit(
        n_splits=1, test_size=hpo_inner_test_size, random_state=seed)
    inner_train_idx, inner_val_idx = next(sss.split(train_X, train_y))

    inner_train_X = train_X[inner_train_idx]
    inner_train_y = train_y[inner_train_idx]
    inner_val_X = train_X[inner_val_idx]
    inner_val_y = train_y[inner_val_idx]

    if logger:
        logger.info(f"  HPO inner split: train={len(inner_train_idx)}, "
                    f"val={len(inner_val_idx)} "
                    f"(test_size={hpo_inner_test_size})")
        logger.info(f"  XGBoost device={device}, "
                    f"early_stopping={'on ('+str(early_stopping_rounds)+')' if use_early_stop else 'off'}")

    best_score = [-1.0]
    trial_count = [0]

    def objective(params):
        lr, depth, reg_lam, n_est = params
        trial_count[0] += 1

        hpo_params = dict(xgb_extra)
        if use_early_stop:
            hpo_params['early_stopping_rounds'] = early_stopping_rounds

        model = xgb.XGBClassifier(
            learning_rate=lr,
            max_depth=depth,
            reg_lambda=reg_lam,
            n_estimators=n_est,
            objective='multi:softprob',
            num_class=3,
            eval_metric='mlogloss',
            random_state=seed,
            **hpo_params,
        )

        if use_early_stop:
            model.fit(
                inner_train_X, inner_train_y,
                eval_set=[(inner_val_X, inner_val_y)],
                verbose=False,
            )
            actual_iters = model.best_iteration + 1 if hasattr(model, 'best_iteration') else n_est
        else:
            model.fit(inner_train_X, inner_train_y, verbose=False)
            actual_iters = n_est

        preds = model.predict(inner_val_X)
        score = f1_score(inner_val_y, preds, average='macro')

        if score > best_score[0]:
            best_score[0] = score

        if logger and trial_count[0] % 5 == 0:
            iter_info = f", iters={actual_iters}/{n_est}" if use_early_stop else ""
            logger.info(f"  Trial {trial_count[0]}: F1={score:.5f} "
                        f"(lr={lr:.4f}, depth={depth}, lambda={reg_lam:.4f}, "
                        f"n_est={n_est}{iter_info})")

        return -score  # minimize

    result = gp_minimize(
        objective,
        search_space,
        n_calls=n_calls,
        n_random_starts=n_random_starts,
        random_state=seed,
        noise=0.01,
    )

    best_params = {
        'learning_rate': float(result.x[0]),
        'max_depth': int(result.x[1]),
        'reg_lambda': float(result.x[2]),
        'n_estimators': int(result.x[3]),
    }

    if logger:
        logger.info(f"  Best: F1={-result.fun:.5f}, params={best_params}")

    # Retrain with best params on FULL TRAIN data (inner_val is part of train)
    final_model = xgb.XGBClassifier(
        **best_params,
        objective='multi:softprob',
        num_class=3,
        eval_metric='mlogloss',
        random_state=seed,
        **xgb_extra,
    )
    final_model.fit(train_X, train_y, verbose=False)

    return final_model, best_params


# ─── Main ─────────────────────────────────────────────────────────────────────

VALID_DRUG_CATEGORIES = {'biolink:Drug', 'biolink:SmallMolecule', 'biolink:ChemicalEntity'}
VALID_DISEASE_CATEGORIES = {'biolink:Disease', 'biolink:PhenotypicFeature'}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", type=str, default=os.path.join(ROOTPath, "log_folder"))
    parser.add_argument("--log_name", type=str, default="step17_xgboost_ensemble.log")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--splits_dir", type=str, required=True)
    parser.add_argument("--pair_emb", type=str, default="concatenate")
    parser.add_argument("--drug_list", type=str, required=True)
    parser.add_argument("--disease_list", type=str, required=True)
    parser.add_argument("--seed", type=int, default=1023)
    parser.add_argument("--n_shards", type=int, default=3,
                        help="Number of ensemble shards (EC uses 3)")
    parser.add_argument("--n_replacements", type=int, default=2,
                        help="Replacements per positive pair per direction (EC uses 2)")
    parser.add_argument("--n_calls", type=int, default=20,
                        help="skopt GP optimization calls per shard (EC uses 20)")
    parser.add_argument("--n_random_starts", type=int, default=5)
    parser.add_argument("--hpo_inner_test_size", type=float, default=0.1,
                        help="Inner split test fraction for HPO (EC uses 0.1)")
    parser.add_argument("--device", type=str, default="cpu",
                        help="XGBoost device: 'cpu' or 'cuda' for GPU acceleration")
    parser.add_argument("--early_stopping_rounds", type=int, default=0,
                        help="Early stopping rounds for HPO trials (0 to disable)")
    parser.add_argument("--output_folder", type=str, required=True)
    args = parser.parse_args()

    logger = utils.get_logger(os.path.join(args.log_dir, args.log_name))
    logger.info(args)
    utils.set_random_seed(args.seed)

    # ── Load entity embeddings ─────────────────────────────────────────────
    entity_embeddings_dict = utils.load_graphsage_unsupervised_embeddings(args.data_dir)
    logger.info(f"Loaded {len(entity_embeddings_dict)} entity embeddings "
                f"(dim={len(next(iter(entity_embeddings_dict.values())))})")

    # ── Load train/test data ────────────────────────────────────────────
    train_data = pl.read_csv(os.path.join(args.splits_dir, 'train_pairs.txt'), separator='\t')
    test_data = pl.read_csv(os.path.join(args.splits_dir, 'test_pairs.txt'), separator='\t')

    # Filter to entities with embeddings
    available = set(entity_embeddings_dict.keys())

    def _filter_by_embeddings(df):
        return df.filter(pl.col('source').is_in(available) & pl.col('target').is_in(available))

    pre_train, pre_test = train_data.height, test_data.height
    train_data = _filter_by_embeddings(train_data)
    test_data = _filter_by_embeddings(test_data)
    logger.info(f"Filtered by embeddings: train {pre_train}->{train_data.height}, "
                f"test {pre_test}->{test_data.height}")
    logger.info(f"Train={train_data.height}, Test={test_data.height} "
                f"(HPO inner test_size={args.hpo_inner_test_size})")

    # ── Load drug/disease lists ──────────────────────────────────────────
    drug_list_df = pl.read_csv(args.drug_list, separator='\t')
    disease_list_df = pl.read_csv(args.disease_list, separator='\t')

    drug_list_df = drug_list_df.with_columns(
        pl.col('categories').map_elements(
            lambda c: bool(set(json.loads(c)) & VALID_DRUG_CATEGORIES),
            return_dtype=pl.Boolean
        ).alias('has_valid_drug_cat')
    )
    disease_list_df = disease_list_df.with_columns(
        pl.col('categories').map_elements(
            lambda c: bool(set(json.loads(c)) & VALID_DISEASE_CATEGORIES),
            return_dtype=pl.Boolean
        ).alias('has_valid_disease_cat')
    )
    drug_list_df = drug_list_df.filter(pl.col('has_valid_drug_cat'))
    disease_list_df = disease_list_df.filter(pl.col('has_valid_disease_cat'))

    all_drug_ids = [d for d in drug_list_df['drug_id'].unique().to_list()
                    if d in entity_embeddings_dict]
    all_disease_ids = [d for d in disease_list_df['disease_id'].unique().to_list()
                       if d in entity_embeddings_dict]
    logger.info(f"Drug pool: {len(all_drug_ids)}, Disease pool: {len(all_disease_ids)}")

    # ── Prepare base data (TP + TN only) ────────────────────────────────
    train_base = train_data.filter(pl.col('y') != 2)
    logger.info(f"Base train (y=0,1 only): {train_base.height}")

    # All existing pairs (to avoid duplicates in synthetic generation)
    all_existing = set(zip(
        pl.concat([train_data, test_data])['source'].to_list(),
        pl.concat([train_data, test_data])['target'].to_list(),
    ))

    drug_pool = set(all_drug_ids)
    disease_pool = set(all_disease_ids)

    # ── Create output directory ──────────────────────────────────────────
    folder_name = 'xgboost_model_3class'
    out_dir = os.path.join(args.output_folder, folder_name)
    os.makedirs(out_dir, exist_ok=True)

    # ── Train ensemble shards ────────────────────────────────────────────
    logger.info(f"\n{'='*60}")
    logger.info(f"Training {args.n_shards}-shard XGBoost ensemble")
    logger.info(f"{'='*60}")

    shard_models = []
    shard_params = []

    for shard_idx in range(args.n_shards):
        shard_seed = args.seed + shard_idx * 1000
        logger.info(f"\n--- Shard {shard_idx} (seed={shard_seed}) ---")

        # Generate shard-specific synthetic negatives (TRAIN only)
        train_positives = train_base.filter(pl.col('y') == 1)
        synth_train = generate_replacement_negatives(
            train_positives, drug_pool, disease_pool,
            all_existing, n_replacements=args.n_replacements,
            seed=shard_seed,
        )
        synth_train = _filter_by_embeddings(synth_train)

        shard_train = pl.concat([train_base, synth_train]).sample(
            fraction=1.0, shuffle=True, seed=shard_seed
        )

        logger.info(f"  Shard {shard_idx} train: {shard_train.height} "
                     f"(pos={shard_train.filter(pl.col('y')==1).height}, "
                     f"neg={shard_train.filter(pl.col('y')==0).height}, "
                     f"synth={shard_train.filter(pl.col('y')==2).height})")

        # Generate features
        train_X, train_y = generate_X_and_y(shard_train, entity_embeddings_dict,
                                             pair_emb=args.pair_emb)

        # Train with skopt GP HPO (inner StratifiedShuffleSplit for evaluation)
        logger.info(f"  Training shard {shard_idx} with skopt GP ({args.n_calls} calls)...")
        model, params = train_single_shard(
            train_X, train_y,
            hpo_inner_test_size=args.hpo_inner_test_size,
            n_calls=args.n_calls,
            n_random_starts=args.n_random_starts,
            seed=shard_seed,
            device=args.device,
            early_stopping_rounds=args.early_stopping_rounds,
            logger=logger,
        )

        shard_models.append(model)
        shard_params.append(params)

        # Save individual shard
        shard_dir = os.path.join(out_dir, f'shard_{shard_idx}')
        os.makedirs(shard_dir, exist_ok=True)
        joblib.dump(model, os.path.join(shard_dir, 'xgboost_model.pt'))
        with open(os.path.join(shard_dir, 'model_params.json'), 'w') as f:
            json.dump(params, f, indent=2)

    # ── Build ensemble ───────────────────────────────────────────────────
    ensemble = XGBoostEnsemble(shard_models)
    joblib.dump(ensemble, os.path.join(out_dir, 'xgboost_model.pt'))
    logger.info(f"\nEnsemble saved ({args.n_shards} shards)")

    # ── Save entity embeddings for RL pipeline compatibility ─────────────
    emb_path = os.path.join(out_dir, 'entity_embeddings.npy')
    if not os.path.exists(emb_path):
        entity2id, id2entity = utils.load_index(os.path.join(args.data_dir, 'entity2freq.txt'))
        zero_dim = len(next(iter(entity_embeddings_dict.values())))
        entity_embeddings = [
            entity_embeddings_dict[id2entity[eid]] if eid != 0 else np.zeros(zero_dim)
            for eid in id2entity
        ]
        np.save(emb_path, entity_embeddings)
        logger.info(f"Saved entity_embeddings.npy ({len(entity_embeddings)} entities, dim={zero_dim})")

    # ── Evaluate ensemble on test set ────────────────────────────────────
    logger.info(f"\n{'='*60}")
    logger.info(f"Evaluating ensemble on test set")
    logger.info(f"{'='*60}")

    test_X, test_y = generate_X_and_y(test_data, entity_embeddings_dict,
                                       pair_emb=args.pair_emb)

    test_probas = ensemble.predict_proba(test_X)
    test_preds = np.argmax(test_probas, axis=1)
    test_acc = (test_preds == test_y).mean()
    test_f1 = f1_score(test_y, test_preds, average='macro')
    logger.info(f"Test Accuracy: {test_acc:.5f}")
    logger.info(f"Test Macro F1: {test_f1:.5f}")

    for i, model in enumerate(shard_models):
        shard_probas = model.predict_proba(test_X)
        shard_preds = np.argmax(shard_probas, axis=1)
        shard_acc = (shard_preds == test_y).mean()
        shard_f1 = f1_score(test_y, shard_preds, average='macro')
        logger.info(f"  Shard {i}: acc={shard_acc:.5f}, F1={shard_f1:.5f}")

    with open(os.path.join(out_dir, 'classification_results.pkl'), 'wb') as f:
        pickle.dump({
            'test_acc': test_acc,
            'test_f1': test_f1,
            'shard_params': shard_params,
            'test_probas': test_probas,
            'test_y': test_y,
        }, f)

    # ── Build all drug × all disease evaluation matrix ────────────────────
    logger.info(f"\n{'='*60}")
    logger.info("Building full drug × disease evaluation matrix")
    logger.info(f"{'='*60}")
    logger.info(f"Matrix: {len(all_drug_ids)} drugs × {len(all_disease_ids)} diseases")

    exclude_pairs = set(zip(
        train_data['source'].to_list(), train_data['target'].to_list()
    ))

    test_pos_pairs = set(zip(
        test_data.filter(pl.col('y') == 1)['source'].to_list(),
        test_data.filter(pl.col('y') == 1)['target'].to_list(),
    ))
    test_neg_pairs = set(zip(
        test_data.filter(pl.col('y') == 0)['source'].to_list(),
        test_data.filter(pl.col('y') == 0)['target'].to_list(),
    ))
    logger.info(f"Excluding {len(exclude_pairs):,} train pairs; "
                f"{len(test_pos_pairs)} test positives, {len(test_neg_pairs)} test negatives")

    sources_all, targets_all, scores_all = [], [], []
    pos_flags, neg_flags = [], []

    for disease_id in tqdm(all_disease_ids, desc="Scoring drug-disease pairs"):
        drugs_batch = [d for d in all_drug_ids if (d, disease_id) not in exclude_pairs]
        if not drugs_batch:
            continue
        disease_emb = entity_embeddings_dict[disease_id]
        drug_embs = np.array([entity_embeddings_dict[d] for d in drugs_batch])
        if args.pair_emb == 'concatenate':
            X = np.hstack([drug_embs, np.tile(disease_emb, (len(drugs_batch), 1))])
        else:
            X = drug_embs * disease_emb

        treat_scores = ensemble.predict_proba(X)[:, 1]

        sources_all.extend(drugs_batch)
        targets_all.extend([disease_id] * len(drugs_batch))
        scores_all.extend(treat_scores.tolist())
        pos_flags.extend((d, disease_id) in test_pos_pairs for d in drugs_batch)
        neg_flags.extend((d, disease_id) in test_neg_pairs for d in drugs_batch)

    matrix = pl.DataFrame({
        'source': sources_all,
        'target': targets_all,
        'treat score': scores_all,
        'is_known_positive': pos_flags,
        'is_known_negative': neg_flags,
    })
    logger.info(f"Matrix: {matrix.height:,} pairs, "
                f"{matrix.filter(pl.col('is_known_positive')).height} test positives, "
                f"{matrix.filter(pl.col('is_known_negative')).height} test negatives")
    matrix.write_parquet(os.path.join(out_dir, 'evaluation_matrix.parquet'))

    # ── Recall@n ─────────────────────────────────────────────────────────
    n_max = matrix.height
    log_targets = [100, 1000, 10000, 100000]
    n_lst = sorted(set(
        [int(n) for n in np.linspace(100, n_max, 1000)]
        + [t for t in log_targets if t <= n_max]
    ))
    recall_scores = give_recall_at_n(matrix, n_lst)
    recall_map = dict(zip(n_lst, recall_scores))
    for target_n in log_targets:
        if target_n <= n_max:
            logger.info(f"Recall@{target_n}: {recall_map[target_n]:.5f}")

    # ── Hit@k ────────────────────────────────────────────────────────────
    k_max = min(len(all_drug_ids), 200)
    hit_at_k_df = give_hit_at_k(matrix, k_max)
    for k in [1, 5, 10, 20, 50, 100]:
        if k > k_max:
            break
        row = hit_at_k_df.filter(pl.col('k') == k)
        if row.height > 0:
            logger.info(f"Hit@{k}: {row['hit_at_k'][0]:.5f}")

    # ── MRR ──────────────────────────────────────────────────────────────
    mrr = give_disease_specific_mrr(matrix)
    logger.info(f"Disease-specific MRR: {mrr:.5f}")

    # ── Save ranking metrics to JSON for cross-setting comparison ────────
    ranking_metrics = {
        'test_acc': float(test_acc),
        'test_f1': float(test_f1),
        'mrr': float(mrr),
    }
    for target_n in log_targets:
        if target_n <= n_max:
            ranking_metrics[f'recall_at_{target_n}'] = float(recall_map[target_n])
    for k in [1, 5, 10, 20, 50, 100]:
        if k <= k_max:
            row = hit_at_k_df.filter(pl.col('k') == k)
            if row.height > 0:
                ranking_metrics[f'hit_at_{k}'] = float(row['hit_at_k'][0])
    with open(os.path.join(out_dir, 'ranking_metrics.json'), 'w') as f:
        json.dump(ranking_metrics, f, indent=2)

    # ── Draw evaluation plots ────────────────────────────────────────────
    plot_av_ranking_metrics(
        matrices_all=(matrix,),
        model_names=("XGBoost Ensemble",),
        n_max=100000, k_max=100,
        sup_title="XGBoost Ensemble Evaluation",
        save_path=os.path.join(out_dir, 'ranking_metrics.png'),
    )
    plot_negative_metrics(
        matrices_all=(matrix,),
        model_names=("XGBoost Ensemble",),
        n_max=n_max, k_max=k_max,
        sup_title="XGBoost Ensemble Negative Metrics",
        save_path=os.path.join(out_dir, 'negative_metrics.png'),
    )

    # ── Additional evaluation plots ──────────────────────────────────────

    # Treat score distribution
    pos_scores = matrix.filter(pl.col('is_known_positive'))['treat score'].to_numpy()
    neg_scores = matrix.filter(pl.col('is_known_negative'))['treat score'].to_numpy()
    other_scores = matrix.filter(
        ~pl.col('is_known_positive') & ~pl.col('is_known_negative')
    )['treat score'].to_numpy()

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(other_scores, bins=100, alpha=0.4, label='Unknown', color='gray', density=True)
    ax.hist(neg_scores, bins=100, alpha=0.6, label='Test Negative (TN)', color='tab:red', density=True)
    ax.hist(pos_scores, bins=100, alpha=0.6, label='Test Positive (TP)', color='tab:blue', density=True)
    ax.set_xlabel('Treat Score (P(y=1))')
    ax.set_ylabel('Density')
    ax.set_title('Treat Score Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'treat_score_distribution.png'), dpi=150)
    plt.close()
    logger.info("Saved treat_score_distribution.png")

    # Precision-Recall curve
    prec, rec = give_precision_recall_curve(matrix)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(rec, prec, linewidth=2, label='XGBoost Ensemble')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curve')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'precision_recall_curve.png'), dpi=150)
    plt.close()
    logger.info("Saved precision_recall_curve.png")

    # ── Generate Markdown evaluation report ──────────────────────────────
    report_dir = os.path.join(ROOTPath, 'reports')
    os.makedirs(report_dir, exist_ok=True)
    figures_dir = os.path.join(report_dir, 'figures')
    os.makedirs(figures_dir, exist_ok=True)

    # Copy plots to report figures directory
    import shutil
    for fname in ['treat_score_distribution.png', 'precision_recall_curve.png',
                  'ranking_metrics.png', 'negative_metrics.png']:
        src = os.path.join(out_dir, fname)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(figures_dir, fname))

    # Compute renormalized 2-class metrics (exclude y=2 synthetic pairs from test)
    test_mask_2class = test_y != 2
    test_y_2class = test_y[test_mask_2class]
    test_preds_2class = test_preds[test_mask_2class]
    test_probas_2class = test_probas[test_mask_2class]

    acc_2class = accuracy_score(test_y_2class, test_preds_2class)
    f1_2class = f1_score(test_y_2class, test_preds_2class, average='macro')
    prec_2class = precision_score(test_y_2class, test_preds_2class, average='macro', zero_division=0)
    rec_2class = recall_score(test_y_2class, test_preds_2class, average='macro', zero_division=0)

    # 3-class metrics
    acc_3class = accuracy_score(test_y, test_preds)
    f1_3class = f1_score(test_y, test_preds, average='macro')
    cm_3class = confusion_matrix(test_y, test_preds)

    report_lines = [
        "# XGBoost Ensemble — Evaluation Report",
        "",
        f"_Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}_",
        "",
        "## 1. Experimental Setup",
        "",
        "### Data",
        f"- **Train pairs**: {train_data.height:,} "
        f"(TP: {train_data.filter(pl.col('y')==1).height:,}, "
        f"TN: {train_data.filter(pl.col('y')==0).height:,})",
        f"- **Test pairs**: {test_data.height:,} "
        f"(TP: {test_data.filter(pl.col('y')==1).height:,}, "
        f"TN: {test_data.filter(pl.col('y')==0).height:,})",
        f"- **Drug list (filtered)**: {len(all_drug_ids):,} drugs",
        f"- **Disease list**: {len(all_disease_ids):,} diseases",
        f"- **Evaluation matrix**: {len(all_drug_ids):,} × {len(all_disease_ids):,} "
        f"= {matrix.height:,} drug-disease pairs",
        "",
        "### Model Settings",
        f"- **Ensemble shards**: {args.n_shards}",
        f"- **Pair embedding**: {args.pair_emb}",
        f"- **Embedding dim**: {len(next(iter(entity_embeddings_dict.values())))}",
        f"- **Synthetic negatives per shard**: "
        f"~{shard_models[0].n_features_in_ if hasattr(shard_models[0], 'n_features_in_') else 'N/A'} features, "
        f"n_replacements={args.n_replacements}",
        f"- **HPO**: skopt GP, {args.n_calls} calls, "
        f"inner test_size={args.hpo_inner_test_size}",
        f"- **Seed**: {args.seed}",
        "",
    ]

    # Per-shard HPO results
    report_lines.extend([
        "### Per-Shard Hyperparameters",
        "",
        "| Shard | learning_rate | max_depth | reg_lambda | n_estimators |",
        "|:-----:|:------------:|:---------:|:----------:|:------------:|",
    ])
    for i, params in enumerate(shard_params):
        report_lines.append(
            f"| {i} | {params['learning_rate']:.4f} | {params['max_depth']} | "
            f"{params['reg_lambda']:.4f} | {params['n_estimators']} |"
        )

    report_lines.extend([
        "",
        "## 2. Classification Performance",
        "",
        "### 3-Class Metrics (TP / TN / Synthetic)",
        "",
        "| Metric | Value |",
        "|--------|:-----:|",
        f"| Accuracy | {acc_3class:.4f} |",
        f"| Macro F1 | {f1_3class:.4f} |",
        "",
        "### Confusion Matrix (3-class)",
        "",
        "| | Pred TP | Pred TN | Pred Synth |",
        "|---|:---:|:---:|:---:|",
    ])
    for i, label in enumerate(['Actual TP', 'Actual TN', 'Actual Synth']):
        if i < cm_3class.shape[0]:
            row = cm_3class[i]
            report_lines.append(f"| {label} | {row[0]} | {row[1]} | {row[2] if len(row) > 2 else 0} |")

    report_lines.extend([
        "",
        "### Renormalized 2-Class Metrics (TP vs TN only)",
        "",
        "| Metric | Value |",
        "|--------|:-----:|",
        f"| Accuracy | {acc_2class:.4f} |",
        f"| Macro F1 | {f1_2class:.4f} |",
        f"| Macro Precision | {prec_2class:.4f} |",
        f"| Macro Recall | {rec_2class:.4f} |",
        "",
        "## 3. Ranking Performance (Drug Repurposing Evaluation)",
        "",
        "### MRR & Hit@K",
        "",
        "| Metric | Value |",
        "|--------|:-----:|",
        f"| MRR | {mrr:.5f} |",
    ])
    for k in [1, 5, 10, 20, 50, 100]:
        key = f'hit_at_{k}'
        if key in ranking_metrics:
            report_lines.append(f"| Hit@{k} | {ranking_metrics[key]:.5f} |")

    report_lines.extend([
        "",
        "### Recall@N",
        "",
        "| N | Recall@N |",
        "|--:|:--------:|",
    ])
    for target_n in log_targets:
        key = f'recall_at_{target_n}'
        if key in ranking_metrics:
            report_lines.append(f"| {target_n:,} | {ranking_metrics[key]:.5f} |")

    report_lines.extend([
        "",
        "## 4. Visualization",
        "",
        "![Treat Score Distribution](figures/treat_score_distribution.png)",
        "",
        "![Precision-Recall Curve](figures/precision_recall_curve.png)",
        "",
        "![Ranking Performance](figures/ranking_metrics.png)",
        "",
        "![Negative Metrics](figures/negative_metrics.png)",
        "",
    ])

    report_path = os.path.join(report_dir, 'evaluation_report.md')
    with open(report_path, 'w') as f:
        f.write('\n'.join(report_lines) + '\n')
    logger.info(f"Evaluation report saved to {report_path}")

    # Also save ranking metrics with additional 2-class metrics
    ranking_metrics['acc_2class'] = float(acc_2class)
    ranking_metrics['f1_2class'] = float(f1_2class)
    ranking_metrics['prec_2class'] = float(prec_2class)
    ranking_metrics['rec_2class'] = float(rec_2class)
    with open(os.path.join(out_dir, 'ranking_metrics.json'), 'w') as f:
        json.dump(ranking_metrics, f, indent=2)

    logger.info("\nXGBoost ensemble training and evaluation done.")
