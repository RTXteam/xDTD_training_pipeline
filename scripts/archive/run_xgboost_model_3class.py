"""Train and evaluate an XGBoost 3-class drug-disease prediction model."""

import argparse
import os
import pickle
import sys

import joblib
import numpy as np
import optuna
import polars as pl
import xgboost as xgb
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

from tqdm import tqdm

## Import Personal Packages
pathlist = os.getcwd().split(os.path.sep)
ROOTindex = pathlist.index("xDTD_training_pipeline")
ROOTPath = os.path.sep.join([*pathlist[:(ROOTindex + 1)]])
sys.path.append(os.path.join(ROOTPath, 'scripts'))
import utils
from utils import (give_recall_at_n, give_hit_at_k, give_disease_specific_mrr,
                   plot_av_ranking_metrics, plot_negative_metrics)


def generate_X_and_y(data_df, entity_embeddings_dict, pair_emb='concatenate'):
    sources = data_df['source'].to_list()
    targets = data_df['target'].to_list()
    if pair_emb == 'concatenate':
        X = np.vstack([np.hstack([entity_embeddings_dict[s], entity_embeddings_dict[t]]) for s, t in zip(sources, targets)])
    elif pair_emb == 'hadamard':
        X = np.vstack([entity_embeddings_dict[s] * entity_embeddings_dict[t] for s, t in zip(sources, targets)])
    else:
        raise ValueError("Only 'concatenate' or 'hadamard' is acceptable")
    y = data_df['y'].to_numpy()
    return X, y


def evaluate(model, X, y_true, calculate_metric=True):
    probas = model.predict_proba(X)
    if calculate_metric:
        acc = utils.calculate_acc(probas, y_true)
        macro_f1 = utils.calculate_f1score(probas, y_true, 'macro')
        micro_f1 = utils.calculate_f1score(probas, y_true, 'micro')
        return acc, macro_f1, micro_f1, y_true, probas
    return None, None, None, y_true, probas


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", type=str, help="The path of logfile folder", default=os.path.join(ROOTPath, "log_folder"))
    parser.add_argument("--log_name", type=str, help="log file name", default="step17_xgboost_3class.log")
    parser.add_argument("--data_dir", type=str, help="The path of data folder", default=os.path.join(ROOTPath, "data"))
    parser.add_argument("--pair_emb", type=str, help="The method for the pair embedding (concatenate or hadamard).", default="concatenate")
    parser.add_argument("--drug_list", type=str, required=True, help="Path to curated drug_list.txt from step4")
    parser.add_argument("--disease_list", type=str, required=True, help="Path to curated disease_list.txt from step4")
    parser.add_argument('--seed', type=int, help='Random seed (default: 1023)', default=1023)
    parser.add_argument('--n_trials', type=int, help='Number of Optuna hyperparameter search trials (default: 25)', default=25)
    parser.add_argument('--n_startup_trials', type=int, help='Optuna pruner startup trials before pruning begins (default: 5)', default=5)
    parser.add_argument('--early_stopping_rounds', type=int, help='XGBoost early stopping patience (default: 30)', default=30)
    parser.add_argument("--output_folder", type=str, help="The path of output folder", default=os.path.join(ROOTPath, "models"))
    args = parser.parse_args()

    logger = utils.get_logger(os.path.join(args.log_dir, args.log_name))
    logger.info(args)
    utils.set_random_seed(args.seed)

    ## ── Load GraphSAGE unsupervised entity embeddings ────────────────────
    entity_embeddings_dict = utils.load_graphsage_unsupervised_embeddings(args.data_dir)

    ## ── Load train/test data (val + test combined as test set) ────────────────
    data_dir = os.path.join(args.data_dir, 'pretrain_reward_shaping_model_train_val_test_data_3class')
    train_data = pl.read_csv(os.path.join(data_dir, 'train_pairs.txt'), separator='\t')
    val_data = pl.read_csv(os.path.join(data_dir, 'val_pairs.txt'), separator='\t')
    test_data_orig = pl.read_csv(os.path.join(data_dir, 'test_pairs.txt'), separator='\t')
    test_data = pl.concat([val_data, test_data_orig])
    logger.info(f"Combined val ({val_data.height}) + test ({test_data_orig.height}) = {test_data.height} pairs as test set")

    ## ── Generate X and y for train/test data ──────────────────────────────────
    train_X, train_y = generate_X_and_y(train_data, entity_embeddings_dict, pair_emb=args.pair_emb)
    test_X, test_y = generate_X_and_y(test_data, entity_embeddings_dict, pair_emb=args.pair_emb)

    ## ── Create output directory ─────────────────────────────────────────────
    folder_name = 'xgboost_model_3class'
    out_dir = os.path.join(args.output_folder, folder_name)
    os.makedirs(out_dir, exist_ok=True)

    ## ── Save entity embeddings ───────────────────────────────────────────────
    emb_path = os.path.join(out_dir, 'entity_embeddings.npy')
    if not os.path.exists(emb_path):
        entity2id, id2entity = utils.load_index(os.path.join(args.data_dir, 'entity2freq.txt'))
        zero_dim = len(entity_embeddings_dict[id2entity[1]])
        entity_embeddings = [entity_embeddings_dict[id2entity[eid]] if eid != 0 else np.zeros(zero_dim) for eid in id2entity]
        np.save(emb_path, entity_embeddings)

    ## ── Calculate class weights ───────────────────────────────────────────────
    class_counts = np.bincount(train_y)
    class_weights = len(train_y) / (len(class_counts) * class_counts)
    sample_weights = class_weights[train_y]

    ## ── Train XGBoost model (Optuna Bayesian hyperparameter search) ────────
    logger.info('Start training model')

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)

    def objective(trial):
        params = {
            'max_depth': trial.suggest_int('max_depth', 3, 35),
            'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.3, log=True),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'gamma': trial.suggest_float('gamma', 0.0, 5.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
        }
        fold_scores = []
        for fold, (tr_idx, va_idx) in enumerate(skf.split(train_X, train_y)):
            model = xgb.XGBClassifier(
                **params,
                n_estimators=2000,
                objective='multi:softprob',
                num_class=3,
                eval_metric='mlogloss',
                early_stopping_rounds=args.early_stopping_rounds,
                random_state=args.seed,
                n_jobs=-1,
                tree_method='hist',
            )
            model.fit(
                train_X[tr_idx], train_y[tr_idx],
                sample_weight=sample_weights[tr_idx],
                eval_set=[(train_X[va_idx], train_y[va_idx])],
                sample_weight_eval_set=[sample_weights[va_idx]],
                verbose=False,
            )
            preds = model.predict(train_X[va_idx])
            fold_scores.append(f1_score(train_y[va_idx], preds, average='macro'))

            trial.report(np.mean(fold_scores), fold)
            if trial.should_prune():
                raise optuna.TrialPruned()

        return np.mean(fold_scores)

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=args.seed),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=args.n_startup_trials, n_warmup_steps=1),
    )
    study.optimize(objective, n_trials=args.n_trials)
    logger.info(f"Best trial: value={study.best_trial.value:.5f}, params={study.best_trial.params}")
    with open(os.path.join(out_dir, 'optuna_study.pkl'), 'wb') as f:
        pickle.dump(study, f)

    best_params = study.best_trial.params
    fitModel = xgb.XGBClassifier(
        **best_params,
        n_estimators=2000,
        objective='multi:softprob',
        num_class=3,
        eval_metric='mlogloss',
        early_stopping_rounds=args.early_stopping_rounds,
        random_state=args.seed,
        n_jobs=-1,
        tree_method='hist',
    )
    fitModel.fit(
        train_X, train_y,
        sample_weight=sample_weights,
        eval_set=[(test_X, test_y)],
        verbose=False,
    )
    logger.info(f"Final model: best_iteration={fitModel.best_iteration}")
    joblib.dump(fitModel, os.path.join(out_dir, 'xgboost_model.pt'))

    ## ── Evaluate best model ──────────────────────────────────────────────────
    logger.info('')
    logger.info('#### Evaluate best model ####')
    train_acc, train_macro_f1, train_micro_f1, train_y_true, train_y_probs = evaluate(fitModel, train_X, train_y)
    test_acc, test_macro_f1, test_micro_f1, test_y_true, test_y_probs = evaluate(fitModel, test_X, test_y)

    train_data = train_data.with_columns(
        pl.Series('prob', train_y_probs[:, 1]),
        pl.Series('pred_y', np.argmax(train_y_probs, axis=1)),
    )
    test_data = test_data.with_columns(
        pl.Series('prob', test_y_probs[:, 1]),
        pl.Series('pred_y', np.argmax(test_y_probs, axis=1)),
    )

    ## ── Save results ─────────────────────────────────────────────────────────
    with open(os.path.join(out_dir, 'xgboost_results.pkl'), 'wb') as f:
        pickle.dump([train_data, test_data], f)

    ## ── Calculate classification metrics ────────────────────────────────────────
    logger.info(f'Accuracy: Train={train_acc:.5f}, Test={test_acc:.5f}')
    logger.info(f'Macro F1: Train={train_macro_f1:.5f}, Test={test_macro_f1:.5f}')
    logger.info(f'Micro F1: Train={train_micro_f1:.5f}, Test={test_micro_f1:.5f}')

    ## ── Scale to 2-class classification (for comparison with 2-class models) ──
    logger.info('Scaled to 2-class classification (for comparison with 2-class models)')
    mask_train = (train_data['y'] != 2).to_numpy()
    mask_test = (test_data['y'] != 2).to_numpy()
    train_acc_2c = utils.calculate_acc(train_y_probs[mask_train][:, :2], train_y_true[mask_train])
    test_acc_2c = utils.calculate_acc(test_y_probs[mask_test][:, :2], test_y_true[mask_test])
    logger.info(f'2-class Accuracy: Train={train_acc_2c:.8f}, Test={test_acc_2c:.8f}')
    train_f1_2c = utils.calculate_f1score(train_y_probs[mask_train][:, :2], train_y_true[mask_train], None).mean()
    test_f1_2c = utils.calculate_f1score(test_y_probs[mask_test][:, :2], test_y_true[mask_test], None)[:2].mean()
    logger.info(f'2-class Macro F1: Train={train_f1_2c:.8f}, Test={test_f1_2c:.8f}')

    all_eval = {
        'evaluation_acc_score': [train_acc, test_acc],
        'evaluation_macro_f1_score': [train_macro_f1, test_macro_f1],
        'evaluation_micro_f1_score': [train_micro_f1, test_micro_f1],
        'evaluation_y_true': [train_y_true, test_y_true],
        'evaluation_y_probas': [train_y_probs, test_y_probs],
    }
    with open(os.path.join(out_dir, 'classification_results.pkl'), 'wb') as f:
        pickle.dump(all_eval, f)

    ## ── Build all drug × all disease evaluation matrix ─────────────────────────
    logger.info('')
    logger.info('#### All drug × all disease matrix evaluation ####')

    drug_list_df = pl.read_csv(args.drug_list, separator='\t')
    disease_list_df = pl.read_csv(args.disease_list, separator='\t')
    all_drug_ids = [d for d in drug_list_df['drug_id'].unique().to_list() if d in entity_embeddings_dict]
    all_disease_ids = [d for d in disease_list_df['disease_id'].unique().to_list() if d in entity_embeddings_dict]
    logger.info(f"Matrix: {len(all_drug_ids)} drugs × {len(all_disease_ids)} diseases = {len(all_drug_ids) * len(all_disease_ids):,} pairs")

    exclude_pairs = set(
        zip(train_data['source'].to_list(), train_data['target'].to_list())
    )
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
        treat_scores = fitModel.predict_proba(X)[:, 1]

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
    logger.info(f"Matrix built: {matrix.height:,} pairs, "
                f"{matrix.filter(pl.col('is_known_positive')).height} test positives, "
                f"{matrix.filter(pl.col('is_known_negative')).height} test negatives")
    matrix.write_parquet(os.path.join(out_dir, 'evaluation_matrix.parquet'))

    ## ── Recall@n (full-matrix ranking) ──────────────────────────────────────────
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

    ## ── Hit@k (disease-specific ranking) ────────────────────────────────────────
    k_max = min(len(all_drug_ids), 200)
    hit_at_k_df = give_hit_at_k(matrix, k_max)
    for k in [1, 5, 10, 20, 50, 100]:
        if k > k_max:
            break
        row = hit_at_k_df.filter(pl.col('k') == k)
        if row.height > 0:
            logger.info(f"Hit@{k}: {row['hit_at_k'][0]:.5f}")

    ## ── MRR (disease-specific ranking) ──────────────────────────────────────────
    mrr = give_disease_specific_mrr(matrix)
    logger.info(f"Disease-specific MRR: {mrr:.5f}")

    ## ── Draw evaluation plots ───────────────────────────────────────────────────
    plot_av_ranking_metrics(
        matrices_all=(matrix,),
        model_names=("XGBoost 3-class",),
        n_max=100000,
        k_max=100,
        sup_title="XGBoost 3-class Evaluation",
        save_path=os.path.join(out_dir, 'ranking_metrics.png'),
    )
    plot_negative_metrics(
        matrices_all=(matrix,),
        model_names=("XGBoost 3-class",),
        n_max=n_max,
        k_max=k_max,
        sup_title="XGBoost 3-class Negative Metrics",
        save_path=os.path.join(out_dir, 'negative_metrics.png'),
    )
