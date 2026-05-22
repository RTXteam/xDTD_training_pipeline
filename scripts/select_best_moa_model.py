"""Select the best MOA (mechanism of action) model from trained ADAC policy checkpoints."""
import argparse
import os
import pickle
import shutil
import sys
from glob import glob

import joblib
import numpy as np
import polars as pl
import torch
from hummingbird.ml import convert
from knowledge_graph import KnowledgeGraph
from kg_env import KGEnvironment
from models import DiscriminatorActorCritic

## Import Personal Packages
pathlist = os.getcwd().split(os.path.sep)
ROOTindex = pathlist.index("xDTD_training_pipeline")
ROOTPath = os.path.sep.join([*pathlist[:(ROOTindex + 1)]])
sys.path.append(os.path.join(ROOTPath, 'scripts'))
import utils


def evaluate_model(args):
    kg = KnowledgeGraph(args, bandwidth=args.bandwidth, entity_dim=args.entity_dim, entity_type_dim=args.entity_type_dim, relation_dim=args.relation_dim, emb_dropout_rate=args.emb_dropout_rate, bucket_interval=args.bucket_interval, load_graph=True)
    ## ── Read pre-train model ───────────────────────────────────────────────────
    pretrain_model = joblib.load(args.pretrain_model_path)
    ## ── Convert sklearn model to pytorch ───────────────────────────────────────
    pretrain_model = convert(pretrain_model, 'pytorch')
    if args.use_gpu is True:
        pretrain_model.to(f"cuda:{args.gpu}")
    env = KGEnvironment(args, pretrain_model, kg, max_path_len=args.max_path, state_pre_history=args.state_history)

    def _read_pairs_as_ids(path):
        df = pl.read_csv(path, separator='\t')
        c = df.columns
        return np.column_stack([
            [kg.entity2id[x] for x in df[c[0]].to_list()],
            [kg.entity2id[x] for x in df[c[1]].to_list()],
        ])

    def _pairs_to_dict(pairs):
        d = {}
        for src, tgt in pairs:
            d.setdefault(src, set()).add(tgt)
        return {k: list(v) for k, v in d.items()}

    val_pairs = _read_pairs_as_ids(os.path.join(args.data_dir, 'RL_model_train_val_test_data', 'val_pairs.txt'))
    test_pairs = _read_pairs_as_ids(os.path.join(args.data_dir, 'RL_model_train_val_test_data', 'test_pairs.txt'))
    eval_pairs = np.vstack([val_pairs, test_pairs])
    eval_drug_disease_dict = _pairs_to_dict(eval_pairs)
    all_pairs = _read_pairs_as_ids(os.path.join(args.data_dir, 'RL_model_train_val_test_data', 'all_pairs.txt'))
    all_drug_disease_dict = _pairs_to_dict(all_pairs)

    ## ── Set up ADAC model ───────────────────────────────────────────────────────
    best_epoch = -1
    best_recall = 0
    best_precision = 0
    best_hr = 0
    sorted_model_files = sorted(glob(os.path.join(args.policy_net_folder,'*')),key=lambda path: int(os.path.basename(path).split('.')[0].replace('policy_model_epoch','')))
    for path in sorted_model_files:
        this_epoch = os.path.basename(path).split('.')[0].split('_')[-1]
        args.logger.info(f"Evaluating the model from {this_epoch}")
        model = DiscriminatorActorCritic(args, kg, args.state_history, args.gamma, args.target_update, args.ac_hidden, args.disc_hidden, args.metadisc_hidden)
        policy_net = torch.load(path, map_location=args.device)
        model_temp = model.policy_net.state_dict()
        model_temp.update(policy_net)
        model.policy_net.load_state_dict(model_temp)

        env.reset()
        args.logger.info(f"Evaluate model with test dataset:")
        results = utils.evaluate(args, eval_drug_disease_dict, env, model, all_drug_disease_dict, save_paths=args.save_pred_paths)
        if results['recall'] > best_recall and results['precision'] > best_precision and results['hr'] > best_hr:
            best_epoch = this_epoch
            best_recall = results['recall']
            best_precision = results['precision']   
            best_hr = results['hr']
    
    args.logger.info(f"Best epoch: {best_epoch}")
    best_model_save_path = os.path.join(args.policy_net_folder, 'best_moa_model.pt')
    best_epoch_path = os.path.join(args.policy_net_folder, f'policy_model_{best_epoch}.pt')
    shutil.copyfile(best_epoch_path, best_model_save_path)
    args.logger.info(f"Best model saved to {best_model_save_path}")
    ## ── Delete all other models ───────────────────────────────────────────────────
    for path in sorted_model_files:
        if path != best_epoch_path:
            os.remove(path)
            args.logger.info(f"Delete {path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    ## folder parameters
    parser.add_argument("--log_dir", type=str, help="The path of logfile folder", default=os.path.join(ROOTPath, "log_folder"))
    parser.add_argument("--log_name", type=str, help="log file name", default="step21_select_best_moa_model.log")
    parser.add_argument('--data_dir', type=str, help='Full path of data folder', default=os.path.join(ROOTPath, "data"))
    parser.add_argument('--policy_net_folder', type=str, help='The full path of folder containing trained policy network model', default=os.path.join(ROOTPath, 'models', 'ADAC_model', 'policy_net'))
    ## knowledge graph and environment parameters
    parser.add_argument('--entity_dim', type=int, help='Dimension of entity embedding', default=100)
    parser.add_argument('--relation_dim', type=int, help='Dimension of relation embedding', default=100)
    parser.add_argument('--entity_type_dim', type=int, help='Dimension of entity type embedding', default=100)
    parser.add_argument('--max_path', type=int, help='Maximum length of path', default=3)
    parser.add_argument('--bandwidth', type=int, help='Maximum number of neighbors', default=3000)
    parser.add_argument('--bucket_interval', type=int, help='adjacency list bucket size to save memory (default: 50)', default=50)
    parser.add_argument('--state_history', type=int, help='state history length', default=1)
    parser.add_argument("--emb_dropout_rate", type=float, help="Knowledge entity and relation embedding vector dropout rate (default: 0)", default=0)
    parser.add_argument("--pretrain_model_path", type=str, help="The path of pretrain model", default=os.path.join(ROOTPath, 'models', 'xgboost_model_3class', 'xgboost_model.pt'))
    parser.add_argument('--tp_reward', type=float, help='Reward if the agent hits the target entity (default: 1.0)', default=1.0)

    # discriminator parameters
    parser.add_argument('--disc_hidden', type=int, nargs='*', help='Path discriminator hidden dim parameter', default=[512, 512])
    parser.add_argument('--disc_dropout_rate', type=float, help='Path discriminator dropout rate', default=0.3)

    # metadiscriminator parameters
    parser.add_argument('--metadisc_hidden', type=int, nargs='*', help='Meta discriminator hidden dim parameters', default=[512, 256])
    parser.add_argument('--metadisc_dropout_rate', type=float, help='Meta discriminator dropout rate', default=0.3)

    # AC model parameters
    parser.add_argument('--ac_hidden', type=int, nargs='*', help='ActorCritic hidden dim parameters', default=[512, 512])
    parser.add_argument('--actor_dropout_rate', type=float, help='actor dropout rate', default=0.3)
    parser.add_argument('--critic_dropout_rate', type=float, help='critic dropout rate', default=0.3)
    parser.add_argument('--act_dropout', type=float, help='action dropout rate', default=0.5)
    parser.add_argument('--target_update', type=float, help='update ratio of target network', default=0.05)
    parser.add_argument('--gamma', type=float, help='reward discount factor', default=0.99)

    # other training parameters
    parser.add_argument('--seed', type=int, help='Random seed (default: 1023)', default=1023)
    parser.add_argument("--use_gpu", action="store_true", help="Use GPU to train model", default=False)
    parser.add_argument('--gpu', type=int, help='gpu device (default: 0)', default=0)

    # evaluation parameters
    parser.add_argument('--factor', type=float, help='decay factor for calculating path probability score (default: 0.9)', default=0.9)
    parser.add_argument('--topk', type=int, help='top ranked diseases to recommend', default=50)
    parser.add_argument('--eval_batch_size', type=int, help='batch size for evaluation step (default: 2)', default=2)
    parser.add_argument('--save_pred_paths', action="store_true", help="whether to save the predicted paths", default=False)

    args = parser.parse_args()

    ## ── Initialize logger ───────────────────────────────────────────────────────
    logger = utils.get_logger(os.path.join(args.log_dir, args.log_name))
    ## ── Check device ─────────────────────────────────────────────────────────────
    args.use_gpu, args.device = utils.check_device(logger, use_gpu=args.use_gpu, gpu=args.gpu)
    ## ── Set logger ───────────────────────────────────────────────────────────
    args.logger = logger
    logger.info(args)

    utils.set_random_seed(args.seed)

    ## ── Load entity and type vocabularies ────────────────────────────────────
    type2id, id2type = utils.load_index(os.path.join(args.data_dir, 'type2freq.txt'))
    with open(os.path.join(args.data_dir, 'entity2typeid.pkl'), 'rb') as infile:
        entity2typeid = pickle.load(infile)
        
    ## ── Find all disease ids ───────────────────────────────────────────────────
    disease_type = ['biolink:Disease', 'biolink:PhenotypicFeature']
    disease_type_ids = [type2id[x] for x in disease_type]
    args.disease_ids = [index for index, typeid in enumerate(entity2typeid) if typeid in disease_type_ids]

    ## ── Start to evaluate ADAC model ─────────────────────────────────────────────
    logger.info('Start to evaluate ADAC model')
    evaluate_model(args)