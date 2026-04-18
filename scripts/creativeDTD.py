"""Creative Drug Target Discovery: predict drugs and explanatory paths for a disease."""

import itertools
import os
import pickle
import sys

import graph_tool.all as gt
import numpy as np
import polars as pl
import torch
from hummingbird.ml import convert
from tqdm import tqdm

from knowledge_graph import KnowledgeGraph
from kg_env import KGEnvironment
from models import DiscriminatorActorCritic

## Import Personal Packages
pathlist = os.getcwd().split(os.path.sep)
ROOTindex = pathlist.index("xDTD_training_pipeline")
ROOTPath = os.path.sep.join([*pathlist[:(ROOTindex + 1)]])
sys.path.append(os.path.join(ROOTPath, 'scripts'))
import utils

UNINFORMATIVE_PREDICATES = frozenset([
    'biolink:related_to', 'biolink:part_of',
    'biolink:coexists_with', 'biolink:contraindicated_for',
])


class creativeDTD:

    def __init__(self, args, data_path: str, model_path: str):
        self.args = args
        self.args.data_dir = data_path
        device_flag, device = utils.check_device(logger=args.logger, use_gpu=args.use_gpu, gpu=args.gpu)
        self.args.use_gpu = device_flag
        self.args.device = device
        args.logger.info(f"Use device {device}")

        ## Load datasets
        self.entity_embeddings_dict = utils.load_graphsage_unsupervised_embeddings(data_path)
        self.args.entity2id, self.args.id2entity = utils.load_index(os.path.join(data_path, 'entity2freq.txt'))
        self.args.relation2id, self.args.id2relation = utils.load_index(os.path.join(data_path, 'relation2freq.txt'))
        self.args.type2id, self.args.id2type = utils.load_index(os.path.join(data_path, 'type2freq.txt'))
        with open(os.path.join(data_path, 'entity2typeid.pkl'), 'rb') as f:
            self.args.entity2typeid = pickle.load(f)
        with open(os.path.join(data_path, 'filtered_drug_nodes_for_precomputation.pkl'), 'rb') as f:
            self.drug_curie_ids = pickle.load(f)

        ## Load ML model
        self.ML_model = utils.load_ML_model(model_path)
        pretrain_model = convert(self.ML_model, 'pytorch')

        ## Load RL model
        self.kg = KnowledgeGraph(
            args, bandwidth=args.bandwidth, entity_dim=args.entity_dim,
            entity_type_dim=args.entity_type_dim, relation_dim=args.relation_dim,
            emb_dropout_rate=args.emb_dropout_rate, bucket_interval=args.bucket_interval,
            load_graph=True,
        )
        if self.args.use_gpu:
            utils.empty_gpu_cache(args)
        self.env = KGEnvironment(args, pretrain_model, self.kg, max_path_len=args.max_path, state_pre_history=args.state_history)
        self.RL_model = DiscriminatorActorCritic(
            args, self.kg, args.state_history, args.gamma, args.target_update,
            args.ac_hidden, args.disc_hidden, args.metadisc_hidden,
        )
        policy_path = os.path.join(model_path, 'ADAC_model', 'policy_net', 'best_moa_model.pt')
        policy_state = torch.load(policy_path, map_location=device)
        state_dict = self.RL_model.policy_net.state_dict()
        state_dict.update(policy_state)
        self.RL_model.policy_net.load_state_dict(state_dict)
        del policy_state, state_dict
        if self.args.use_gpu:
            utils.empty_gpu_cache(args)

        ## Load graph-tool graph
        self.G, self.etype = utils.load_gt_kg(self.kg)

        ## Relation IDs to filter out of paths
        self._filter_edge_ids = np.array([
            self.args.relation2id[e] for e in UNINFORMATIVE_PREDICATES
            if e in self.args.relation2id
        ])

        self.disease_curie = None
        self.top_N_drugs = None

    def set_query_disease(self, disease_curie: str):
        ok, normalized = utils.check_curie_available(
            logger=self.args.logger, curie=disease_curie,
            available_curies_dict=self.args.entity2id,
        )
        if ok:
            self.disease_curie = normalized
            return True
        self.args.logger.warning(f"Can't find curie {disease_curie}")
        return False

    def predict_top_N_drugs(self, N: int = 50, threshold: float = 0.6):
        if not self.disease_curie:
            self.args.logger.warning("No disease curie set. Call set_query_disease() first.")
            return None

        self.args.logger.info(f"Predicting top {N} drugs for disease {self.disease_curie}")

        drug_embs = np.array([self.entity_embeddings_dict[d] for d in self.drug_curie_ids])
        disease_emb = self.entity_embeddings_dict[self.disease_curie]
        X = np.hstack([drug_embs, np.tile(disease_emb, (len(self.drug_curie_ids), 1))])
        probs = self.ML_model.predict_proba(X)

        self.top_N_drugs = (
            pl.DataFrame({
                'drug_id': self.drug_curie_ids,
                'disease_id': [self.disease_curie] * len(self.drug_curie_ids),
                'tn_score': probs[:, 0].tolist(),
                'tp_score': probs[:, 1].tolist(),
                'unknown_score': probs[:, 2].tolist(),
            })
            .sort('tp_score', descending=True)
            .with_columns(
                pl.col('drug_id').map_elements(utils.id_to_name, return_dtype=pl.Utf8).alias('drug_name'),
                pl.col('disease_id').map_elements(utils.id_to_name, return_dtype=pl.Utf8).alias('disease_name'),
            )
            .select(['drug_id', 'drug_name', 'disease_id', 'disease_name', 'tn_score', 'tp_score', 'unknown_score'])
            .filter(pl.col('tp_score') >= threshold)
            .head(N)
        )

        if self.top_N_drugs.height > 0:
            return self.top_N_drugs
        self.args.logger.warning(f"No predictions above threshold {threshold} for {self.disease_curie}")
        return None

    def _filter_edge_tensor(self, edge_mat, node_mat):
        """Remove paths containing uninformative predicates."""
        if edge_mat.shape[0] == 0 or len(self._filter_edge_ids) == 0:
            return edge_mat, node_mat
        edge_arr = edge_mat.numpy()
        mask = ~np.any(np.isin(edge_arr[:, 1:], self._filter_edge_ids), axis=1)
        keep = np.where(mask)[0]
        return edge_mat[keep], node_mat[keep]

    def _extract_all_paths(self):
        if self.top_N_drugs is None:
            self.args.logger.warning("Call predict_top_N_drugs() before extracting paths.")
            return False

        self.args.logger.info(f"Extracting all paths (max len 3) for disease {self.top_N_drugs['disease_id'][0]}")
        self.filtered_res_all_paths = {}
        self_loop = self.args.relation2id['SELF_LOOP_RELATION']

        for row in self.top_N_drugs.iter_rows(named=True):
            source, target = row['drug_id'], row['disease_id']
            src_id = utils.check_curie(source, self.args.entity2id)[1]
            tgt_id = utils.check_curie(target, self.args.entity2id)[1]
            if src_id is None or tgt_id is None:
                continue

            entity_paths, relation_paths = [], []
            for path in gt.all_paths(self.G, src_id, tgt_id, cutoff=3):
                path = list(path)
                path_with_rels = [path[0]]
                for i in range(len(path) - 1):
                    path_with_rels.append(list(self.etype[self.G.edge(path[i], path[i + 1])]))
                    path_with_rels.append(path[i + 1])

                for flat in itertools.product(
                    *([x] if not isinstance(x, list) else x for x in path_with_rels)
                ):
                    if len(flat) == 7:
                        relation_paths.append([self_loop] + [flat[i] for i in (1, 3, 5)])
                        entity_paths.append([flat[i] for i in (0, 2, 4, 6)])
                    elif len(flat) == 5:
                        relation_paths.append([self_loop, flat[1], flat[3], self_loop])
                        entity_paths.append([flat[0], flat[2], flat[4], flat[4]])

            if not entity_paths:
                continue
            edge_mat = torch.tensor(relation_paths)
            node_mat = torch.tensor(np.array(entity_paths, dtype=int))
            edge_mat, node_mat = self._filter_edge_tensor(edge_mat, node_mat)
            if edge_mat.shape[0] > 0:
                self.filtered_res_all_paths[(source, target)] = [edge_mat, node_mat]

        return True

    def _make_path(self, rel_vec, ent_vec, score):
        segments = [
            self.args.id2entity[ent_vec[i]] + '->' + self.args.id2relation[rel_vec[i + 1]]
            for i in range(len(ent_vec) - 1)
        ]
        segments.append(self.args.id2entity[ent_vec[-1]])
        return ['->'.join(segments), score]

    def _batch_get_true(self, batch_action_spaces, batch_true_actions):
        (batch_r_space, batch_e_space), _ = batch_action_spaces
        device = self.args.device
        true_r = batch_true_actions[0].view(-1, 1).to(device)
        true_e = batch_true_actions[1].view(-1, 1).to(device)
        true_idx = torch.where((batch_r_space == true_r) * (batch_e_space == true_e))[1]
        return true_idx, (true_r, true_e)

    def _select_true_action(self, model, batch_state, batch_action_spaces, batch_true_actions):
        device = self.args.device
        state_inputs = model.process_state(model.history_len, batch_state).to(device)
        true_idx, true_next_actions = self._batch_get_true(batch_action_spaces, batch_true_actions)

        probs, _ = model.policy_net(state_inputs, batch_action_spaces)
        if self.args.use_gpu:
            utils.empty_gpu_cache(self.args)
        true_prob = probs.gather(1, true_idx.to(device).view(-1, 1)).view(-1)
        weighted_logprob = torch.log(
            (true_prob.view(-1, 1) + utils.TINY_VALUE) * torch.count_nonzero(probs, dim=1).view(-1, 1)
        )
        return true_next_actions, weighted_logprob

    def _batch_calculate_prob_score(self, batch_paths):
        args = self.args
        self.env.reset()
        self.RL_model.policy_net.eval()
        dataloader = utils.ACDataLoader(list(range(batch_paths[1].shape[0])), args.batch_size, permutation=False)

        pred_scores = []
        while dataloader.has_next():
            batch_ids = dataloader.get_batch()
            source_ids = batch_paths[1][batch_ids][:, 0]
            self.env.initialize_path(source_ids)
            act_num = 1

            log_prob = utils.zeros_var_cuda(len(batch_ids), args, use_gpu=args.use_gpu)

            while not self.env._done:
                batch_true = [batch_paths[0][batch_ids][:, act_num], batch_paths[1][batch_ids][:, act_num]]
                true_next, weighted_logprob = self._select_true_action(
                    self.RL_model, self.env._batch_curr_state,
                    self.env._batch_curr_action_spaces, batch_true,
                )
                self.env.batch_step(true_next)
                if args.use_gpu:
                    utils.empty_gpu_cache(args)
                log_prob = log_prob.view(-1, 1) + args.factor ** (act_num - 1) * weighted_logprob
                act_num += 1

            pred_scores.append(log_prob.view(-1).cpu().detach())
            self.env.reset()
            if args.use_gpu:
                utils.empty_gpu_cache(args)

        return np.concatenate(pred_scores)

    def predict_top_M_paths(self, M: int = 10):
        if not self._extract_all_paths():
            return None

        self.args.logger.info("Calculating path scores")

        for source, target in tqdm(self.filtered_res_all_paths, desc="Scoring paths"):
            edge_mat, node_mat = self.filtered_res_all_paths[(source, target)]
            if node_mat.shape[0] == 0:
                continue

            scores = torch.tensor(self._batch_calculate_prob_score([edge_mat, node_mat]))
            sorted_scores, indices = torch.sort(scores, descending=True)
            edge_sorted = edge_mat[indices]
            node_sorted = node_mat[indices]

            seen, top_indices = set(), []
            for i, nodes in enumerate(node_sorted.numpy()):
                key = tuple(nodes)
                if key not in seen:
                    seen.add(key)
                top_indices.append(i)
                if len(seen) >= M:
                    break

            self.filtered_res_all_paths[(source, target)] = [
                self._make_path(
                    edge_sorted[i].numpy(), node_sorted[i].numpy(),
                    sorted_scores[i].numpy().item(),
                )
                for i in top_indices
            ]

        return self.filtered_res_all_paths
