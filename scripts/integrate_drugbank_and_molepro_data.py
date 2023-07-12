## Import Standard Packages
import pandas as pd
import os, sys
import argparse
from tqdm import tqdm, trange
import json
import requests
import asyncio
import httpx
from multiprocessing import Pool
from typing import Optional, List
import traceback
import time
import subprocess
import csv
csv.field_size_limit(sys.maxsize)

## Import Personal Packages
pathlist = os.getcwd().split(os.path.sep)
ROOTindex = pathlist.index("xDTD_training_pipeline")
ROOTPath = os.path.sep.join([*pathlist[:(ROOTindex + 1)]])
sys.path.append(os.path.join(ROOTPath, 'scripts'))
import utils
from node_synonymizer import NodeSynonymizer
synonymizer = NodeSynonymizer()

# class MoleProData:

#     def __init__(self, logger, molepro_api_link: str = 'https://molepro-trapi.transltr.io/molepro/trapi/v1.3'):
#         """
#         Initial Method
#         """
#         ## setup basic information
#         self.molepro_api_link = molepro_api_link
#         self.logger = logger
#         # self.client = httpx.Client()

#     async def _call_async_melepro(self, client, curie_id: str, category: str):
#             try:
#                 request_body = self._generate_query_graph(curie_id, category)
#                 resp = await client.post(f'{self.molepro_api_link}/query', json = request_body, headers={'accept': 'application/json'})
#             except Exception:
#                 # print(f"ERROR: {curie_id}", flush=True)
#                 self.logger.error(f"{curie_id}")
#                 return pd.DataFrame([], columns=['subject','object','pmid'])

#             if not resp.status_code == 200:
#                 # print(f"WARNING: {curie_id} fails to call molepro api with status code {resp.status_code}")
#                 self.logger.warning(f"{curie_id} fails to call molepro api with status code {resp.status_code}")
#                 return pd.DataFrame([], columns=['subject','object','pmid'])

#             resp_res = resp.json()
#             temp_pairs = self._extract_drug_target_pairs_from_kg(resp_res['message']['knowledge_graph'])
    
#             return temp_pairs

#     async def _get_data(self, param_list: List):
#         async with httpx.AsyncClient(timeout=120) as client:
#             tasks = [asyncio.create_task(self._call_async_melepro(client, curie_id, category)) for curie_id, category in param_list]
#             self.logger.info("starting to extract data from molepro api")
#             temp_results = await asyncio.gather(*tasks)
#             self.results = pd.concat(temp_results).reset_index(drop=True)
#             self.logger.info("Extracted data from molepro api done")

#     @staticmethod
#     def _generate_query_graph(curie_id, category):
#         if type(curie_id) is str:
#             query_id = [curie_id]
#         else:
#             query_id = curie_id

#         query_graph = {
#             "message": {
#                 "query_graph": {
#                 "edges": {
#                     "e00": {
#                     "subject": "n00",
#                     "predicates": [
#                         "biolink:affects",
#                         "biolink:interacts_with"
#                     ],
#                     "object": "n01"
#                     }
#                 },
#                 "nodes": {
#                     "n00": {
#                     "ids": query_id,
#                     "categories": [
#                         category
#                     ]
#                     },
#                     "n01": {
#                     "categories": [
#                         "biolink:Gene",
#                         "biolink:Protein"
#                     ]
#                     }
#                 }
#                 }
#             }
#         }

#         return query_graph

#     @staticmethod
#     def _extract_drug_target_pairs_from_kg(kg, pmid_support=True):

#         if pmid_support:
#             res = [(kg['edges'][key]['subject'], kg['edges'][key]['object'], attr['value']) for key in kg['edges'] for attr in kg['edges'][key]['attributes'] if attr['original_attribute_name']=='publication']
#             return pd.DataFrame(res, columns=['subject','object','pmid'])
#         else:
#             res = [(kg['edges'][key]['subject'], kg['edges'][key]['object']) for key in kg['edges']]
#             return pd.DataFrame(res, columns=['subject','object'])


#     def get_molepro_data(self, param_list):

#         # param_list = [(row[0], row[1]) for row in res.to_numpy()]
#         ## start the asyncio program
#         asyncio.run(self._get_data(param_list))

class MoleProData:

    def __init__(self, logger, molepro_aws_link: str = 'https://molepro.s3.amazonaws.com'):
        """
        Initial Method
        """
        ## setup basic information
        self.molepro_aws_link = molepro_aws_link
        self.logger = logger
        
    def _download_data(self, output: str):
        if not os.path.exists(output):
            os.makedirs(output)
        
            ## download data from aws
            self.logger.info("starting to download data from molepro aws")
            subprocess.run(f"wget -O {output}/nodes.tsv {self.molepro_aws_link}/nodes.tsv", shell=True)
            subprocess.run(f"wget -O {output}/edges.tsv {self.molepro_aws_link}/edges.tsv", shell=True)
        else:
            if not os.path.exists(f"{output}/nodes.tsv"):
                subprocess.run(f"wget -O {output}/nodes.tsv {self.molepro_aws_link}/nodes.tsv", shell=True)
            if not os.path.exists(f"{output}/edges.tsv"):
                subprocess.run(f"wget -O {output}/edges.tsv {self.molepro_aws_link}/edges.tsv", shell=True)
    
    def load_data(self, output: str):
        
        self._download_data(output)
        self.logger.info("starting to load data from molepro aws")
        mapping_node_to_type = {}
        self.nodes_dict = {} 
        with open(f"{output}/nodes.tsv", newline='', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter='\t')
            ## remove the header
            next(reader)
            for row in reader:
                mapping_node_to_type[row[0]] = row[1]
                self.nodes_dict[(row[0], row[1])] = 1
        self.edges_dict = {}
        with open(f"{output}/edges.tsv", newline='', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter='\t')
            ## remove the header
            header = next(reader)
            for row in reader:
                if row[header.index('predicate')] not in ["biolink:affects", "biolink:interacts_with"]:
                    continue
                if mapping_node_to_type[row[header.index('object')]] not in ["biolink:Gene", "biolink:Protein"]:
                    continue
                if row[header.index('subject')] not in self.edges_dict:
                    self.edges_dict[row[header.index('subject')]] = set()    
                self.edges_dict[row[header.index('subject')]].add((row[header.index('object')], row[header.index('biolink:publications')]))
        
    def extract_drug_target_pairs_from_kg(self, query_node, query_type, pmid_support=True):

        if pmid_support:
            result = []
            if (query_node, query_type) not in self.nodes_dict:
                # self.logger.warning(f"{query_node} is not in the molepro kg")
                return pd.DataFrame([], columns=['subject','object','pmids'])
            else:
                if query_node not in self.edges_dict:
                    return pd.DataFrame([], columns=['subject','object','pmids'])
                for key in self.edges_dict[query_node]:
                    if key[1] == '':
                        continue
                    else:
                        result += [(query_node, key[0], key[1].split('|'))]
                return pd.DataFrame(result, columns=['subject','object','pmids'])
        else:
            result = []
            if (query_node, query_type) not in self.nodes_dict:
                # self.logger.warning(f"{query_node} is not in the molepro kg")
                return pd.DataFrame([], columns=['subject','object','pmids'])
            else:
                if query_node not in self.edges_dict:
                    return pd.DataFrame([], columns=['subject','object','pmids'])
                for key in self.edges_dict[query_node]:
                    result += [(query_node, key[0], )]
                return pd.DataFrame(result, columns=['subject','object', 'pmids'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", type=str, help="The path of logfile folder", default=os.path.join(ROOTPath, "log_folder"))
    parser.add_argument("--log_name", type=str, help="log file name", default="integrate_drugbank_and_molepro_data.log")
    parser.add_argument("--db_config_path", type=str, help="path to database config file", default="../config_dbs.json")
    parser.add_argument("--secret_config_path", type=str, help="path to secret config file", default="../config_secrets.json")
    parser.add_argument("--batchsize", type=int, help="Batch Size", default=50000)
    parser.add_argument("--process", type=int, help="Use number of processes to run the program", default=50)
    parser.add_argument("--drugbank_export_paths", type=str, help='path to a file containing drugbank-based paths', default=os.path.join(ROOTPath, "data", "expert_path_files", "p_expert_paths.txt"))
    # parser.add_argument('--molepro_api_link', type=str, help='API link of Molecular Data Provider', default='https://molepro-trapi.transltr.io/molepro/trapi/v1.3')
    parser.add_argument('--molepro_aws_link', type=str, help='API link of Molecular Data Provider', default='https://molepro.s3.amazonaws.com')
    parser.add_argument("--output_folder", type=str, help="The path of output folder", default=os.path.join(ROOTPath, "data"))
    args = parser.parse_args()


    logger = utils.get_logger(os.path.join(args.log_dir,args.log_name))
    logger.info(args)


    if not os.path.exists(os.path.join(args.output_folder, 'expert_path_files', 'all_drugs.txt')):

        ## Load the contents of two config files (e.g., config_dbs.json and config_secrets.json)
        with open(args.db_config_path, 'rb') as file_in:
            config_dbs = json.load(file_in)
        with open(args.secret_config_path, 'rb') as file_in:
            config_secrets = json.load(file_in)

        ## Connect to neo4j database
        neo4j_instance = config_dbs["neo4j"]["KG2c"]
        neo4j_bolt = f"bolt://{neo4j_instance}:7687"
        neo4j_username = config_secrets["neo4j"]["KG2c"]["username"]
        neo4j_password = config_secrets["neo4j"]["KG2c"]["password"]
        conn = utils.Neo4jConnection(uri=neo4j_bolt, user=neo4j_username, pwd=neo4j_password)

        ## extract all possible drug entities from neo4j database
        res = conn.query(f"match (n) where n.category='biolink:SmallMolecule' or n.category='biolink:ChemicalEntity' or n.category='biolink:Drug' return distinct n.id, n.category, n.equivalent_curies")    
        res.columns = ['id','category','equivalent_curies']
        res.to_csv(os.path.join(args.output_folder, 'expert_path_files', 'all_drugs.txt'), sep='\t', index=None)
    else:
        res = pd.read_csv(os.path.join(args.output_folder, 'expert_path_files', 'all_drugs.txt'), sep='\t', header=0)
        res = res.apply(lambda row: [row[0], row[1], eval(row[2])], axis=1, result_type='expand')
        res.columns = ['id','category','equivalent_curies']

    ## read drugbank processed data
    p_expert_paths = pd.read_csv(args.drugbank_export_paths, sep='\t', header=None)
    p_expert_paths = p_expert_paths.loc[~p_expert_paths[2].isna(),:]
    p_expert_paths.columns = ['drugbankid', 'subject', 'object']

    ## query molepro (Molecular Data Provider) API
    if not os.path.exists(os.path.join(args.output_folder, 'expert_path_files', 'molepro_df_backup.txt')):
        
        ## Get all equivalent nodes
        all_equivalent_nodes = []
        for row in tqdm(res.to_numpy(), desc='Get all equivalent nodes'):
            preferred_curie = row[0]
            normalizer = synonymizer.get_normalizer_results(preferred_curie)[preferred_curie]
            if normalizer:
                all_equivalent_nodes += [(item['identifier'], item['category']) for item in normalizer['nodes']]
            else:
                all_equivalent_nodes += [(row[0],row[1])] 

        # # set up the batches
        # # pair_list = [[row[0], row[1], args.molepro_api_link] for row in res.to_numpy()]
        # pair_list = [[row[0], row[1]] for row in all_equivalent_nodes]
        # batch =list(range(0,len(pair_list), args.batchsize))
        # batch.append(len(pair_list))
        # logger.info(f'total batch: {len(batch)-1}')

        # molepro_df = pd.DataFrame(columns=['subject','object','pmid'])
        # molepro_obj = MoleProData(logger)
        # ## run in acyclic mode
        # for i in range(len(batch)):
        #     if((i+1)<len(batch)):
        #         print(f'batch {i+1} out of {len(batch)-1}', flush=True)
        #         start = batch[i]
        #         end = batch[i+1]
        #         # use httpx instead of requests and asyncio instead of multiprocessing to call API
        #         molepro_obj.get_molepro_data(pair_list[start:end])
        #         molepro_df = pd.concat([molepro_df,molepro_obj.results]).reset_index(drop=True)

        #         # save intermediate results
        #         molepro_df.to_csv(os.path.join(args.output_folder, 'expert_path_files', 'molepro_df_backup.txt'), sep='\t', index=None)
        #         logger.info(f'Sleep 5 seconds before next request')
        #         time.sleep(5)

        pair_list = [[row[0], row[1]] for row in all_equivalent_nodes]
        molepro_obj = MoleProData(logger, args.molepro_aws_link)
        molepro_obj.load_data(os.path.join(args.output_folder, 'expert_path_files', 'temp_molepro'))
        molepro_df = pd.DataFrame(columns=['subject','object','pmids'])
        for row in tqdm(pair_list, desc='Get molepro data'):
            molepro_df = pd.concat([molepro_df,molepro_obj.extract_drug_target_pairs_from_kg(row[0], row[1], pmid_support=True)]).reset_index(drop=True)

        molepro_df.to_csv(os.path.join(args.output_folder, 'expert_path_files', 'molepro_df_backup.txt'), sep='\t', index=None)
    else:
        molepro_df = pd.read_csv(os.path.join(args.output_folder, 'expert_path_files', 'molepro_df_backup.txt'), sep='\t', header=0)
        molepro_df = molepro_df.apply(lambda row: [row[0], row[1], eval(row[2])], axis=1, result_type='expand')
        molepro_df.columns = ['subject','object','pmids']


    temp_dict = dict()
    for index in trange(len(molepro_df)):
        source, target, pmids = molepro_df.loc[index,'subject'], molepro_df.loc[index,'object'], molepro_df.loc[index,'pmids']
        normalizer = synonymizer.get_canonical_curies(source)[source]
        if normalizer:
            source = normalizer['preferred_curie']
        else:
            continue
        pmids = [pmid for pmid in pmids if pmid is not None and isinstance(pmid, str) and pmid.startswith('PMID:')]
        if (source, target) not in temp_dict:
            temp_dict[(source, target)] = pmids
        else:
            temp_dict[(source, target)] += list(set(temp_dict[(source, target)] + pmids))
    molepro_df = pd.DataFrame([(key[0], key[1], value) for key, value in temp_dict.items()])
    molepro_df.columns = ['subject','object','pmids']

    combined_table = molepro_df.merge(p_expert_paths,how='outer',on=['subject','object'])
    combined_table.loc[(~combined_table.loc[:,'pmids'].isna()) & (~combined_table.loc[:,'drugbankid'].isna()),'supported_sources'] = 'drugbank&molepro'
    combined_table.loc[(combined_table.loc[:,'pmids'].isna()) & (~combined_table.loc[:,'drugbankid'].isna()),'supported_sources'] = 'drugbank'
    combined_table.loc[(~combined_table.loc[:,'pmids'].isna()) & (combined_table.loc[:,'drugbankid'].isna()),'supported_sources'] = 'molepro'

    ## output the results
    combined_table.to_csv(os.path.join(args.output_folder, 'expert_path_files', 'p_expert_paths_combined.txt'), sep='\t', index=None)
