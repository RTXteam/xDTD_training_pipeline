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

## Import Personal Packages
pathlist = os.getcwd().split(os.path.sep)
ROOTindex = pathlist.index("xDTD_training_pipeline")
ROOTPath = os.path.sep.join([*pathlist[:(ROOTindex + 1)]])
sys.path.append(os.path.join(ROOTPath, 'scripts'))
import utils

class MoleProData:

    def __init__(self, logger, molepro_api_link: str = 'https://molepro-trapi.transltr.io/molepro/trapi/v1.3'):
        """
        Initial Method
        """
        ## setup basic information
        self.molepro_api_link = molepro_api_link
        self.logger = logger
        # self.client = httpx.Client()

    async def _call_async_melepro(self, client, curie_id: str, category: str):
            try:
                request_body = self._generate_query_graph(curie_id, category)
                resp = await client.post(f'{self.molepro_api_link}/query', json = request_body, headers={'accept': 'application/json'})
            except Exception:
                # print(f"ERROR: {curie_id}", flush=True)
                self.logger.error(f"{curie_id}")
                return pd.DataFrame([], columns=['subject','object','pmid'])

            if not resp.status_code == 200:
                # print(f"WARNING: {curie_id} fails to call molepro api with status code {resp.status_code}")
                self.logger.warning(f"{curie_id} fails to call molepro api with status code {resp.status_code}")
                return pd.DataFrame([], columns=['subject','object','pmid'])

            resp_res = resp.json()
            temp_pairs = self._extract_drug_target_pairs_from_kg(resp_res['message']['knowledge_graph'])
    
            return temp_pairs

    async def _get_data(self, param_list: List):
        async with httpx.AsyncClient(timeout=60) as client:
            tasks = [asyncio.create_task(self._call_async_melepro(client, curie_id, category)) for curie_id, category in param_list]
            self.logger.info("starting to extract data from molepro api")
            temp_results = await asyncio.gather(*tasks)
            self.results = pd.concat(temp_results).reset_index(drop=True)
            self.logger.info("Extracted data from molepro api done")

    @staticmethod
    def _generate_query_graph(curie_id, category):
        if type(curie_id) is str:
            query_id = [curie_id]
        else:
            query_id = curie_id

        query_graph = {
            "message": {
                "query_graph": {
                "edges": {
                    "e00": {
                    "subject": "n00",
                    "predicates": [
                        "biolink:affects",
                        "biolink:interacts_with"
                    ],
                    "object": "n01"
                    }
                },
                "nodes": {
                    "n00": {
                    "ids": query_id,
                    "categories": [
                        category
                    ]
                    },
                    "n01": {
                    "categories": [
                        "biolink:Gene",
                        "biolink:Protein"
                    ]
                    }
                }
                }
            }
        }

        return query_graph

    @staticmethod
    def _extract_drug_target_pairs_from_kg(kg, pmid_support=True):

        if pmid_support:
            res = [(kg['edges'][key]['subject'], kg['edges'][key]['object'], attr['value']) for key in kg['edges'] for attr in kg['edges'][key]['attributes'] if attr['original_attribute_name']=='publication']
            return pd.DataFrame(res, columns=['subject','object','pmid'])
        else:
            res = [(kg['edges'][key]['subject'], kg['edges'][key]['object']) for key in kg['edges']]
            return pd.DataFrame(res, columns=['subject','object'])


    def get_molepro_data(self, param_list):

        # param_list = [(row[0], row[1]) for row in res.to_numpy()]
        ## start the asyncio program
        asyncio.run(self._get_data(param_list))


# def get_melepro_data(params: tuple):

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

#     def _extract_drug_target_pairs_from_kg(kg, pmid_support=True):

#         if pmid_support:
#             res = [(kg['edges'][key]['subject'], kg['edges'][key]['object'], attr['value']) for key in kg['edges'] for attr in kg['edges'][key]['attributes'] if attr['original_attribute_name']=='publication']
#             return pd.DataFrame(res, columns=['subject','object','pmid'])
#         else:
#             res = [(kg['edges'][key]['subject'], kg['edges'][key]['object']) for key in kg['edges']]
#             return pd.DataFrame(res, columns=['subject','object'])


#     curie_id, category, molepro_api_link, session = params
#     try:
#         request_body = _generate_query_graph(curie_id, category)
#         resp = session.post(f'{molepro_api_link}/query', json = request_body, headers={'accept': 'application/json'}, timeout=120) 
#     except Exception:
#         print(f"ERROR: {curie_id}", flush=True)
#         return pd.DataFrame([], columns=['subject','object','pmid'])

#     if not resp.status_code == 200:
#         print(f"WARNING: {curie_id} fails to call molepro api with status code {resp.status_code}")
#         return pd.DataFrame([], columns=['subject','object','pmid'])

#     resp_res = resp.json()
#     temp_pairs = _extract_drug_target_pairs_from_kg(resp_res['message']['knowledge_graph'])

#     return temp_pairs



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", type=str, help="The path of logfile folder", default=os.path.join(ROOTPath, "log_folder"))
    parser.add_argument("--log_name", type=str, help="log file name", default="integrate_drugbank_and_molepro_data.log")
    parser.add_argument("--db_config_path", type=str, help="path to database config file", default="../config_dbs.json")
    parser.add_argument("--secret_config_path", type=str, help="path to secret config file", default="../config_secrets.json")
    parser.add_argument("--batchsize", type=int, help="Batch Size", default=50000)
    parser.add_argument("--process", type=int, help="Use number of processes to run the program", default=50)
    parser.add_argument("--drugbank_export_paths", type=str, help='path to a file containing drugbank-based paths', default=os.path.join(ROOTPath, "data", "expert_path_files", "p_expert_paths.txt"))
    parser.add_argument('--molepro_api_link', type=str, help='API link of Molecular Data Provider', default='https://molepro-trapi.transltr.io/molepro/trapi/v1.3')
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
        res = conn.query(f"match (n) where n.category='biolink:SmallMolecule' or n.category='biolink:Drug' return distinct n.id, n.category, n.equivalent_curies")    
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
        # api_res = requests.get(f'{args.molepro_api_link}/meta_knowledge_graph')
        # if api_res.status_code == 200:
        #     molepro_meta_kg = api_res.json()

        # set up the batches
        # pair_list = [[row[0], row[1], args.molepro_api_link] for row in res.to_numpy()]
        pair_list = [[row[0], row[1]] for row in res.to_numpy()]
        batch =list(range(0,len(pair_list), args.batchsize))
        batch.append(len(pair_list))
        logger.info(f'total batch: {len(batch)-1}')

        molepro_df = pd.DataFrame(columns=['subject','object','pmid'])
        molepro_obj = MoleProData(logger)
        ## run in acyclic mode
        for i in range(len(batch)):
            if((i+1)<len(batch)):
                print(f'batch {i+1} out of {len(batch)-1}', flush=True)
                start = batch[i]
                end = batch[i+1]

                # create a new session object per batch to avoid exceeding the maximum requests that URL allows
                # with requests.Session() as session:
                #     if args.process == -1:
                #         with Pool() as executor:
                #             temp_pair_list = [[*param] + [session] for param in pair_list[start:end]]
                #             out_res = executor.map(get_melepro_data, temp_pair_list)
                #     else:
                #         with Pool(processes=args.process) as executor:
                #             temp_pair_list = [[*param] + [session] for param in pair_list[start:end]]
                #             out_res = executor.map(get_melepro_data, temp_pair_list)
                # temp_molepro_df = pd.concat(out_res).reset_index(drop=True)
                # molepro_df = pd.concat([molepro_df,temp_molepro_df]).reset_index(drop=True)

                # use httpx instead of requests and asyncio instead of multiprocessing to call API
                molepro_obj.get_molepro_data(pair_list[start:end])
                molepro_df = pd.concat([molepro_df,molepro_obj.results]).reset_index(drop=True)

                # save intermediate results
                molepro_df.to_csv(os.path.join(args.output_folder, 'expert_path_files', 'molepro_df_backup.txt'), sep='\t', index=None)
                logger.info(f'Sleep 5 seconds before next request')
                time.sleep(5)

        molepro_df.to_csv(os.path.join(args.output_folder, 'expert_path_files', 'molepro_df_backup.txt'), sep='\t', index=None)
    else:
        molepro_df = pd.read_csv(os.path.join(args.output_folder, 'expert_path_files', 'molepro_df_backup.txt'), sep='\t', header=0)

    temp_dict = dict()
    for index in range(len(molepro_df)):
        source, target, pmid = molepro_df.loc[index,'subject'], molepro_df.loc[index,'object'], molepro_df.loc[index,'pmid']
        if (source, target) not in temp_dict:
            temp_dict[(source, target)] = [pmid]
        else:
            temp_dict[(source, target)].append(pmid)
    molepro_df = pd.DataFrame([(key[0], key[1], value) for key, value in temp_dict.items()])
    molepro_df.columns = ['subject','object','pmid']

    combined_table = molepro_df.merge(p_expert_paths,how='outer',on=['subject','object'])
    combined_table.loc[(~combined_table.loc[:,'pmid'].isna()) & (~combined_table.loc[:,'drugbankid'].isna()),'supported_sources'] = 'drugbank&molepro'
    combined_table.loc[(combined_table.loc[:,'pmid'].isna()) & (~combined_table.loc[:,'drugbankid'].isna()),'supported_sources'] = 'drugbank'
    combined_table.loc[(~combined_table.loc[:,'pmid'].isna()) & (combined_table.loc[:,'drugbankid'].isna()),'supported_sources'] = 'molepro'

    ## output the results
    combined_table.to_csv(os.path.join(args.output_folder, 'expert_path_files', 'p_expert_paths_combined.txt'), sep='\t', index=None)
