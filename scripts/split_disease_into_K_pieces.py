
import os, sys
import pandas as pd
import pickle
import argparse
import numpy as np
import json
from tqdm import tqdm

## Import Personal Packages
pathlist = os.getcwd().split(os.path.sep)
ROOTindex = pathlist.index("xDTD_training_pipeline")
ROOTPath = os.path.sep.join([*pathlist[:(ROOTindex + 1)]])
sys.path.append(os.path.join(ROOTPath, 'scripts'))
import utils
from drugconflator import DrugConflator

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Split diseases into K pieces')
    parser.add_argument("--log_dir", type=str, help="The path of logfile folder", default=os.path.join(ROOTPath, "log_folder"))
    parser.add_argument("--log_name", type=str, help="log file name", default="split_disease_into_K_pieces.log")
    parser.add_argument('--K', type=int, default=5, help='K for data seperation')
    parser.add_argument('--data_dir', type=str, help='Full path of data folder', default=os.path.join(ROOTPath, "data"))
    parser.add_argument("--db_config_path", type=str, help="path to database config file", default="../config_dbs.json")
    args = parser.parse_args()

    logger = utils.get_logger(os.path.join(args.log_dir,args.log_name))
    logger.info(args)

    ## Load the contents of two config files (e.g., config_dbs.json and config_secrets.json)
    with open(args.db_config_path, 'rb') as file_in:
        config_dbs = json.load(file_in)

    ## set up output folder
    logger.info(f"Output Directory: {args.out_dir}")
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    ## load datasets
    logger.info(f"Load datasets from {args.data_dir}")
    entity2id, id2entity = utils.load_index(os.path.join(args.data_dir, 'entity2freq.txt'))
    relation2id, id2relation = utils.load_index(os.path.join(args.data_dir, 'relation2freq.txt'))
    type2id, id2type = utils.load_index(os.path.join(args.data_dir, 'type2freq.txt'))
    with open(os.path.join(args.data_dir, 'entity2typeid.pkl'), 'rb') as infile:
        entity2typeid = pickle.load(infile)
    all_graph_nodes_info = pd.read_csv(os.path.join(args.data_dir, 'all_graph_nodes_info.txt'), sep='\t', header=0)

    ## get disease meta info
    disease_type = ['biolink:Disease', 'biolink:PhenotypicFeature', 'biolink:BehavioralFeature', 'biolink:DiseaseOrPhenotypicFeature']
    disease_type_ids = [type2id[x] for x in disease_type]
    disease_curie_ids = [id2entity[index] for index, typeid in enumerate(entity2typeid) if typeid in disease_type_ids]
    filtered_disease_nodes = [x for x in disease_curie_ids if 'MONDO' in x]
    disease_meta_info = all_graph_nodes_info.loc[all_graph_nodes_info['id'].isin(filtered_disease_nodes),['id','name','all_names']].reset_index(drop=True)

    ## split data into K pieces
    logger.info(f"Split disease into {args.K} pieces")
    disease_meta_info_K = [disease_meta_info.loc[indices,:] for indices in np.array_split(np.array(disease_meta_info.sample(frac=1).index), args.K)]
    for index, df in enumerate(disease_meta_info_K):
        df.to_csv(os.path.join(args.out_dir, f'disease_set{index+1}.txt'), sep='\t', index=None)

    ## get drug info
    drug_nodes = list(set(all_graph_nodes_info.query(f"category in {['biolink:Drug','biolink:SmallMolecule','biolink:ChemicalEntity']}")['id']))

    ###### Use DrugConflator to filter drugs that has RxCUI ids
    node_synonymizer_path = config_dbs["database_downloads"]["node_synonymizer"]
    node_synonymizer_name = node_synonymizer_path.split('/')[-1]
    ## initialize DrugConflator
    dc = DrugConflator(node_synonymizer_path=os.path.join(args.data_dir, node_synonymizer_name), mychem_data_path=os.path.join(args.data_dir, 'mychem_rxcui.json'))

    if not os.path.exists(os.path.join(args.data_dir, 'filtered_drugs')):
        os.makedirs(os.path.join(args.data_dir, 'filtered_drugs'))

    ## filter drug nodes
    existing_drug_curies = list(set([curie for x in os.listdir(os.path.join(args.data_dir, 'filtered_drugs')) for curie in set(pd.read_csv(f"{args.data_dir}/filtered_drugs/{x}", sep='\t', header=0)['curie'])]))
    drug_list = list(set(drug_nodes).difference(set(existing_drug_curies)))
    offset = len(existing_drug_curies)

    filtered_drug_nodes = []
    for index, drug_curie in enumerate(tqdm(drug_list)):
        print(f"Processing {drug_curie}", flush=True)
        index += offset
        if index != 0 + offset and index % 1000 == 0:
            if len(filtered_drug_nodes) != 0:
                out_df = pd.DataFrame(filtered_drug_nodes, columns = ['curie', 'has_rxcui'])
                filtered_drug_nodes = []
                out_df.to_csv(os.path.join(args.data_dir, 'filtered_drugs', f"batch_{int(index/1000)}.tsv"), sep = '\t', index=False)
        if len(dc.get_rxcui_results(drug_curie, use_curie_name=False)) > 0:
            filtered_drug_nodes += [(drug_curie, True)]
        else:
            filtered_drug_nodes += [(drug_curie, False)]

    ## group all drug results together and save it to a pkl file
    existing_drug_curies = pd.concat([pd.read_csv(f"{args.data_dir}/filtered_drugs/{x}", sep='\t', header=0) for x in os.listdir(os.path.join(args.data_dir, 'filtered_drugs'))]).reset_index(drop=True)
    existing_drug_curies = list(existing_drug_curies.query("has_rxcui == True")['curie'])
    with open(os.path.join(args.data_dir, 'filtered_drug_nodes_for_precomputation.pkl'), 'wb') as f:
        pickle.dump(existing_drug_curies, f)

