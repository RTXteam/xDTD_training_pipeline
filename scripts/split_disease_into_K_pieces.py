
import os, sys
import pandas as pd
import pickle
import argparse
import numpy as np

## Import Personal Packages
pathlist = os.getcwd().split(os.path.sep)
ROOTindex = pathlist.index("xDTD_training_pipeline")
ROOTPath = os.path.sep.join([*pathlist[:(ROOTindex + 1)]])
sys.path.append(os.path.join(ROOTPath, 'scripts'))
import utils

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Split diseases into K pieces')
    parser.add_argument("--log_dir", type=str, help="The path of logfile folder", default=os.path.join(ROOTPath, "log_folder"))
    parser.add_argument("--log_name", type=str, help="log file name", default="split_disease_into_K_pieces.log")
    parser.add_argument('--K', type=int, default=5, help='K for data seperation')
    parser.add_argument('--data_dir', type=str, help='Full path of data folder', default=os.path.join(ROOTPath, "data"))
    parser.add_argument('--out_dir', type=str, required=True, help='Output Directory')
    args = parser.parse_args()

    logger = utils.get_logger(os.path.join(args.log_dir,args.log_name))
    logger.info(args)

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
    disease_meta_info = all_graph_nodes_info.loc[all_graph_nodes_info['id'].isin(disease_curie_ids),['id','name','all_names']].reset_index(drop=True)

    ## split data into K pieces
    logger.info(f"Split disease into {args.K} pieces")
    disease_meta_info_K = [disease_meta_info.loc[indices,:] for indices in np.array_split(np.array(disease_meta_info.sample(frac=1).index), args.K)]
    for index, df in enumerate(disease_meta_info_K):
        df.to_csv(os.path.join(args.out_dir, f'disease_set{index+1}.txt'), sep='\t', index=None)

