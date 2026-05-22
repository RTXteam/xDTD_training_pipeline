import argparse
import os
import pickle
import sys

import numpy as np
import polars as pl
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
    parser.add_argument("--log_name", type=str, help="log file name", default="step22_split_disease_into_K_pieces.log")
    parser.add_argument('--K', type=int, default=5, help='K for data seperation')
    parser.add_argument('--data_dir', type=str, help='Full path of data folder', default=os.path.join(ROOTPath, "data"))
    args = parser.parse_args()

    logger = utils.get_logger(os.path.join(args.log_dir, args.log_name))
    logger.info(args)

    ## ── Create output directory ───────────────────────────────────────────────────────
    out_dir = os.path.join(args.data_dir, 'disease_sets')
    logger.info(f"Output Directory: {out_dir}")
    os.makedirs(out_dir, exist_ok=True)

    ## ── Load datasets ───────────────────────────────────────────────────────────────
    logger.info(f"Load datasets from {args.data_dir}")
    entity2id, id2entity = utils.load_index(os.path.join(args.data_dir, 'entity2freq.txt'))
    relation2id, id2relation = utils.load_index(os.path.join(args.data_dir, 'relation2freq.txt'))
    type2id, id2type = utils.load_index(os.path.join(args.data_dir, 'type2freq.txt'))
    with open(os.path.join(args.data_dir, 'entity2typeid.pkl'), 'rb') as infile:
        entity2typeid = pickle.load(infile)
    all_graph_nodes_info = pl.read_csv(os.path.join(args.data_dir, 'all_graph_nodes_info.txt'), separator='\t')

    ## ── Get disease meta info ───────────────────────────────────────────────────────
    disease_type = ['biolink:Disease', 'biolink:PhenotypicFeature']
    disease_type_ids = [type2id[x] for x in disease_type]
    disease_curie_ids = [id2entity[index] for index, typeid in enumerate(entity2typeid) if typeid in disease_type_ids]
    filtered_disease_nodes = [x for x in disease_curie_ids if 'MONDO' in x]
    disease_meta_info = all_graph_nodes_info.filter(pl.col('id').is_in(filtered_disease_nodes)).select(['id', 'name'])

    ## ── Split data into K pieces ─────────────────────────────────────────────────────
    logger.info(f"Split disease into {args.K} pieces")
    shuffled = disease_meta_info.sample(fraction=1.0, shuffle=True)
    chunks = np.array_split(np.arange(shuffled.height), args.K)
    disease_meta_info_K = [shuffled[chunk.tolist()] for chunk in chunks]
    for index, df in enumerate(disease_meta_info_K):
        df.write_csv(os.path.join(out_dir, f'disease_set{index+1}.txt'), separator='\t')

    ## ── Get drug info ───────────────────────────────────────────────────────────────
    drug_type = ['biolink:Drug','biolink:SmallMolecule','biolink:ChemicalEntity']
    drug_type_ids = [type2id[x] for x in drug_type]
    drug_nodes = [id2entity[index] for index, typeid in enumerate(entity2typeid) if typeid in drug_type_ids]

    ## ── Use DrugConflator to filter drugs that has RxCUI ids ───────────────────────
    dc = DrugConflator(mychem_data_path=os.path.join(args.data_dir, 'mychem_rxcui.json'))

    os.makedirs(os.path.join(args.data_dir, 'filtered_drugs'), exist_ok=True)

    ## ── Filter drug nodes by RxCUI ids ──────────────────────────────────────────────
    existing_drug_curies = list(set([curie for x in os.listdir(os.path.join(args.data_dir, 'filtered_drugs')) for curie in set(pl.read_csv(f"{args.data_dir}/filtered_drugs/{x}", separator='\t')['curie'].to_list())]))
    drug_list = list(set(drug_nodes).difference(set(existing_drug_curies)))
    offset = len(existing_drug_curies)

    filtered_drug_nodes = []
    for index, drug_curie in enumerate(tqdm(drug_list)):
        logger.info(f"Processing {drug_curie}")
        index += offset
        if index != 0 + offset and index % 1000 == 0:
            if len(filtered_drug_nodes) != 0:
                out_df = pl.DataFrame(filtered_drug_nodes, schema=['curie', 'has_rxcui'], orient='row')
                filtered_drug_nodes = []
                out_df.write_csv(os.path.join(args.data_dir, 'filtered_drugs', f"batch_{int(index/1000)}.tsv"), separator='\t')
        if len(dc.get_rxcui_results(drug_curie, use_curie_name=False)) > 0:
            filtered_drug_nodes += [(drug_curie, True)]
        else:
            filtered_drug_nodes += [(drug_curie, False)]
    if len(filtered_drug_nodes) != 0:
        out_df = pl.DataFrame(filtered_drug_nodes, schema=['curie', 'has_rxcui'], orient='row')
        out_df.write_csv(os.path.join(args.data_dir, 'filtered_drugs', f"batch_{int(index/1000)+1}.tsv"), separator='\t')

    ## group all drug results together and save it to a pkl file
    existing_drug_curies = pl.concat([pl.read_csv(f"{args.data_dir}/filtered_drugs/{x}", separator='\t') for x in os.listdir(os.path.join(args.data_dir, 'filtered_drugs'))])
    existing_drug_curies = list(existing_drug_curies.filter(pl.col('has_rxcui') == True).select('curie').to_series().to_list())
    with open(os.path.join(args.data_dir, 'filtered_drug_nodes_for_precomputation.pkl'), 'wb') as f:
        pickle.dump(existing_drug_curies, f)

