"""Transform GraphSAGE output into a CURIE-keyed embedding dictionary (pkl)."""

import argparse
import os
import pickle
import sys

import numpy as np
import polars as pl

## Import Personal Packages
pathlist = os.getcwd().split(os.path.sep)
ROOTindex = pathlist.index("xDTD_training_pipeline")
ROOTPath = os.path.sep.join([*pathlist[:(ROOTindex + 1)]])
sys.path.append(os.path.join(ROOTPath, 'scripts'))
import utils


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", type=str, help="The path of logfile folder", default=os.path.join(ROOTPath, "log_folder"))
    parser.add_argument("--log_name", type=str, help="log file name", default="step16_transform_format.log")
    parser.add_argument("--data_dir", type=str, help="The path of data folder", default=os.path.join(ROOTPath, "data"))
    parser.add_argument("--input", type=str, help="The full path of graphsage output folder")
    args = parser.parse_args()

    logger = utils.get_logger(os.path.join(args.log_dir, args.log_name))
    logger.info(args)

    vectors = np.load(os.path.join(args.input, 'val.npy'))
    ids = pl.read_csv(os.path.join(args.input, 'val.txt'), has_header=False, new_columns=['id'])
    sort_order = ids['id'].arg_sort()
    sorted_vectors = vectors[sort_order.to_numpy()]

    curie_map = pl.read_csv(os.path.join(args.data_dir, 'graphsage_input', 'id_map.txt'), separator='\t')
    emb_dict = {curie: sorted_vectors[i] for i, curie in enumerate(curie_map['curie'].to_list())}

    out_dir = os.path.join(args.data_dir, 'graphsage_output')
    os.makedirs(out_dir, exist_ok=True)

    out_path = os.path.join(out_dir, 'unsuprvised_graphsage_entity_embeddings.pkl')
    with open(out_path, 'wb') as f:
        pickle.dump(emb_dict, f)

    logger.info(f'Saved {len(emb_dict)} embeddings to {out_path}')
