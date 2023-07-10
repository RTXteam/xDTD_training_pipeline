## This script is used to generate random walk file (eg. walks.txt, please see https://github.com/williamleif/GraphSAGE
# for more details) via batch by batch for running Graphsage
## Import Standard Packages
from __future__ import print_function

import json
import numpy as np
import pandas as pd
import random
import os
import sys
import argparse
from networkx.readwrite import json_graph
import multiprocessing
import walker
from tqdm import tqdm, trange
# from itertools import chain

## Import Personal Packages
pathlist = os.getcwd().split(os.path.sep)
ROOTindex = pathlist.index("xDTD_training_pipeline")
ROOTPath = os.path.sep.join([*pathlist[:(ROOTindex + 1)]])
sys.path.append(os.path.join(ROOTPath, 'scripts'))
import utils

# ## setting functions compatible with parallel running
# def run_random_walks(this):

#     pairs = []
#     node, num_walks, walk_len = this

#     if G.degree(node) == 0:
#         pairs = pairs
#     else:
#         for _ in range(num_walks):
#             curr_node = node
#             for _ in range(walk_len):
#                 neighbors = [n for n in G.neighbors(curr_node)]
#                 if len(neighbors) == 0:
#                     ## no neighbor, stop BFS searching
#                     break
#                 else:
#                     next_node = random.choice(neighbors)
#                 # self co-occurrences are useless
#                 if curr_node != node:
#                     pairs.append((node, curr_node))
#                 curr_node = next_node
#     return pd.DataFrame(pairs)

def convert_array_to_pair_list(walk_result):
    pairs = []
    for row in walk_result:
        pairs.extend((row[0], item) for item in row[1:])
    return pairs

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", type=str, help="The path of logfile folder", default=os.path.join(ROOTPath, "log_folder"))
    parser.add_argument("--log_name", type=str, help="log file name", default="generate_random_walk.log")
    parser.add_argument("--Gjson", type=str, help="The path of G.json file")
    parser.add_argument("--walk_length", type=int, help="Random walk length", default=30)
    parser.add_argument("--number_of_walks", type=int, help="Number of random walks per node", default=10)
    parser.add_argument("--batch_size", type=int, help="Size of batch for each run", default=200000)
    # parser.add_argument("--process", type=int, help="Number of processes to be used", default=-1)
    parser.add_argument("--output_folder", type=str, help="The path of output folder", default=os.path.join(ROOTPath, "data", "graphsage_input"))

    args = parser.parse_args()

    logger = utils.get_logger(os.path.join(args.log_dir,args.log_name))
    logger.info(args)

    #create output directory
    if not os.path.isdir(args.output_folder):
        os.makedirs(args.output_folder)

    #read the graph file
    with open(args.Gjson,'r') as input_file:
	    G_data = json.load(input_file)

    # transform to networkx graph format
    G = json_graph.node_link_graph(G_data)
    # pull out the training nodes and generate the training subgraph
    G_nodes = [n for n in G.nodes() if not G.nodes[n]["val"] and not G.nodes[n]["test"]]
    G = G.subgraph(G_nodes)
    del G_data ## delete variable to release ram

    # set up the batches
    batch =list(range(0,len(G_nodes),args.batch_size))
    batch.append(len(G_nodes))

    logger.info(f'Total training data: {len(G_nodes)}')
    logger.info(f'The number of nodes in training graph: {len(G.nodes)}')

    logger.info(f'total batch: {len(batch)-1}')

    for i in range(len(batch)):
        if((i+1)<len(batch)):
            logger.info(f'Here is batch{i+1}')
            start = batch[i]
            end = batch[i+1]
            indexes = walker.random_walks(G, n_walks=10, walk_len=30, start_nodes=range(start, end))
            walk_result = np.array(G_nodes)[indexes]
            out_res = convert_array_to_pair_list(walk_result)
            
            with open(os.path.join(args.output_folder, 'data-walks.txt'), "a") as fp:
                if i==0:
                    fp.write("\n".join([str(p[0]) + "\t" + str(p[1]) for p in out_res]))
                else:
                    fp.write("\n")
                    fp.write("\n".join([str(p[0]) + "\t" + str(p[1]) for p in out_res]))

    # ## run each batch in parallel
    # for i in range(len(batch)):
    #     if((i+1)<len(batch)):
    #         logger.info(f'Here is batch{i+1}')
    #         start = batch[i]
    #         end = batch[i+1]
    #         if args.process == -1:
    #             with multiprocessing.Pool() as executor:
    #                 out_iters = [(node, args.number_of_walks, args.walk_length) for node in G_nodes[start:end]]
    #                 out_res = pd.concat(executor.map(run_random_walks, out_iters))
    #         else:
    #             with multiprocessing.Pool(processes=args.process) as executor:
    #                 out_iters = [(node, args.number_of_walks, args.walk_length) for node in G_nodes[start:end]]
    #                 out_res = pd.concat(executor.map(run_random_walks, out_iters))
            
    #         with open(os.path.join(args.output_folder, 'data-walks.txt'), "a") as fp:
    #             if i==0:
    #                 fp.write("\n".join([str(p[0]) + "\t" + str(p[1]) for p in out_res]))
    #             else:
    #                 fp.write("\n")
    #                 fp.write("\n".join([str(p[0]) + "\t" + str(p[1]) for p in out_res]))

