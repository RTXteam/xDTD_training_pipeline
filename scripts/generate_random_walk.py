"""Generate random-walk co-occurrence pairs (walks.txt) for GraphSAGE.

Uses the `walker` library to perform fast random walks batch-by-batch,
then writes (source, co-occurring node) pairs to a TSV file.
"""

import argparse
import json
import os
import sys

import numpy as np
import walker
from networkx.readwrite import json_graph

## Import Personal Packages
pathlist = os.getcwd().split(os.path.sep)
ROOTindex = pathlist.index("xDTD_training_pipeline")
ROOTPath = os.path.sep.join([*pathlist[:(ROOTindex + 1)]])
sys.path.append(os.path.join(ROOTPath, 'scripts'))
import utils


def convert_array_to_pair_list(walk_result):
    """Convert walk arrays into (start_node, visited_node) pairs."""
    pairs = []
    for row in walk_result:
        pairs.extend((row[0], item) for item in row[1:])
    return pairs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", type=str, help="The path of logfile folder", default=os.path.join(ROOTPath, "log_folder"))
    parser.add_argument("--log_name", type=str, help="log file name", default="step14_generate_random_walk.log")
    parser.add_argument("--Gjson", type=str, help="The path of G.json file")
    parser.add_argument("--walk_length", type=int, help="Random walk length", default=30)
    parser.add_argument("--number_of_walks", type=int, help="Number of random walks per node", default=10)
    parser.add_argument("--batch_size", type=int, help="Size of batch for each run", default=200000)
    parser.add_argument("--output_folder", type=str, help="The path of output folder", default=os.path.join(ROOTPath, "data", "graphsage_input"))
    args = parser.parse_args()

    logger = utils.get_logger(os.path.join(args.log_dir, args.log_name))
    logger.info(args)

    os.makedirs(args.output_folder, exist_ok=True)

    with open(args.Gjson, 'r') as f:
        G_data = json.load(f)

    edge_key = "links" if "links" in G_data else "edges"
    G = json_graph.node_link_graph(G_data, edges=edge_key)
    # pull out the training nodes and generate the training subgraph
    G_nodes = [n for n in G.nodes() if not G.nodes[n]["val"] and not G.nodes[n]["test"]]
    G = G.subgraph(G_nodes)
    del G_data ## delete variable to release ram

    n_nodes = len(G_nodes)
    logger.info(f'Total training data: {n_nodes}')
    logger.info(f'The number of nodes in training graph: {len(G.nodes)}')

    out_path = os.path.join(args.output_folder, 'data-walks.txt')
    n_batches = (n_nodes + args.batch_size - 1) // args.batch_size
    logger.info(f'Total batches: {n_batches}')

    with open(out_path, 'w') as fp:
        for batch_idx in range(n_batches):
            start = batch_idx * args.batch_size
            end = min(start + args.batch_size, n_nodes)
            logger.info(f'Batch {batch_idx + 1}/{n_batches} (nodes {start}-{end})')

            indexes = walker.random_walks(
                G,
                n_walks=args.number_of_walks,
                walk_len=args.walk_length,
                start_nodes=range(start, end),
            )
            walk_result = np.array(G_nodes)[indexes]
            pairs = convert_array_to_pair_list(walk_result)

            if batch_idx > 0:
                fp.write("\n")
            fp.write("\n".join(f"{p[0]}\t{p[1]}" for p in pairs))

    logger.info(f'Walks written to {out_path}')
