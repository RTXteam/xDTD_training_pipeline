## Import Standard Packages
import sys, os
import pandas as pd
import argparse
import math
import sqlite3
import json
from tqdm import tqdm, trange
from multiprocessing import Pool
import networkx as nx

## Import Personal Packages
pathlist = os.getcwd().split(os.path.sep)
ROOTindex = pathlist.index("xDTD_training_pipeline")
ROOTPath = os.path.sep.join([*pathlist[:(ROOTindex + 1)]])
sys.path.append(os.path.join(ROOTPath, 'scripts'))
import utils
from biolink_helper import BiolinkHelper


def calculate_ngd(concept_pubmed_ids):

    if concept_pubmed_ids[0] is None or concept_pubmed_ids[1] is None:
        return None
    concept_pubmed_ids = (eval(concept_pubmed_ids[0]),eval(concept_pubmed_ids[1]))

    marginal_counts = list(map(lambda pmid_list: len(set(pmid_list)), concept_pubmed_ids))
    joint_count = len(set(concept_pubmed_ids[0]).intersection(set(concept_pubmed_ids[1])))

    if 0 in marginal_counts or 0. in marginal_counts:
        return None
    elif joint_count == 0 or joint_count == 0.:
        return None
    else:
        try:
            return (max([math.log(count) for count in marginal_counts]) - math.log(joint_count)) / \
                (math.log(utils.NGD_normalizer) - min([math.log(count) for count in marginal_counts]))
        except ValueError:
            return None

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", type=str, help="The path of logfile folder", default=os.path.join(ROOTPath, "log_folder"))
    parser.add_argument("--log_name", type=str, help="log file name", default="filter_kg2_nodes_and_edges.log")
    parser.add_argument("--db_config_path", type=str, help="path to database config file", default="../config_dbs.json")
    parser.add_argument("--graph_nodes", type=str, help="Raw graph node file", default=os.path.join(ROOTPath, "data", "all_graph_nodes_info.txt"))
    parser.add_argument("--graph_edges", type=str, help="Raw graph edge file", default=os.path.join(ROOTPath, "data", "graph_edges.txt"))
    parser.add_argument("--pub_threshold", type=float, help="Threshold used to filter number of supported publications", default=10)
    parser.add_argument("--ngd_threshold", type=float, help="Threshold used to filter NGD score", default=0.6)
    parser.add_argument("--num_core", type=int, help="Number of CPU cores to run", default=64)
    parser.add_argument("--batch_size", type=int, help="Batch Size", default=200000)
    parser.add_argument("--biolink_version", type=str, help="Biolink version", default="3.1.2")
    parser.add_argument("--output_folder", type=str, help="The path of output folder", default=os.path.join(ROOTPath, "data"))
    args = parser.parse_args()

    logger = utils.get_logger(os.path.join(args.log_dir,args.log_name))
    logger.info(args)
    output_path = args.output_folder

    ## Read raw KG2 edge list and node list
    logger.info("Read raw KG2 edge list and node list")
    raw_graph_edges = pd.read_csv(args.graph_edges, sep='\t', header=0)
    raw_graph_nodes = pd.read_csv(args.graph_nodes, sep='\t', header=0)
    all_nodes_from_edges = list(set(list(raw_graph_edges['source']) + list(raw_graph_edges['target'])))
    raw_graph_nodes = raw_graph_nodes.loc[raw_graph_nodes['id'].isin(all_nodes_from_edges),:].reset_index(drop=True)

    ## Filter KG2c nodes and edges based on categories
    logger.info("Filter KG2c nodes and edges based on categories")
    removed_category = ['biolink:Activity', 'biolink:Agent', 'biolink:ClinicalIntervention', 'biolink:Cohort', 'biolink:Device', 'biolink:EnvironmentalFeature', 'biolink:EnvironmentalProcess', 'biolink:Event', 'biolink:GeographicLocation', 'biolink:IndividualOrganism', 'biolink:InformationContentEntity', 'biolink:InformationResource', 'biolink:LifeStage', 'biolink:MaterialSample', 'biolink:NamedThing', 'biolink:Phenomenon', 'biolink:PhysicalEntity', 'biolink:PopulationOfIndividualOrganisms', 'biolink:Procedure', 'biolink:Publication', 'biolink:Treatment']
    filtered_graph_nodes = raw_graph_nodes.loc[~raw_graph_nodes['category'].isin(removed_category),:].reset_index(drop=True)
    filtered_graph_edges = raw_graph_edges.loc[raw_graph_edges['source'].isin(filtered_graph_nodes['id']) & raw_graph_edges['target'].isin(filtered_graph_nodes['id']),:].reset_index(drop=True)

    ## Get PMID Info
    logger.info("Get PMID Info")
    with open(args.db_config_path, 'rb') as file_in:
        config_dbs = json.load(file_in)
        curie_to_pmids_path = config_dbs["database_downloads"]["curie_to_pmids"]
        curie_to_pmids_name = curie_to_pmids_path.split('/')[-1]
    cnx = sqlite3.connect(os.path.join(ROOTPath, "data", curie_to_pmids_name))
    temp_df = pd.read_sql_query("SELECT * FROM curie_to_pmids", cnx)
    curie_to_pmids_dict = dict(zip(temp_df['curie'], temp_df['pmids']))

    ## Split edges into SemMedDB edges and Non-SemMedDB edges
    logger.info("Split edges into SemMedDB edges and Non-SemMedDB edges")
    semmeddb_edges = []
    non_semmeddb_edges = []
    for edge in tqdm(filtered_graph_edges.to_numpy()):
        new_edge = [edge[0], edge[1], edge[2], len(eval(edge[3])) if type(edge[3]) is str else 0, eval(edge[4])]
        if len(new_edge[4])==1 and new_edge[4][0]=='infores:semmeddb':
            semmeddb_edges.append(new_edge)
        else:
            non_semmeddb_edges.append(new_edge)

    ## Calculate NGD score
    logger.info("Calculate NGD score")
    # set up the batches
    params = [(curie_to_pmids_dict.get(edge[0]), curie_to_pmids_dict.get(edge[1])) for edge in semmeddb_edges]
    batch =list(range(0,len(params), args.batch_size))
    batch.append(len(params))
    logger.info(f'Total batch: {len(batch)-1}')

    ## run each batch in parallel
    all_ngd_scores = []
    for i in trange(len(batch)):
        if((i+1)<len(batch)):
            logger.info(f'Calculting batch{i+1}')
            start = batch[i]
            end = batch[i+1]
            with Pool(processes=args.num_core) as executor:
                all_ngd_scores += executor.map(calculate_ngd, params[start:end])
    semmeddb_edges_df = pd.DataFrame(semmeddb_edges)
    semmeddb_edges_df.columns = ['source', 'target', 'predicate', 'num_publications', 'p_knowledge_source']
    semmeddb_edges_df['ngd_score'] = all_ngd_scores

    ## Filter edges based on NGD score and number of supported publications
    logger.info("Filter edges based on NGD score and number of supported publications")
    filtered_semmeddb_edges_df = semmeddb_edges_df.loc[(semmeddb_edges_df['ngd_score'] <= args.ngd_threshold) & (semmeddb_edges_df['num_publications'] >= args.pub_threshold),:].reset_index(drop=True)
    filtered_semmeddb_edges_df.drop(columns=['ngd_score'], inplace=True)

    ## Combine SemMedDB edges and Non-SemMedDB edges
    logger.info("Combine SemMedDB edges and Non-SemMedDB edges")
    non_semmeddb_edges_df = pd.DataFrame(non_semmeddb_edges)
    non_semmeddb_edges_df.columns = ['source', 'target', 'predicate', 'num_publications', 'p_knowledge_source']
    combined_edges_df = pd.concat([non_semmeddb_edges_df, filtered_semmeddb_edges_df], axis=0).reset_index(drop=True)
    
    ## Filter "redundant" edges based on Biolink edge hierarchy
    logger.info("Filter 'redundant' edges based on Biolink edge hierarchy")
    G = nx.DiGraph()
    for row in tqdm(combined_edges_df.to_numpy()):
        subject_id, predicate, object_id = row[0], row[2], row[1]
        if (subject_id, object_id) in G.edges:
            G.edges[(subject_id, object_id)]['predicate'].update([predicate])
        else:
            G.add_edge(subject_id, object_id, predicate=set([predicate]))
    
    biolink_helper = BiolinkHelper(biolink_version=args.biolink_version)
    for edge_id in tqdm(G.edges):
        predicates = G.edges[edge_id]['predicate']
        temp_predicates = set()
        for predicate in predicates:
            ancestors = [ancestor for ancestor in biolink_helper.get_ancestors(predicate, include_mixins=False)]
            ancestors.remove(predicate)
            temp_predicates.update(ancestors)
        predicates = predicates.difference(temp_predicates)
        G.edges[edge_id]['predicate'] = predicates

    ## Convert NetworkX format to Pandas DataFrame
    logger.info("Save filtered nodes and edges")
    edges = []
    for edge_id in tqdm(G.edges):
        subject_id, object_id = edge_id
        predicates = G.edges[edge_id]['predicate']
        for predicate in predicates:
            edges.append([subject_id, object_id, predicate])
    temp_df = pd.DataFrame(edges, columns=['source', 'target', 'predicate'])
    temp_df.merge(combined_edges_df, on=['source', 'target', 'predicate'], how='left')

    ## Save filtered nodes and edges
    logger.info("Save filtered nodes and edges")
    out_edges_df = temp_df.merge(combined_edges_df, on=['source', 'target', 'predicate'], how='left')
    out_edges_df.to_csv(os.path.join(output_path, "filtered_graph_edges.txt"), sep='\t', index=False)
    all_nodes_from_edges = list(set(list(out_edges_df['source']) + list(out_edges_df['target'])))
    out_graph_nodes = filtered_graph_nodes.loc[filtered_graph_nodes['id'].isin(all_nodes_from_edges),:].reset_index(drop=True)
    out_graph_nodes.to_csv(os.path.join(output_path, "filtered_graph_nodes_info.txt"), sep='\t', index=False)
