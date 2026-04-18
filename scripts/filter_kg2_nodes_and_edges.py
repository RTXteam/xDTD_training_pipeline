## Import Standard Packages
import sys, os
import polars as pl
import argparse
import sqlite3
import json
from tqdm import tqdm, trange
from multiprocessing import Pool

## Import Personal Packages
pathlist = os.getcwd().split(os.path.sep)
ROOTindex = pathlist.index("xDTD_training_pipeline")
ROOTPath = os.path.sep.join([*pathlist[:(ROOTindex + 1)]])
sys.path.append(os.path.join(ROOTPath, 'scripts'))
import utils

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", type=str, help="The path of logfile folder", default=os.path.join(ROOTPath, "log_folder"))
    parser.add_argument("--log_name", type=str, help="log file name", default="step3_filter_kg2_nodes_and_edges.log")
    parser.add_argument("--graph_nodes", type=str, help="Raw graph node file", default=os.path.join(ROOTPath, "data", "all_graph_nodes_info.txt"))
    parser.add_argument("--graph_edges", type=str, help="Raw graph edge file", default=os.path.join(ROOTPath, "data", "graph_edges.txt"))
    parser.add_argument("--pub_threshold", type=float, help="Threshold used to filter number of supported publications", default=10)
    parser.add_argument("--biolink_version", type=str, help="Biolink version", default="4.2.0")
    parser.add_argument("--output_folder", type=str, help="The path of output folder", default=os.path.join(ROOTPath, "data"))
    args = parser.parse_args()

    logger = utils.get_logger(os.path.join(args.log_dir, args.log_name))
    logger.info(args)
    output_path = args.output_folder

    ## Read raw translator KG edge list and node list
    logger.info("Read raw translator KG edge list and node list")
    raw_graph_edges = pl.read_csv(args.graph_edges, separator='\t')
    raw_graph_nodes = pl.read_csv(args.graph_nodes, separator='\t')
    all_nodes_from_edges = set(raw_graph_edges['source'].to_list() + raw_graph_edges['target'].to_list())
    raw_graph_nodes = raw_graph_nodes.filter(pl.col('id').is_in(all_nodes_from_edges))

    ## Filter translator KG nodes and edges based on categories
    logger.info("Filter translator KG nodes and edges based on categories")
    removed_category_set = ['biolink:Mammal', 'biolink:PhysicalEntity', 'biolink:LifeStage', 'biolink:EnvironmentalFeature', 'biolink:Attribute', 'biolink:NamedThing', 'biolink:Human', 'biolink:EnvironmentalProcess', 'biolink:GeographicLocation', 'biolink:CellularOrganism', 'biolink:CellularComponent', 'biolink:Device', 'biolink:OrganismalEntity', 'biolink:InformationResource', 'biolink:Publication', 'biolink:Event', 'biolink:AdministrativeEntity', 'biolink:ClinicalIntervention', 'biolink:Phenomenon', 'biolink:Procedure', 'biolink:IndividualOrganism', 'biolink:Activity', 'biolink:Cell', 'biolink:Treatment', 'biolink:Cohort', 'biolink:PopulationOfIndividualOrganisms', 'biolink:ClinicalAttribute', 'biolink:GrossAnatomicalStructure', 'biolink:Agent', 'biolink:AnatomicalEntity', 'biolink:InformationContentEntity', 'biolink:MaterialSample', 'biolink:StudyPopulation']
    filtered_graph_nodes = raw_graph_nodes.filter(~pl.col('primary_category').is_in(removed_category_set))
    valid_node_ids = set(filtered_graph_nodes['id'].to_list())
    filtered_graph_edges = raw_graph_edges.filter(
        pl.col('source').is_in(valid_node_ids) & pl.col('target').is_in(valid_node_ids)
    )

    ## Split edges into SemMedDB edges and Non-SemMedDB edges
    logger.info("Split edges into SemMedDB edges and Non-SemMedDB edges")
    semmeddb_edges = []
    non_semmeddb_edges = []
    for edge in tqdm(filtered_graph_edges.iter_rows(named=True)):
        p_publications = edge['p_publications'] if isinstance(edge['p_publications'], list) else json.loads(edge['p_publications'])
        p_knowledge_source = edge['p_knowledge_source'] if isinstance(edge['p_knowledge_source'], list) else json.loads(edge['p_knowledge_source'])
        new_edge = [edge['source'], edge['target'], edge['predicate'], len(p_publications), p_knowledge_source]
        if len(new_edge[4]) == 1 and new_edge[4][0] == 'infores:semmeddb':
            semmeddb_edges.append(new_edge)
        else:
            non_semmeddb_edges.append(new_edge)

    ## Filter edges based on number of supported publications
    logger.info("Filter edges based on number of supported publications")
    semmeddb_edges_df = pl.DataFrame(
        semmeddb_edges,
        schema=['source', 'target', 'predicate', 'num_publications', 'p_knowledge_source'],
        orient='row',
    )
    filtered_semmeddb_edges_df = semmeddb_edges_df.filter(pl.col('num_publications') >= args.pub_threshold)

    ## Combine SemMedDB edges and Non-SemMedDB edges
    logger.info("Combine SemMedDB edges and Non-SemMedDB edges")
    non_semmeddb_edges_df = pl.DataFrame(
        non_semmeddb_edges,
        schema=['source', 'target', 'predicate', 'num_publications', 'p_knowledge_source'],
        orient='row',
    )
    combined_edges_df = pl.concat([non_semmeddb_edges_df, filtered_semmeddb_edges_df])

    ## Filter "redundant" edges based on Biolink edge hierarchy
    logger.info("Filter 'redundant' edges based on Biolink edge hierarchy")

    biolink_helper = utils.get_biolink_helper(biolink_version=args.biolink_version)

    unique_predicates = combined_edges_df['predicate'].unique().to_list()
    logger.info(f"Pre-computing ancestors for {len(unique_predicates)} unique predicates")
    predicate_ancestors = {}
    for pred in unique_predicates:
        ancestors = set(biolink_helper.get_ancestors(pred, include_mixins=False))
        ancestors.discard(pred)
        predicate_ancestors[pred] = ancestors

    edge_pair_predicates = {}
    for row in tqdm(combined_edges_df.iter_rows(named=True), desc="Grouping predicates by edge pair"):
        key = (row['source'], row['target'])
        edge_pair_predicates.setdefault(key, set()).add(row['predicate'])

    edges = []
    for (subject_id, object_id), predicates in tqdm(edge_pair_predicates.items(), desc="Filtering redundant predicates"):
        all_ancestors = set()
        for pred in predicates:
            all_ancestors.update(predicate_ancestors.get(pred, set()))
        for pred in predicates - all_ancestors:
            edges.append([subject_id, object_id, pred])

    temp_df = pl.DataFrame(edges, schema=['source', 'target', 'predicate'], orient='row')

    ## Save filtered nodes and edges
    logger.info("Save filtered nodes and edges")
    out_edges_df = temp_df.join(combined_edges_df, on=['source', 'target', 'predicate'], how='left')
    out_edges_df = out_edges_df.with_columns(
        pl.col('p_knowledge_source').map_elements(lambda s: json.dumps(list(s)), return_dtype=pl.Utf8),
    )
    out_edges_df.write_csv(os.path.join(output_path, "filtered_graph_edges.txt"), separator='\t')
    all_nodes_from_edges = set(out_edges_df['source'].to_list() + out_edges_df['target'].to_list())
    out_graph_nodes = filtered_graph_nodes.filter(pl.col('id').is_in(all_nodes_from_edges))
    out_graph_nodes.write_csv(os.path.join(output_path, "filtered_graph_nodes_info.txt"), separator='\t')
