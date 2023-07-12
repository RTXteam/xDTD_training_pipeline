## Import Standard Packages
import sys, os
import pandas as pd
import argparse
import json
from tqdm import tqdm

## Import Personal Packages
pathlist = os.getcwd().split(os.path.sep)
ROOTindex = pathlist.index("xDTD_training_pipeline")
ROOTPath = os.path.sep.join([*pathlist[:(ROOTindex + 1)]])
sys.path.append(os.path.join(ROOTPath, 'scripts'))
import utils


def get_database_subpath(path: str):
    path_chunks = path.split("/")
    last_two_chunks = path_chunks[-2:]
    return "/".join(last_two_chunks)

def get_remote_location(path: str):
    db_host = 'arax-databases.rtx.ai'
    db_username = 'rtxconfig'
    databases_server_dir_path = '/home/rtxconfig'
    database_subpath = get_database_subpath(path)
    return f"{db_username}@{db_host}:{databases_server_dir_path}/{database_subpath}"

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", type=str, help="The path of logfile folder", default=os.path.join(ROOTPath, "log_folder"))
    parser.add_argument("--log_name", type=str, help="log file name", default="download_data_and_kg.log")
    parser.add_argument("--db_config_path", type=str, help="path to database config file", default="../config_dbs.json")
    parser.add_argument("--secret_config_path", type=str, help="path to secret config file", default="../config_secrets.json")
    parser.add_argument("--output_folder", type=str, help="The path of output folder", default=os.path.join(ROOTPath, "data"))
    args = parser.parse_args()

    logger = utils.get_logger(os.path.join(args.log_dir,args.log_name))
    logger.info(args)
    output_path = args.output_folder

    ## Load the contents of two config files (e.g., config_dbs.json and config_secrets.json)
    with open(args.db_config_path, 'rb') as file_in:
        config_dbs = json.load(file_in)
    with open(args.secret_config_path, 'rb') as file_in:
        config_secrets = json.load(file_in)

    ## Download NodeSynonymizer from server if it doesn't exist
    node_synonymizer_path = config_dbs["database_downloads"]["node_synonymizer"]
    node_synonymizer_name = node_synonymizer_path.split('/')[-1]
    if not os.path.exists(os.path.join(ROOTPath, "data", node_synonymizer_name)):
        os.system(f"scp {get_remote_location(node_synonymizer_path)} {os.path.join(ROOTPath, 'data', node_synonymizer_name)}")
    ## Download CurieToPmids database from server if it doesn't exist
    curie_to_pmids_path = config_dbs["database_downloads"]["curie_to_pmids"]
    curie_to_pmids_name = curie_to_pmids_path.split('/')[-1]
    if not os.path.exists(os.path.join(ROOTPath, "data", curie_to_pmids_name)):
        os.system(f"scp {get_remote_location(curie_to_pmids_path)} {os.path.join(ROOTPath, 'data', curie_to_pmids_name)}")

    ## Connect to neo4j database
    neo4j_instance = config_dbs["neo4j"]["KG2c"]
    neo4j_bolt = f"bolt://{neo4j_instance}:7687"
    neo4j_username = config_secrets["neo4j"]["KG2c"]["username"]
    neo4j_password = config_secrets["neo4j"]["KG2c"]["password"]
    conn = utils.Neo4jConnection(uri=neo4j_bolt, user=neo4j_username, pwd=neo4j_password)

    ## Pull a dataframe of all graph edges
    query = "match (disease) where (disease.category='biolink:Disease' or disease.category='biolink:PhenotypicFeature' or disease.category='biolink:BehavioralFeature' or disease.category='biolink:DiseaseOrPhenotypicFeature') with collect(distinct disease.id) as disease_ids match (drug) where (drug.category='biolink:Drug' or drug.category='biolink:ChemicalEntity' or drug.category='biolink:SmallMolecule') with collect(distinct drug.id) as drug_ids, disease_ids as disease_ids match (m1)<-[r]-(m2) where m1<>m2 and not (m1.id in drug_ids and m2.id in disease_ids) and not (m1.id in disease_ids and m2.id in drug_ids) with distinct m1 as node1, r as edge, m2 as node2 return node2.id as source, node1.id as target, edge.predicate as predicate, edge.publications as publications, edge.primary_knowledge_source as primary_knowledge_source"
    KG_alledges = conn.query(query)
    KG_alledges.columns = ['source','target','predicate', 'p_publications', 'p_knowledge_source']
    logger.info(f"Total number of triples in kg after removing all edges between drug entities and disease entities: {len(KG_alledges)}")
    temp_dict = {} 
    for row in tqdm(KG_alledges.to_numpy(), desc="Merge edges with same subject/predicate/object"):
        source, target, predicate, p_publications, p_knowledge_source = row
        p_publications = p_publications if isinstance(p_publications, list) else []
        p_knowledge_source = [x.strip() for x in p_knowledge_source.split(';')]
        if (source, predicate, target) in temp_dict:
            temp_dict[(source, predicate, target)]["p_publications"] += p_publications
            temp_dict[(source, predicate, target)]["p_knowledge_source"] += p_knowledge_source
        else:
            temp_dict[(source, predicate, target)] = {"p_publications": p_publications, "p_knowledge_source": p_knowledge_source}
    KG_alledges = pd.DataFrame([(key[0], key[2], key[1], list(set(value['p_publications'])), list(set(value['p_knowledge_source']))) for key, value in tqdm(temp_dict.items(), desc="Convert dictionary to dataframe")], columns=['source','target','predicate', 'p_publications', 'p_knowledge_source'])
    KG_alledges.to_csv(os.path.join(output_path, 'graph_edges.txt'), sep='\t', index=None)

    ## Pulls a dataframe of all graph nodes with category label
    query = "match (n) with distinct n.id as id, n.category as category, n.name as name, n.all_names as all_names, n.description as des return id, category, name, all_names, des"
    KG_allnodes_label = conn.query(query)
    KG_allnodes_label.columns = ['id','category','name', 'all_names', 'des']
    for index in range(len(KG_allnodes_label)):
        KG_allnodes_label.loc[index,'all_names'] = list(set([x.lower() for x in KG_allnodes_label.loc[index,'all_names']])) if KG_allnodes_label.loc[index,'all_names'] else KG_allnodes_label.loc[index,'all_names']
    logger.info(f"Total number of entities: {len(KG_allnodes_label)}")
    for i in range(len(KG_allnodes_label)):
        if KG_allnodes_label.loc[i, "des"]:
            KG_allnodes_label.loc[i, "des"] = " ".join(KG_allnodes_label.loc[i, "des"].replace("\n", " ").split())

    KG_allnodes_label = KG_allnodes_label.apply(lambda row: [row[0], row[1], utils.clean_up_name(row[2]), list(set([utils.clean_up_name(name) for name in row[3]])) if row[3] is not None else [''], utils.clean_up_desc(row[4])], axis=1, result_type='expand')
    KG_allnodes_label.columns = ['id','category','name', 'all_names', 'des']

    KG_allnodes_label.to_csv(os.path.join(output_path, 'all_graph_nodes_info.txt'), sep='\t', index=None)


