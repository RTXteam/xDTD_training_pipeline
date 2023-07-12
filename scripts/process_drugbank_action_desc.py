## Import Standard Packages
import xmltodict
import pandas as pd
import os, sys
import pickle
import argparse
import json

## Import Personal Packages
pathlist = os.getcwd().split(os.path.sep)
ROOTindex = pathlist.index("xDTD_training_pipeline")
ROOTPath = os.path.sep.join([*pathlist[:(ROOTindex + 1)]])
sys.path.append(os.path.join(ROOTPath, 'scripts'))
import utils

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", type=str, help="The path of logfile folder", default=os.path.join(ROOTPath, "log_folder"))
    parser.add_argument("--log_name", type=str, help="log file name", default="process_drugbank_action_desc.log")
    parser.add_argument("--db_config_path", type=str, help="path to database config file", default="../config_dbs.json")
    parser.add_argument("--secret_config_path", type=str, help="path to secret config file", default="../config_secrets.json")
    parser.add_argument('--drugbankxml', type=str, help='Path to the drugbank xml file downloaded from https://go.drugbank.com/releases/latest')
    parser.add_argument("--output_folder", type=str, help="The path of output folder", default=os.path.join(ROOTPath, "data"))
    args = parser.parse_args()

    logger = utils.get_logger(os.path.join(args.log_dir,args.log_name))
    logger.info(args)

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
    res = conn.query(f"match (n) where n.category='biolink:SmallMolecule' or n.category='biolink:Drug' or n.category='biolink:ChemicalEntity' return distinct n.id, n.category, n.equivalent_curies")

    ## select possible drug entities which have drugbank ids in their synonyms
    res = res.apply(lambda row: [row[0], row[1], row[2], len([synonym for synonym in row[2] if synonym.split(':')[0]=='DRUGBANK'])>0], axis=1, result_type='expand')
    res = res.loc[res[3].isin([True]),[0,1,2]].reset_index(drop=True)
    drug_curies = res.apply(lambda row: [row[0], row[1], [synonym for synonym in row[2] if synonym.split(':')[0]=='DRUGBANK']], axis=1, result_type='expand')
    
    ## parse drugbank database xml file and read its content
    ## download 'drugbank.xml' release(2021-01-3) from DrugBank website 'https://go.drugbank.com/releases/latest' and put it in data/ folder
    with open(args.drugbankxml) as infile:
        doc = xmltodict.parse(infile.read())
        
    ## collect information(e.g. drug descriptions, indications, drug action mechanism, targets) for each drugbank id which has these informaton and store them in a dictionary
    drugbank_dict = dict()
    for index in range(len(doc['drugbank']['drug'])):
        if type(doc['drugbank']['drug'][index]['drugbank-id']) is list:
            drugbankid = doc['drugbank']['drug'][index]['drugbank-id'][0]['#text']
        else:
            drugbankid = doc['drugbank']['drug'][index]['drugbank-id']['#text']
        if doc['drugbank']['drug'][index]['mechanism-of-action'] is not None:
            drugbank_dict[drugbankid] = dict()
            drugbank_dict[drugbankid]['name'] = doc['drugbank']['drug'][index]['name']
            if doc['drugbank']['drug'][index]['description'] is not None:
                drugbank_dict[drugbankid]['description'] = doc['drugbank']['drug'][index]['description'].replace('\r\n\r\n','###################')
            else:
                drugbank_dict[drugbankid]['description'] = None
            if doc['drugbank']['drug'][index]['pharmacodynamics'] is not None:
                drugbank_dict[drugbankid]['pharmacodynamics'] = doc['drugbank']['drug'][index]['pharmacodynamics'].replace('\r\n\r\n','###################')
            else:
                drugbank_dict[drugbankid]['pharmacodynamics'] = None
            if doc['drugbank']['drug'][index]['mechanism-of-action']:
                drugbank_dict[drugbankid]['mechanism-of-action'] = doc['drugbank']['drug'][index]['mechanism-of-action'].replace('\r\n\r\n','###################')
            else:
                drugbank_dict[drugbankid]['mechanism-of-action'] = None
            if doc['drugbank']['drug'][index]['indication'] is not None:
                drugbank_dict[drugbankid]['indication'] = doc['drugbank']['drug'][index]['indication'].replace('\r\n\r\n','###################')
            else:
                drugbank_dict[drugbankid]['indication'] = None
            if doc['drugbank']['drug'][index]['targets'] is not None:
                targets_info = doc['drugbank']['drug'][index]['targets']['target']
                drugbank_dict[drugbankid]['targets'] = []
                if type(targets_info) is list:
                    for target in targets_info:
                        temp = [target['name'], target['organism'], target['known-action']]
                        if 'polypeptide' in target:
                            try:
                                temp += [(target['polypeptide']['gene-name'],target['polypeptide']['@source'],target['polypeptide']['@id'],target['polypeptide']['specific-function'])]
                            except:
                                temp += [target['polypeptide']]
                        drugbank_dict[drugbankid]['targets'].append(temp)
                else:
                    target = targets_info
                    temp = [target['name'], target['organism'], target['known-action']]
                    if 'polypeptide' in target:
                        try:
                            temp += [(target['polypeptide']['gene-name'],target['polypeptide']['@source'],target['polypeptide']['@id'],target['polypeptide']['specific-function'])]
                        except:
                            temp += [target['polypeptide']]
                    drugbank_dict[drugbankid]['targets'].append(temp)
    
    ## filter out possible drug entities which don't have drug action mechanism description
    res = drug_curies.apply(lambda row: [row[0], row[1], row[2], len([synonym for synonym in row[2] if synonym.split(':')[1] in drugbank_dict]) > 0], axis=1, result_type='expand')
    res = res.loc[res[3],:].reset_index(drop=True)
    
    ## store mapping drug entity identifier in drugbank dict
    for index in range(len(res)):
        curie_id = res.loc[index,0]
        drugbank_ids = res.loc[index,2]
        for drugbank_id in drugbank_ids:
            if drugbank_id.split(':')[1] in drugbank_dict:
                drugbank_dict[drugbank_id.split(':')[1]]['source_curie'] = curie_id

    args.outdir = os.path.join(args.output_folder, 'expert_path_files')
    if not os.path.isdir(args.outdir):
        os.makedirs(args.outdir)

    ## save drugbank dict
    with open(os.path.join(args.outdir,'drugbank_dict.pkl'),'wb') as outfile:
        pickle.dump(drugbank_dict, outfile)
    
    ## convert drugbank dict to a drugbank mapping text file
    drugbank_ids = list(drugbank_dict.keys())
    df = []
    for drugbankid in drugbank_dict:
        if drugbank_dict[drugbankid].get('source_curie'):
            df += [(drugbank_dict[drugbankid]['source_curie'],drugbankid,drugbank_dict[drugbankid]['name'],drugbank_dict[drugbankid]['description'],drugbank_dict[drugbankid]['pharmacodynamics'],drugbank_dict[drugbankid]['mechanism-of-action'],drugbank_dict[drugbankid]['indication'])]
    drugbank_mapping = pd.DataFrame(df).rename(columns={0:'source curie',1:'corresponding drugbank id',2:'name',3:'description',4:'pharmacodynamics',5:'mechanism-of-action',6:'indication'})
    
    ## save drugbank mapping file
    drugbank_mapping.to_csv(os.path.join(args.outdir,'drugbank_mapping.txt'), sep='\t', index=None)
    
    ## for each mapping entity, we pair it with all targets (eg. protein) described in drugbank database
    df = []
    for index in range(len(drugbank_mapping)):
        drugbank_id = drugbank_mapping.loc[index,'corresponding drugbank id']
        if drugbank_dict[drugbank_id].get('targets'):
            for target in drugbank_dict[drugbank_id]['targets']:
                # if target[2] == 'yes':
                # use all potential targets
                try:
                    df += [(drugbank_id, drugbank_dict[drugbank_id]['source_curie'],'UniProtKB:'+target[3][2])]
                except:
                    df += [(drugbank_id, drugbank_dict[drugbank_id]['source_curie'],None)]
        else:
            df += [(drugbank_id, drugbank_dict[drugbank_id]['source_curie'],None)]
            
            
    ## save these drug-protein pairs in a file
    pd.DataFrame(df).to_csv(os.path.join(args.outdir,'p_expert_paths.txt'), sep='\t', index=None, header=None)