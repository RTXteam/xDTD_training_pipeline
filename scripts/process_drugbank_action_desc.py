## Import Standard Packages
import xmltodict
import polars as pl
import os, sys
import pickle
import argparse
import json
from tqdm import tqdm

## Import Personal Packages
pathlist = os.getcwd().split(os.path.sep)
ROOTindex = pathlist.index("xDTD_training_pipeline")
ROOTPath = os.path.sep.join([*pathlist[:(ROOTindex + 1)]])
sys.path.append(os.path.join(ROOTPath, 'scripts'))
import utils

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", type=str, help="The path of logfile folder", default=os.path.join(ROOTPath, "log_folder"))
    parser.add_argument("--log_name", type=str, help="log file name", default="step7_process_drugbank_action_desc.log")
    parser.add_argument("--nodes_jsonl", type=str, help="path to translator KG nodes JSONL", default=os.path.join(ROOTPath, "data", "translator_kg", "nodes.jsonl"))
    parser.add_argument('--drugbankxml', type=str, help='Path to the drugbank xml file downloaded from https://go.drugbank.com/releases/latest')
    parser.add_argument("--output_folder", type=str, help="The path of output folder", default=os.path.join(ROOTPath, "data"))
    args = parser.parse_args()

    logger = utils.get_logger(os.path.join(args.log_dir, args.log_name))
    logger.info(args)

    ## Extract all possible drug entities from JSONL
    DRUG_CATEGORIES = {'biolink:SmallMolecule', 'biolink:Drug', 'biolink:ChemicalEntity'}
    logger.info(f"Loading drug entities from {args.nodes_jsonl}")
    drug_records = []
    with open(args.nodes_jsonl) as f:
        for line in tqdm(f, desc="Reading nodes for drug entities"):
            node = json.loads(line)
            cats = set(node.get('category', []))
            if cats & DRUG_CATEGORIES:
                node_id = node['id']
                eq_ids = node.get('equivalent_identifiers', [])
                primary_cat = 'biolink:SmallMolecule' if 'biolink:SmallMolecule' in cats else ('biolink:Drug' if 'biolink:Drug' in cats else 'biolink:ChemicalEntity')
                drug_records.append((node_id, primary_cat, eq_ids))

    logger.info(f"Total drug entities: {len(drug_records)}")

    ## Select possible drug entities which have drugbank ids in their synonyms
    drugbank_drug_curies_records = []
    for node_id, category, eq_ids in drug_records:
        drugbank_ids = [synonym for synonym in eq_ids if synonym.split(':')[0] == 'DRUGBANK']
        if len(drugbank_ids) > 0:
            drugbank_drug_curies_records.append((node_id, category, drugbank_ids))

    ## Parse drugbank database xml file and read its content
    with open(args.drugbankxml) as infile:
        doc = xmltodict.parse(infile.read())

    ## Collect information (e.g. drug descriptions, indications, drug action mechanism, targets) for each drugbank id which has these informaton and store them in a dictionary
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

    ## Filter out drug entities that don't have drug action mechanism description
    filtered_drugbank_drug_curies = []
    for node_id, category, drugbank_ids in drugbank_drug_curies_records:
        has_moa = any(drugbank_id.split(':')[1] in drugbank_dict for drugbank_id in drugbank_ids)
        if has_moa:
            filtered_drugbank_drug_curies.append((node_id, category, drugbank_ids))

    ## Store mapping drug entity identifier in drugbank dict
    for node_id, category, drugbank_ids in filtered_drugbank_drug_curies:
        for drugbank_id in drugbank_ids:
            if drugbank_id.split(':')[1] in drugbank_dict:
                drugbank_dict[drugbank_id.split(':')[1]]['source_curie'] = node_id

    args.outdir = os.path.join(args.output_folder, 'expert_path_files')
    if not os.path.isdir(args.outdir):
        os.makedirs(args.outdir)

    ## Save drugbank dict
    with open(os.path.join(args.outdir, 'drugbank_dict.pkl'), 'wb') as outfile:
        pickle.dump(drugbank_dict, outfile)

    ## Convert drugbank dict to a drugbank mapping text file
    df = []
    for drugbankid in drugbank_dict:
        if drugbank_dict[drugbankid].get('source_curie'):
            df += [(drugbank_dict[drugbankid]['source_curie'], drugbankid, drugbank_dict[drugbankid]['name'], drugbank_dict[drugbankid]['description'], drugbank_dict[drugbankid]['pharmacodynamics'], drugbank_dict[drugbankid]['mechanism-of-action'], drugbank_dict[drugbankid]['indication'])]
    drugbank_mapping = pl.DataFrame(df, schema=['source curie', 'corresponding drugbank id', 'name', 'description', 'pharmacodynamics', 'mechanism-of-action', 'indication'], orient='row')

    ## Save drugbank mapping file
    drugbank_mapping.write_csv(os.path.join(args.outdir, 'drugbank_mapping.txt'), separator='\t')

    ## For each mapping entity, pair it with all targets described in drugbank database
    df = []
    for row in drugbank_mapping.iter_rows(named=True):
        drugbank_id = row['corresponding drugbank id']
        if drugbank_dict[drugbank_id].get('targets'):
            for target in drugbank_dict[drugbank_id]['targets']:
                try:
                    df += [(drugbank_id, drugbank_dict[drugbank_id]['source_curie'], 'UniProtKB:' + target[3][2])]
                except:
                    df += [(drugbank_id, drugbank_dict[drugbank_id]['source_curie'], None)]
        else:
            df += [(drugbank_id, drugbank_dict[drugbank_id]['source_curie'], None)]

    ## Save these drug-protein pairs in a file
    pl.DataFrame(df, orient='row').write_csv(os.path.join(args.outdir, 'p_expert_paths.txt'), separator='\t', include_header=False)
