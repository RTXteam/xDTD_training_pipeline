## Import Standard Packages
import sys, os
import polars as pl
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
    parser.add_argument("--log_name", type=str, help="log file name", default="step2_process_translator_kg.log")
    parser.add_argument("--nodes_jsonl", type=str, help="path to translator KG nodes JSONL file", default=os.path.join(ROOTPath, "data", "translator_kg", "nodes.jsonl"))
    parser.add_argument("--edges_jsonl", type=str, help="path to translator KG edges JSONL file", default=os.path.join(ROOTPath, "data", "translator_kg", "edges.jsonl"))
    parser.add_argument("--biolink_version", type=str, help="Biolink version", default="4.2.0")
    parser.add_argument("--output_folder", type=str, help="The path of output folder", default=os.path.join(ROOTPath, "data"))
    args = parser.parse_args()

    output_path = args.output_folder

    logger = utils.get_logger(os.path.join(args.log_dir, args.log_name))
    logger.info(args)

    DRUG_CATEGORIES = {'biolink:Drug', 'biolink:ChemicalEntity', 'biolink:SmallMolecule'}
    DISEASE_CATEGORIES = {'biolink:Disease', 'biolink:PhenotypicFeature'}

    ## ============================================================
    ## Load all nodes from JSONL and assign primary category
    ## ============================================================
    logger.info(f"Loading nodes from {args.nodes_jsonl}")
    node_records = []
    drug_ids = set()
    disease_ids = set()

    with open(args.nodes_jsonl) as f:
        for line in tqdm(f, desc="Reading nodes"):
            node = json.loads(line)
            node_id = node['id']
            categories = node.get('category', [])
            name = node.get('name', '') or ''
            description = node.get('description', '') or ''
            eq_ids = node.get('equivalent_identifiers', [])

            category_set = set(categories)
            if category_set & DRUG_CATEGORIES:
                drug_ids.add(node_id)
            if category_set & DISEASE_CATEGORIES:
                disease_ids.add(node_id)

            primary_category = utils.get_primary_category(categories, biolink_version=args.biolink_version)

            node_records.append({
                'id': node_id,
                'primary_category': primary_category,
                'all_categories': json.dumps(sorted(category_set)),
                'name': utils.clean_up_name(name),
                'description': utils.clean_up_desc(description),
                'equivalent_identifiers': json.dumps(eq_ids),
            })

    logger.info(f"Total nodes loaded: {len(node_records)}")
    logger.info(f"Drug nodes: {len(drug_ids)}, Disease nodes: {len(disease_ids)}")

    KG_allnodes_label = pl.DataFrame(node_records)
    KG_allnodes_label.write_csv(os.path.join(output_path, 'all_graph_nodes_info.txt'), separator='\t')

    ## ============================================================
    ## Load all edges from JSONL, remove drug-disease edges, merge
    ## ============================================================
    logger.info(f"Loading edges from {args.edges_jsonl}")
    temp_dict = {}

    with open(args.edges_jsonl) as f:
        for line in tqdm(f, desc="Reading edges"):
            edge = json.loads(line)
            source = edge['subject']
            target = edge['object']
            predicate = edge['predicate']

            if source == target:
                continue

            # Remove all edges between drug and disease entities
            if (source in drug_ids and target in disease_ids) or (source in disease_ids and target in drug_ids):
                continue

            publications = edge.get('publications', []) or []
            primary_sources = [
                s['resource_id']
                for s in edge.get('sources', [])
                if s.get('resource_role') == 'primary_knowledge_source'
            ]

            key = (source, predicate, target)
            if key in temp_dict:
                temp_dict[key]['p_publications'] += publications
                temp_dict[key]['p_knowledge_source'] += primary_sources
            else:
                temp_dict[key] = {
                    'p_publications': list(publications),
                    'p_knowledge_source': list(primary_sources),
                }

    logger.info(f"Total number of triples in kg after removing all edges between drug entities and disease entities: {len(temp_dict)}")

    edge_records = []
    for (source, predicate, target), value in tqdm(temp_dict.items(), desc="Building edge DataFrame"):
        edge_records.append({
            'source': source,
            'target': target,
            'predicate': predicate,
            'p_publications': list(set(value['p_publications'])),
            'p_knowledge_source': list(set(value['p_knowledge_source'])),
        })

    KG_alledges = pl.DataFrame(edge_records).with_columns(
        pl.col('p_publications').map_elements(lambda s: json.dumps(list(s)), return_dtype=pl.Utf8),
        pl.col('p_knowledge_source').map_elements(lambda s: json.dumps(list(s)), return_dtype=pl.Utf8),
    )
    KG_alledges.write_csv(os.path.join(output_path, 'graph_edges.txt'), separator='\t')
    logger.info(f"Saved {len(KG_alledges)} edges to graph_edges.txt")
    logger.info(f"Saved {len(KG_allnodes_label)} nodes to all_graph_nodes_info.txt")
