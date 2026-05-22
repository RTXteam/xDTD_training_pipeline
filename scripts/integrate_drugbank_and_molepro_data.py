## Import Standard Packages
import polars as pl
import os, sys
import argparse
from tqdm import tqdm
import json
import subprocess
import csv
csv.field_size_limit(sys.maxsize)

## Import Personal Packages
pathlist = os.getcwd().split(os.path.sep)
ROOTindex = pathlist.index("xDTD_training_pipeline")
ROOTPath = os.path.sep.join([*pathlist[:(ROOTindex + 1)]])
sys.path.append(os.path.join(ROOTPath, 'scripts'))
import utils

DRUG_TYPES = {'biolink:SmallMolecule', 'biolink:ChemicalEntity', 'biolink:Drug'}
GENE_TYPES = {'biolink:Gene', 'biolink:Protein'}


def _pick_primary_drug_category(cats):
    """Pick the most specific drug category from a set of biolink categories."""
    if 'biolink:SmallMolecule' in cats:
        return 'biolink:SmallMolecule'
    if 'biolink:Drug' in cats:
        return 'biolink:Drug'
    return 'biolink:ChemicalEntity'


def _pick_primary_gene_category(cats):
    """Pick the most specific gene category from a set of biolink categories."""
    if 'biolink:Gene' in cats:
        return 'biolink:Gene'
    return 'biolink:Protein'

class MoleProData:

    def __init__(self, logger, molepro_aws_link: str = 'https://molepro.s3.amazonaws.com'):
        self.molepro_aws_link = molepro_aws_link
        self.logger = logger

    def _download_data(self, output: str):
        os.makedirs(output, exist_ok=True)
        for filename in ('nodes.tsv', 'edges.tsv'):
            filepath = os.path.join(output, filename)
            if not os.path.exists(filepath):
                self.logger.info(f"Downloading {filename} from MolePro AWS")
                subprocess.run(
                    f"wget -O {filepath} {self.molepro_aws_link}/{filename}",
                    shell=True,
                )

    def load_data(self, output: str):
        self._download_data(output)
        self.logger.info("Starting to load data from MolePro AWS")

        raw_node_ids = []
        with open(os.path.join(output, 'nodes.tsv'), newline='', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter='\t')
            next(reader)
            for row in reader:
                raw_node_ids.append(row[0])

        self.logger.info(f"Normalizing {len(raw_node_ids)} MolePro nodes via Node Normalization API")
        utils.batch_normalize_curies(raw_node_ids)

        self.nodes_dict = {}
        norm_map = {}
        for curie in raw_node_ids:
            info = utils.get_node_norm_info(curie)
            if info is None:
                continue
            norm_id = info['preferred_curie']
            types = set(info['types'])
            if types & DRUG_TYPES:
                primary_cat = _pick_primary_drug_category(types)
            elif types & GENE_TYPES:
                primary_cat = _pick_primary_gene_category(types)
            else:
                continue
            norm_map[curie] = (norm_id, primary_cat)
            self.nodes_dict[(norm_id, primary_cat)] = 1

        self.logger.info(
            f"Normalized {len(norm_map)} nodes; "
            f"kept {len(self.nodes_dict)} drug/gene/protein entries in nodes_dict"
        )

        self.edges_dict = {}
        with open(os.path.join(output, 'edges.tsv'), newline='', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter='\t')
            header = next(reader)
            pred_idx = header.index('predicate')
            subj_idx = header.index('subject')
            obj_idx = header.index('object')
            pub_idx = header.index('biolink:publications')
            for row in reader:
                if row[pred_idx] not in ("biolink:affects", "biolink:interacts_with"):
                    continue
                subj_entry = norm_map.get(row[subj_idx])
                obj_entry = norm_map.get(row[obj_idx])
                if subj_entry is None or obj_entry is None:
                    continue
                subject_norm = subj_entry[0]
                object_norm = obj_entry[0]
                self.edges_dict.setdefault(subject_norm, set()).add((object_norm, row[pub_idx]))


    def extract_drug_target_pairs_from_kg(self, query_node, query_type, pmid_support=True):
        schema = {'subject': pl.Utf8, 'object': pl.Utf8}
        if pmid_support:
            schema['pmids'] = pl.Utf8
        empty_df = pl.DataFrame(schema=schema)

        if (query_node, query_type) not in self.nodes_dict:
            return empty_df
        if query_node not in self.edges_dict:
            return empty_df

        if pmid_support:
            result = [
                (query_node, target, pubs.split('|'))
                for target, pubs in self.edges_dict[query_node]
                if pubs
            ]
            columns = ['subject', 'object', 'pmids']
        else:
            result = [(query_node, target) for target, _ in self.edges_dict[query_node]]
            columns = ['subject', 'object']

        return pl.DataFrame(result, schema=columns, orient='row') if result else empty_df


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", type=str, help="The path of logfile folder", default=os.path.join(ROOTPath, "log_folder"))
    parser.add_argument("--log_name", type=str, help="log file name", default="step8_integrate_drugbank_and_molepro_data.log")
    parser.add_argument("--nodes_jsonl", type=str, help="path to translator KG nodes JSONL", default=os.path.join(ROOTPath, "data", "translator_kg", "nodes.jsonl"))
    parser.add_argument("--drugbank_export_paths", type=str, help='path to a file containing drugbank-based paths', default=os.path.join(ROOTPath, "data", "expert_path_files", "p_expert_paths.txt"))
    parser.add_argument('--molepro_aws_link', type=str, help='AWS link of Molecular Data Provider', default='https://molepro.s3.amazonaws.com')
    parser.add_argument("--output_folder", type=str, help="The path of output folder", default=os.path.join(ROOTPath, "data"))
    args = parser.parse_args()

    logger = utils.get_logger(os.path.join(args.log_dir, args.log_name))
    logger.info(args)

    ## ── Step 1: Build drug list from translator KG ──────────────────────
    all_drugs_path = os.path.join(args.output_folder, 'expert_path_files', 'all_drugs.txt')
    if not os.path.exists(all_drugs_path):
        logger.info(f"Loading drug entities from {args.nodes_jsonl}")
        drug_records = []
        with open(args.nodes_jsonl) as f:
            for line in tqdm(f, desc="Reading nodes for drug entities"):
                node = json.loads(line)
                cats = set(node.get('category', []))
                if cats & DRUG_TYPES:
                    drug_records.append({
                        'id': node['id'],
                        'category': _pick_primary_drug_category(cats),
                        'equivalent_curies': json.dumps(node.get('equivalent_identifiers', [])),
                    })

        res = pl.DataFrame(drug_records)
        res.write_csv(all_drugs_path, separator='\t')
        res_data = [(r['id'], r['category']) for r in drug_records]
    else:
        res = pl.read_csv(all_drugs_path, separator='\t')
        res_data = [(row[0], row[1]) for row in res.iter_rows()]

    ## ── Step 2: Read drugbank processed data ────────────────────────────
    p_expert_paths = pl.read_csv(args.drugbank_export_paths, separator='\t', has_header=False)
    p_expert_paths.columns = ['drugbankid', 'subject', 'object']
    p_expert_paths = p_expert_paths.filter(pl.col('object').is_not_null())

    ## ── Step 3: Query MolePro (Molecular Data Provider) ─────────────────
    molepro_backup = os.path.join(args.output_folder, 'expert_path_files', 'molepro_df_backup.txt')
    if not os.path.exists(molepro_backup):
        molepro_obj = MoleProData(logger, args.molepro_aws_link)
        molepro_obj.load_data(os.path.join(args.output_folder, 'expert_path_files', 'temp_molepro'))

        molepro_frames = []
        for node_id, category in tqdm(res_data, desc='Querying MolePro data'):
            frame = molepro_obj.extract_drug_target_pairs_from_kg(node_id, category, pmid_support=True)
            if frame.shape[0] > 0:
                molepro_frames.append(frame)

        molepro_df = (
            pl.concat(molepro_frames) if molepro_frames
            else pl.DataFrame(schema={'subject': pl.Utf8, 'object': pl.Utf8, 'pmids': pl.Utf8})
        )

        backup_df = molepro_df.with_columns(
            pl.col('pmids').map_elements(lambda s: json.dumps(list(s)), return_dtype=pl.Utf8)
        )
        backup_df.write_csv(molepro_backup, separator='\t')
    else:
        molepro_df = pl.read_csv(molepro_backup, separator='\t')
        molepro_data = [
            (row['subject'], row['object'], json.loads(row['pmids']))
            for row in molepro_df.iter_rows(named=True)
        ]
        molepro_df = pl.DataFrame(molepro_data, schema=['subject', 'object', 'pmids'], orient='row')

    ## ── Step 4: Deduplicate and filter PMIDs per (source, target) pair ──
    pair_pmids = {}
    for row in tqdm(molepro_df.iter_rows(named=True), total=molepro_df.shape[0]):
        source, target, pmids = row['subject'], row['object'], row['pmids']
        valid = {p for p in (pmids if isinstance(pmids, list) else [])
                 if isinstance(p, str) and p.startswith('PMID:')}
        key = (source, target)
        if key in pair_pmids:
            pair_pmids[key] |= valid
        else:
            pair_pmids[key] = valid

    molepro_records = [(s, t, list(ps)) for (s, t), ps in pair_pmids.items()]
    molepro_df = pl.DataFrame(molepro_records, schema=['subject', 'object', 'pmids'], orient='row')

    ## ── Step 5: Merge with drugbank paths ───────────────────────────────
    molepro_for_join = molepro_df.with_columns(
        pl.col('pmids').map_elements(lambda s: json.dumps(list(s)), return_dtype=pl.Utf8).alias('pmids_str')
    ).drop('pmids')

    combined_table = molepro_for_join.join(
        p_expert_paths.select(['subject', 'object', 'drugbankid']),
        on=['subject', 'object'],
        how='full',
        coalesce=True,
    )

    combined_table = combined_table.rename({'pmids_str': 'pmids'}).with_columns(
        pl.when(pl.col('pmids').is_not_null() & pl.col('drugbankid').is_not_null())
          .then(pl.lit('drugbank&molepro'))
          .when(pl.col('drugbankid').is_not_null())
          .then(pl.lit('drugbank'))
          .when(pl.col('pmids').is_not_null())
          .then(pl.lit('molepro'))
          .otherwise(pl.lit(None))
          .alias('supported_sources')
    )

    ## ── Output ──────────────────────────────────────────────────────────
    combined_table.write_csv(
        os.path.join(args.output_folder, 'expert_path_files', 'p_expert_paths_combined.txt'),
        separator='\t',
    )
