"""Convert EC v2.10.0 ground truth TSV to the PSU pipeline format (tp_pairs.txt / tn_pairs.txt).

EC GT format (gt_pairs_raw.tsv): source, target, indication, contraindication, y, drug|disease
PSU format: drug_id, drug_name, drug_primary_category, drug_categories,
            disease_id, disease_name, disease_primary_category, disease_categories

Usage:
    python convert_ec_ground_truth.py \
        --ec_gt_file matrix_project/v0.4.5/.../gt_pairs_raw.tsv \
        --drug_list shared_data/drug_disease_list/drug_list.txt \
        --disease_list shared_data/drug_disease_list/disease_list.txt \
        --graph_nodes shared_data/intersected_graph_nodes_info.txt \
        --biolink_version 4.2.0 \
        --output_folder shared_data/ground_truth_pairs \
        --log_dir log_folder
"""

import argparse
import json
import os
import sys

import polars as pl

pathlist = os.getcwd().split(os.path.sep)
ROOTindex = pathlist.index("xDTD_training_pipeline")
ROOTPath = os.path.sep.join([*pathlist[:(ROOTindex + 1)]])
sys.path.append(os.path.join(ROOTPath, 'scripts'))
import utils

VALID_DRUG_CATEGORIES = {'biolink:SmallMolecule', 'biolink:Drug', 'biolink:ChemicalEntity'}
VALID_DISEASE_CATEGORIES = {'biolink:Disease', 'biolink:PhenotypicFeature'}


def normalize_ec_pairs(df, biolink_version, logger):
    """Normalize EC GT pairs and produce PSU-format DataFrame.

    EC pairs already have normalized CURIEs as source/target,
    but we still need to get names and category info via Node Norm API.
    """
    all_curies = list(set(
        df['source'].unique().to_list() + df['target'].unique().to_list()
    ))
    logger.info(f"Batch-normalizing {len(all_curies)} unique CURIEs from EC GT")
    utils.batch_normalize_curies(all_curies)

    rows = []
    dropped_norm = 0
    dropped_cat = 0
    for row in df.iter_rows(named=True):
        drug_info = utils.get_node_norm_info(row['source'])
        disease_info = utils.get_node_norm_info(row['target'])
        if drug_info is None or disease_info is None:
            dropped_norm += 1
            continue

        drug_cats = drug_info.get('types', [])
        disease_cats = disease_info.get('types', [])

        if not (set(drug_cats) & VALID_DRUG_CATEGORIES):
            dropped_cat += 1
            continue
        if not (set(disease_cats) & VALID_DISEASE_CATEGORIES):
            dropped_cat += 1
            continue

        rows.append({
            'drug_id': drug_info['preferred_curie'],
            'drug_name': drug_info.get('preferred_name') or '',
            'drug_primary_category': utils.get_primary_category(drug_cats, biolink_version),
            'drug_categories': json.dumps(drug_cats),
            'disease_id': disease_info['preferred_curie'],
            'disease_name': disease_info.get('preferred_name') or '',
            'disease_primary_category': utils.get_primary_category(disease_cats, biolink_version),
            'disease_categories': json.dumps(disease_cats),
        })

    logger.info(f"Dropped {dropped_norm} (normalization failed), "
                f"{dropped_cat} (category mismatch)")

    result = pl.DataFrame(rows)
    before = result.height
    result = result.unique(subset=['drug_id', 'disease_id'])
    logger.info(f"Deduplicated: {before} → {result.height} unique pairs")
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Convert EC v2.10.0 GT to PSU pipeline format"
    )
    parser.add_argument("--ec_gt_file", type=str, required=True,
                        help="Path to EC gt_pairs_raw.tsv")
    parser.add_argument("--drug_list", type=str, required=True,
                        help="Path to drug_list.txt from step4")
    parser.add_argument("--disease_list", type=str, required=True,
                        help="Path to disease_list.txt from step4")
    parser.add_argument("--graph_nodes", type=str, required=True,
                        help="Path to intersected_graph_nodes_info.txt")
    parser.add_argument("--biolink_version", type=str, default="4.2.0")
    parser.add_argument("--output_folder", type=str, required=True)
    parser.add_argument("--log_dir", type=str, default=os.path.join(ROOTPath, "log_folder"))
    parser.add_argument("--log_name", type=str,
                        default="step5_convert_ec_ground_truth.log")
    args = parser.parse_args()

    logger = utils.get_logger(os.path.join(args.log_dir, args.log_name))
    logger.info(args)
    os.makedirs(args.output_folder, exist_ok=True)

    # ── Load EC ground truth ─────────────────────────────────────────────
    ec_gt = pl.read_csv(args.ec_gt_file, separator='\t')
    logger.info(f"Loaded EC GT: {ec_gt.height} rows")
    logger.info(f"  Indications (y=1): {ec_gt.filter(pl.col('y')==1).height}")
    logger.info(f"  Contraindications (y=0): {ec_gt.filter(pl.col('y')==0).height}")
    logger.info(f"  Unique drugs: {ec_gt['source'].n_unique()}")
    logger.info(f"  Unique diseases: {ec_gt['target'].n_unique()}")

    # ── Load existing drug/disease lists from step4 ──────────────────────
    existing_drug_df = pl.read_csv(args.drug_list, separator='\t')
    existing_disease_df = pl.read_csv(args.disease_list, separator='\t')
    existing_drug_ids = set(existing_drug_df['drug_id'].to_list())
    existing_disease_ids = set(existing_disease_df['disease_id'].to_list())
    logger.info(f"Loaded {len(existing_drug_ids)} drugs, "
                f"{len(existing_disease_ids)} diseases from step4")

    # ── Load KG node IDs ─────────────────────────────────────────────────
    kg_node_ids = set(
        pl.read_csv(args.graph_nodes, separator='\t')['id'].to_list()
    )
    logger.info(f"KG contains {len(kg_node_ids)} nodes")

    # ── Process TP pairs (y=1) ───────────────────────────────────────────
    logger.info("=== Processing indication pairs (y=1 → TP) ===")
    tp_ec = ec_gt.filter(pl.col('y') == 1)
    tp_pairs = normalize_ec_pairs(tp_ec, args.biolink_version, logger)

    # ── Process TN pairs (y=0) ───────────────────────────────────────────
    logger.info("=== Processing contraindication pairs (y=0 → TN) ===")
    tn_ec = ec_gt.filter(pl.col('y') == 0)
    tn_pairs = normalize_ec_pairs(tn_ec, args.biolink_version, logger)

    # ── Augment drug/disease lists ───────────────────────────────────────
    logger.info("=== Augmenting drug/disease lists ===")
    all_pairs = pl.concat([tp_pairs, tn_pairs])

    new_drug_rows = []
    for drug_id in all_pairs['drug_id'].unique().to_list():
        if drug_id not in existing_drug_ids and drug_id in kg_node_ids:
            info = utils.get_node_norm_info(drug_id)
            if info is not None:
                cats = info.get('types', [])
                new_drug_rows.append({
                    'drug_id': drug_id,
                    'drug_name': info.get('preferred_name', ''),
                    'primary_category': utils.get_primary_category(
                        cats, args.biolink_version
                    ),
                    'categories': json.dumps(cats),
                })

    new_disease_rows = []
    for disease_id in all_pairs['disease_id'].unique().to_list():
        if disease_id not in existing_disease_ids and disease_id in kg_node_ids:
            info = utils.get_node_norm_info(disease_id)
            if info is not None:
                cats = info.get('types', [])
                new_disease_rows.append({
                    'disease_id': disease_id,
                    'disease_name': info.get('preferred_name', ''),
                    'primary_category': utils.get_primary_category(
                        cats, args.biolink_version
                    ),
                    'categories': json.dumps(cats),
                })

    logger.info(f"New drugs to add (in KG): {len(new_drug_rows)}")
    logger.info(f"New diseases to add (in KG): {len(new_disease_rows)}")

    if new_drug_rows:
        updated_drug_df = pl.concat([existing_drug_df, pl.DataFrame(new_drug_rows)])
        updated_drug_df.write_csv(args.drug_list, separator='\t')
        logger.info(f"Updated drug_list: {existing_drug_df.height} → "
                     f"{updated_drug_df.height}")

    if new_disease_rows:
        updated_disease_df = pl.concat(
            [existing_disease_df, pl.DataFrame(new_disease_rows)]
        )
        updated_disease_df.write_csv(args.disease_list, separator='\t')
        logger.info(f"Updated disease_list: {existing_disease_df.height} → "
                     f"{updated_disease_df.height}")

    # ── Filter and save ──────────────────────────────────────────────────
    logger.info("=== Filtering pairs by KG presence ===")
    before_tp = tp_pairs.height
    tp_pairs = tp_pairs.filter(
        pl.col('drug_id').is_in(kg_node_ids)
        & pl.col('disease_id').is_in(kg_node_ids)
    )
    tp_path = os.path.join(args.output_folder, 'tp_pairs.txt')
    tp_pairs.write_csv(tp_path, separator='\t')
    logger.info(f"TP pairs: {before_tp} → {tp_pairs.height} (in KG), saved to {tp_path}")

    before_tn = tn_pairs.height
    tn_pairs = tn_pairs.filter(
        pl.col('drug_id').is_in(kg_node_ids)
        & pl.col('disease_id').is_in(kg_node_ids)
    )
    tn_path = os.path.join(args.output_folder, 'tn_pairs.txt')
    tn_pairs.write_csv(tn_path, separator='\t')
    logger.info(f"TN pairs: {before_tn} → {tn_pairs.height} (in KG), saved to {tn_path}")

    logger.info("EC ground truth conversion complete.")
