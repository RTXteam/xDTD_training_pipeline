import sys
import os
import argparse
import json
import polars as pl

## Import Personal Packages
pathlist = os.getcwd().split(os.path.sep)
ROOTindex = pathlist.index("xDTD_training_pipeline")
ROOTPath = os.path.sep.join([*pathlist[:(ROOTindex + 1)]])
sys.path.append(os.path.join(ROOTPath, 'scripts'))
import utils


def normalize_and_filter(df, id_col, name_col, kg_node_ids, biolink_version, logger):
    """Normalize CURIEs via Node Norm API, then keep only those present in the KG.

    Parameters
    ----------
    df : pl.DataFrame with at least `id_col` and `name_col` columns.
    id_col : str – output column name for the normalized CURIE.
    name_col : str – output column name for the normalized name.
    kg_node_ids : set – valid node IDs from filtered_graph_nodes_info.txt.
    biolink_version : str
    logger : logging.Logger

    Returns
    -------
    pl.DataFrame with columns:
        <id_col>, <name_col>, primary_category, categories
    """
    curies = df[id_col].drop_nulls().unique().to_list()
    logger.info(f"Batch-normalizing {len(curies)} unique CURIEs")
    utils.batch_normalize_curies(curies)

    rows = []
    dropped_norm = 0
    dropped_kg = 0
    for row in df.iter_rows(named=True):
        info = utils.get_node_norm_info(row[id_col])
        if info is None:
            dropped_norm += 1
            continue
        preferred_curie = info['preferred_curie']
        if preferred_curie not in kg_node_ids:
            dropped_kg += 1
            continue
        cats = info.get('types', [])
        rows.append({
            id_col: preferred_curie,
            name_col: info.get('preferred_name') or row[name_col],
            'primary_category': utils.get_primary_category(cats, biolink_version),
            'categories': json.dumps(cats),
        })

    logger.info(f"Dropped {dropped_norm} (normalization failed), {dropped_kg} (not in KG)")
    result = pl.DataFrame(rows).unique(subset=[id_col])
    logger.info(f"{result.height} unique entries after normalization and KG filtering")
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Normalize and filter drug/disease lists against the KG")
    parser.add_argument("--log_dir", type=str, help="Log folder path", default=os.path.join(ROOTPath, "log_folder"))
    parser.add_argument("--log_name", type=str, help="Log file name", default="step4_process_drug_disease_list.log")
    parser.add_argument("--drug_list", type=str, help="Path to matrix-drug-list.tsv", required=True)
    parser.add_argument("--disease_list", type=str, help="Path to matrix-disease-list.tsv", required=True)
    parser.add_argument("--graph_nodes", type=str, help="Path to filtered_graph_nodes_info.txt", required=True)
    parser.add_argument("--biolink_version", type=str, help="Biolink model version", default="4.2.0")
    parser.add_argument("--output_folder", type=str, help="Output folder", default=os.path.join(ROOTPath, "data", "drug_disease_list"))
    args = parser.parse_args()

    logger = utils.get_logger(os.path.join(args.log_dir, args.log_name))
    logger.info(args)
    os.makedirs(args.output_folder, exist_ok=True)

    ## ── Load valid KG node IDs ──────────────────────────────────────────
    logger.info("Loading KG node IDs from filtered_graph_nodes_info.txt")
    kg_nodes = pl.read_csv(args.graph_nodes, separator='\t')
    kg_node_ids = set(kg_nodes['id'].to_list())
    logger.info(f"KG contains {len(kg_node_ids)} nodes")

    ## ── Process drug list ───────────────────────────────────────────────
    logger.info("=== Processing drug list ===")
    drug_df = pl.read_csv(args.drug_list, separator='\t').select([
        pl.col('translator_id').alias('drug_id'),
        pl.col('name').alias('drug_name'),
    ]).drop_nulls()
    logger.info(f"{drug_df.height} drugs after column selection and null removal")

    drug_result = normalize_and_filter(
        drug_df, 'drug_id', 'drug_name', kg_node_ids, args.biolink_version, logger,
    )
    drug_path = os.path.join(args.output_folder, 'drug_list.txt')
    drug_result.write_csv(drug_path, separator='\t')
    logger.info(f"Saved {drug_result.height} drugs to {drug_path}")

    ## ── Process disease list ────────────────────────────────────────────
    logger.info("=== Processing disease list ===")
    disease_df = pl.read_csv(args.disease_list, separator='\t').select([
        pl.col('id').alias('disease_id'),
        pl.col('name').alias('disease_name'),
    ]).drop_nulls()
    logger.info(f"{disease_df.height} diseases after column selection and null removal")

    disease_result = normalize_and_filter(
        disease_df, 'disease_id', 'disease_name', kg_node_ids, args.biolink_version, logger,
    )
    disease_path = os.path.join(args.output_folder, 'disease_list.txt')
    disease_result.write_csv(disease_path, separator='\t')
    logger.info(f"Saved {disease_result.height} diseases to {disease_path}")
