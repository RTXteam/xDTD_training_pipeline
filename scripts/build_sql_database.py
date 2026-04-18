"""This script builds a SQLite database for the pre-computated results of explainable DTD model."""

import os
import sys
import pandas as pd
import argparse
from tqdm import tqdm
import sqlite3

## Import Personal Packages
pathlist = os.getcwd().split(os.path.sep)
ROOTindex = pathlist.index("xDTD_training_pipeline")
ROOTPath = os.path.sep.join([*pathlist[:(ROOTindex + 1)]])
sys.path.append(os.path.join(ROOTPath, 'scripts'))
import utils


class BuildExplainableDTD(object):

    def __init__(self, logger, path_to_score_results, path_to_path_results, database_name=None, outdir=None):
        """
        Args:
            logger: logging.Logger instance.
            path_to_score_results (str): path to a folder containing the prediction score results of all diseases
            path_to_path_results (str): path to a folder containing the path results of all diseases
            database_name (str, optional): database name (Defaults: ExplainableDTD.db).
            outdir (str, optional): path to a folder where the database is generated (Defaults: ./).
        """
        self.logger = logger

        if not os.path.exists(path_to_score_results) or not len(os.listdir(path_to_score_results)) > 0:
            self.logger.error(f"The given path '{path_to_score_results}' doesn't exist or is an empty folder")
            raise
        else:
            self.path_to_score_results = path_to_score_results
        if not os.path.exists(path_to_path_results) or not len(os.listdir(path_to_path_results)) > 0:
            self.logger.error(f"The given path '{path_to_path_results}' doesn't exist or is an empty folder")
            raise
        else:
            self.path_to_path_results = path_to_path_results

        self.database_name = database_name or "ExplainableDTD.db"
        if outdir is None:
            outdir = './'
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        self.outdir = outdir
        self.success_con = self._connect()

    def __del__(self):
        self._disconnect()

    def _connect(self):
        database = os.path.join(self.outdir, self.database_name)
        self.conn = sqlite3.connect(database)
        self.logger.info(f"Connected to database: {database}")
        return True

    def _disconnect(self):
        if self.success_con is True:
            self.conn.commit()
            self.conn.close()
            self.logger.info("Disconnected from database")
            self.success_con = False

    def create_tables(self):
        if self.success_con is not True:
            return
        self.logger.info(f"Creating database {self.database_name}")
        self.conn.execute("DROP TABLE IF EXISTS PREDICTION_SCORE_TABLE")
        self.conn.execute("""
            CREATE TABLE PREDICTION_SCORE_TABLE(
                drug_id VARCHAR(255),
                drug_name VARCHAR(255),
                disease_id VARCHAR(255),
                disease_name VARCHAR(255),
                tn_score FLOAT,
                tp_score FLOAT,
                unknown_score FLOAT
            )
        """)
        self.conn.execute("DROP TABLE IF EXISTS PATH_RESULT_TABLE")
        self.conn.execute("""
            CREATE TABLE PATH_RESULT_TABLE(
                drug_id VARCHAR(255),
                drug_name VARCHAR(255),
                disease_id VARCHAR(255),
                disease_name VARCHAR(255),
                path VARCHAR(255),
                path_score FLOAT
            )
        """)
        self.logger.info("Creating tables is completed")

    def populate_table(self):
        if self.success_con is not True:
            return

        BATCH_SIZE = 50000
        self.conn.execute("PRAGMA journal_mode = WAL")
        self.conn.execute("PRAGMA synchronous = OFF")
        self.conn.execute("PRAGMA cache_size = -2000000")

        SCORE_INSERT = "INSERT INTO PREDICTION_SCORE_TABLE(drug_id, drug_name, disease_id, disease_name, tn_score, tp_score, unknown_score) VALUES (?,?,?,?,?,?,?)"
        PATH_INSERT = "INSERT INTO PATH_RESULT_TABLE(drug_id, drug_name, disease_id, disease_name, path, path_score) VALUES (?,?,?,?,?,?)"

        score_result_list = os.listdir(self.path_to_score_results)
        self.logger.info(f"Inserting {len(score_result_list)} score result files into PREDICTION_SCORE_TABLE...")
        batch = []
        for file_name in tqdm(score_result_list, desc="score_results"):
            filepath = os.path.join(self.path_to_score_results, file_name)
            with open(filepath, 'r') as f:
                next(f)
                for line in f:
                    batch.append(tuple(line.strip().split("\t")))
                    if len(batch) >= BATCH_SIZE:
                        self.conn.executemany(SCORE_INSERT, batch)
                        self.conn.commit()
                        batch = []
        if batch:
            self.conn.executemany(SCORE_INSERT, batch)
            self.conn.commit()
        self.logger.info("Inserting score results is completed")

        path_result_list = os.listdir(self.path_to_path_results)
        self.logger.info(f"Inserting {len(path_result_list)} path result files into PATH_RESULT_TABLE...")
        batch = []
        for file_name in tqdm(path_result_list, desc="path_results"):
            filepath = os.path.join(self.path_to_path_results, file_name)
            with open(filepath, 'r') as f:
                next(f)
                for line in f:
                    batch.append(tuple(line.strip().split("\t")))
                    if len(batch) >= BATCH_SIZE:
                        self.conn.executemany(PATH_INSERT, batch)
                        self.conn.commit()
                        batch = []
        if batch:
            self.conn.executemany(PATH_INSERT, batch)
            self.conn.commit()
        self.logger.info("Inserting path results is completed")

        self.conn.execute("PRAGMA synchronous = NORMAL")
        self.logger.info("Populating tables is completed")

    def create_indexes(self):
        if self.success_con is not True:
            return
        self.logger.info("Creating indexes on PREDICTION_SCORE_TABLE")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_PREDICTION_SCORE_TABLE_drug_id ON PREDICTION_SCORE_TABLE(drug_id)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_PREDICTION_SCORE_TABLE_drug_name ON PREDICTION_SCORE_TABLE(drug_name)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_PREDICTION_SCORE_TABLE_disease_id ON PREDICTION_SCORE_TABLE(disease_id)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_PREDICTION_SCORE_TABLE_disease_name ON PREDICTION_SCORE_TABLE(disease_name)")

        self.logger.info("Creating indexes on PATH_RESULT_TABLE")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_PATH_RESULT_TABLE_drug_id ON PATH_RESULT_TABLE(drug_id)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_PATH_RESULT_TABLE_drug_name ON PATH_RESULT_TABLE(drug_name)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_PATH_RESULT_TABLE_disease_id ON PATH_RESULT_TABLE(disease_id)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_PATH_RESULT_TABLE_disease_name ON PATH_RESULT_TABLE(disease_name)")
        self.conn.commit()
        self.logger.info("Creating indexes is completed")

    def get_top_drugs_for_disease(self, disease_ids):
        """Get top drugs predicted by DTD model for given disease ids.

        Args:
            disease_ids (str|list): disease curie id(s), e.g. "MONDO:0008753" or ["MONDO:0008753","MONDO:0005148"]

        Returns:
            pd.DataFrame with columns: drug_id, drug_name, disease_id, disease_name, tn_score, tp_score, unknown_score
        """
        cursor = self.conn.cursor()
        columns = ["drug_id", "drug_name", "disease_id", "disease_name", "tn_score", "tp_score", "unknown_score"]
        query = "SELECT drug_id, drug_name, disease_id, disease_name, tn_score, tp_score, unknown_score FROM PREDICTION_SCORE_TABLE WHERE disease_id"

        if isinstance(disease_ids, str):
            cursor.execute(f"{query} = ?", (disease_ids,))
        elif isinstance(disease_ids, list):
            placeholders = ','.join('?' * len(disease_ids))
            cursor.execute(f"{query} IN ({placeholders})", list(set(disease_ids)))
        else:
            self.logger.error("disease_ids must be a string or a list")
            return pd.DataFrame([], columns=columns)

        return pd.DataFrame(cursor.fetchall(), columns=columns)

    def get_top_paths_for_disease(self, disease_ids):
        """Get top paths predicted by DTD model for given disease ids.

        Args:
            disease_ids (str|list): disease curie id(s), e.g. "MONDO:0008753" or ["MONDO:0008753","MONDO:0005148"]

        Returns:
            dict mapping (drug_id, disease_id) -> list of [path, path_score]
        """
        cursor = self.conn.cursor()
        query = "SELECT drug_id, disease_id, path, path_score FROM PATH_RESULT_TABLE WHERE disease_id"

        if isinstance(disease_ids, str):
            cursor.execute(f"{query} = ?", (disease_ids,))
        elif isinstance(disease_ids, list):
            placeholders = ','.join('?' * len(disease_ids))
            cursor.execute(f"{query} IN ({placeholders})", list(set(disease_ids)))
        else:
            self.logger.error("disease_ids must be a string or a list")
            return {}

        top_paths = {}
        for drug_id, disease_id, path, path_score in cursor.fetchall():
            key = (drug_id, disease_id)
            top_paths.setdefault(key, []).append([path, path_score])
        return top_paths


def main():
    parser = argparse.ArgumentParser(
        description="Build or query the ExplainableDTD database",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--log_dir", type=str, help="The path of logfile folder", default=os.path.join(ROOTPath, "log_folder"))
    parser.add_argument("--log_name", type=str, help="Log file name", default="build_sql_database.log")
    parser.add_argument('--build', action="store_true", help="If set, (re)build the index from scratch", default=False)
    parser.add_argument('--test', action="store_true", help="If set, run a test of database by doing several lookups", default=False)
    parser.add_argument('--path_to_score_results', type=str, help="Path to a folder containing the prediction score results of all diseases")
    parser.add_argument('--path_to_path_results', type=str, help="Path to a folder containing the path results of all diseases")
    parser.add_argument('--database_name', type=str, default="ExplainableDTD.db", help="Database name")
    parser.add_argument('--outdir', type=str, default="./", help="Path to a folder where the database is generated")
    args = parser.parse_args()

    logger = utils.get_logger(os.path.join(args.log_dir, args.log_name))

    if not args.build and not args.test:
        parser.print_help()
        sys.exit(2)

    EDTDdb = BuildExplainableDTD(logger, args.path_to_score_results, args.path_to_path_results, database_name=args.database_name, outdir=args.outdir)

    if args.build:
        logger.info("==== Creating tables ====")
        EDTDdb.create_tables()
        logger.info("==== Populating tables ====")
        EDTDdb.populate_table()
        logger.info("==== Creating indexes ====")
        EDTDdb.create_indexes()

    if not args.test:
        return

    logger.info("==== Testing: top drugs by disease id ====")
    logger.info(EDTDdb.get_top_drugs_for_disease('MONDO:0000313'))
    logger.info("==== Testing: top paths by disease id ====")
    logger.info(EDTDdb.get_top_paths_for_disease('MONDO:0000313'))


if __name__ == "__main__":
    main()
