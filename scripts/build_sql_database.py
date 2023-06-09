
"""This script will build a database for the pre-computation of explainable DTD model.
Author: Chunyu Ma
"""

import os, sys
import pandas as pd
import numpy as np
import argparse

from tqdm import tqdm
import sqlite3
import logging

## Import Personal Packages
pathlist = os.getcwd().split(os.path.sep)
ROOTindex = pathlist.index("xDTD_training_pipeline")
ROOTPath = os.path.sep.join([*pathlist[:(ROOTindex + 1)]])
data_path = os.path.join(ROOTPath, 'data')
model_path = os.path.join(ROOTPath, 'models')
sys.path.append(os.path.join(ROOTPath, 'scripts'))
import utils

DEBUG = True

class BuildExplainableDTD(object):

    # Constructor
    def __init__(self, args, path_to_score_results, path_to_path_results, database_name=None, outdir=None):
        """
        Args:
            args (argparse): arguments from command line
            path_to_score_results (str): path to a folder containing the prediction score results of all diseases
            path_to_path_results (str): path to a folder containing the path results of all diseases
            database_name (str, optional): database name (Defaults: ExplainableDTD.db).
            outdir (str, optional): path to a folder where the database is generated (Defaults: ./).
        """
        self.logger = utils.get_logger(os.path.join(args.log_dir, args.log_name))

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
        if database_name is None:
            self.database_name = "ExplainableDTD.db"
        else:
            self.database_name = database_name
        if outdir is None:
            self.database_name = './'
        else:
            if not os.path.exists(outdir):
                os.makedirs(outdir)
            self.outdir = outdir
        self.success_con = self.connect()

    def __del__(self):
        self.disconnect()

    # Create and store a database connection
    def connect(self):
        database = f"{self.outdir}/{self.database_name}"

        if os.path.exists(database):
            self.connection = sqlite3.connect(database)
            self.logger.info("Connecting to database")
            return True
        else:
            self.connection = sqlite3.connect(database)
            print("INFO: Connecting to database", flush=True)
            return True

    # Destroy the database connection
    def disconnect(self):

        if self.success_con is True:
            self.connection.commit()
            self.connection.close()
            self.logger.info("Disconnecting from database")
            self.success_con = False
        else:
            self.logger.info("No database was connected! So skip disconnecting from database.")
            return

    # Delete and create the tables
    def create_tables(self):

        if self.success_con is True:
            self.logger.info(f"Creating database {self.database_name}")
            self.connection.execute(f"DROP TABLE IF EXISTS PREDICTION_SCORE_TABLE")
            self.connection.execute(f"CREATE TABLE PREDICTION_SCORE_TABLE( drug_id VARCHAR(255), drug_name VARCHAR(255), disease_id VARCHAR(255), disease_name VARCHAR(255), tn_score FLOAT, tp_score FLOAT, unknown_score FLOAT)")
            self.connection.execute(f"DROP TABLE IF EXISTS PATH_RESULT_TABLE")
            self.connection.execute(f"CREATE TABLE PATH_RESULT_TABLE(  drug_id VARCHAR(255), drug_name VARCHAR(255), disease_id VARCHAR(255), disease_name VARCHAR(255), path VARCHAR(255), path_score FLOAT )")
            self.logger.info(f"Creating tables is completed")

    ## Populate the tables
    def populate_table(self):

        if self.success_con is True:
            # save score results to local database
            score_reulst_list = os.listdir(self.path_to_score_results)
            for file_name in tqdm(score_reulst_list):

                with open(f"{self.path_to_score_results}/{file_name}", 'r') as file_in:
                    content_list = file_in.readlines()
                    col_name = content_list.pop(0)
                    insert_command1 = f"INSERT INTO PREDICTION_SCORE_TABLE("
                    insert_command2 = f" values ("
                    for col in col_name.strip().split("\t"):
                        insert_command1 = insert_command1 + f"{col},"
                        insert_command2 = insert_command2 + f"?,"

                    insert_command = insert_command1 + ")" + insert_command2 + ")"
                    insert_command = insert_command.replace(',)', ')')

                    if DEBUG:
                        print(insert_command, flush=True)
                        print(tuple(content_list[0].strip().split("\t")), flush=True)

                    for line in content_list:
                        line = tuple(line.strip().split("\t"))
                        self.connection.execute(insert_command, line)

                    self.connection.commit()
            self.connection.commit()

            # save path results to local database
            path_reulst_list = os.listdir(self.path_to_path_results)
            for file_name in tqdm(path_reulst_list):

                with open(f"{self.path_to_path_results}/{file_name}", 'r') as file_in:
                    content_list = file_in.readlines()
                    col_name = content_list.pop(0)
                    insert_command1 = f"INSERT INTO PATH_RESULT_TABLE("
                    insert_command2 = f" values ("
                    for col in col_name.strip().split("\t"):
                        insert_command1 = insert_command1 + f"{col},"
                        insert_command2 = insert_command2 + f"?,"

                    insert_command = insert_command1 + ")" + insert_command2 + ")"
                    insert_command = insert_command.replace(',)', ')')

                    if DEBUG:
                        print(insert_command, flush=True)
                        print(tuple(content_list[0].strip().split("\t")), flush=True)

                    for line in content_list:
                        line = tuple(line.strip().split("\t"))
                        self.connection.execute(insert_command, line)

                    self.connection.commit()
            self.connection.commit()

            self.logger.info(f"Populating tables is completed")

    def create_indexes(self):

        if self.success_con is True:
            self.logger.info(f"Creating INDEXes on PREDICTION_SCORE_TABLE",)
            self.connection.execute(f"CREATE INDEX idx_PREDICTION_SCORE_TABLE_drug_id ON PREDICTION_SCORE_TABLE(drug_id)")
            self.connection.execute(f"CREATE INDEX idx_PREDICTION_SCORE_TABLE_drug_name ON PREDICTION_SCORE_TABLE(drug_name)")
            self.connection.execute(f"CREATE INDEX idx_PREDICTION_SCORE_TABLE_disease_id ON PREDICTION_SCORE_TABLE(disease_id)")
            self.connection.execute(f"CREATE INDEX idx_PREDICTION_SCORE_TABLE_disease_name ON PREDICTION_SCORE_TABLE(disease_name)")

            self.logger.info(f"Creating INDEXes on PREDICTION_SCORE_TABLE",)
            self.connection.execute(f"CREATE INDEX idx_PATH_RESULT_TABLE_drug_id ON PATH_RESULT_TABLE(drug_id)")
            self.connection.execute(f"CREATE INDEX idx_PATH_RESULT_TABLE_drug_name ON PATH_RESULT_TABLE(drug_name)")
            self.connection.execute(f"CREATE INDEX idx_PATH_RESULT_TABLE_disease_id ON PATH_RESULT_TABLE(disease_id)")
            self.connection.execute(f"CREATE INDEX idx_PATH_RESULT_TABLE_disease_name ON PATH_RESULT_TABLE(disease_name)")

            self.logger.info(f"INFO: Creating INDEXes is completed")

    def get_top_drugs_for_disease(self, disease_ids):
        """get top drugs predicted by DTD model for given disease ids

        Args:
            disease_ids (str|list): a string of disease curie id or a list of disease curies, e.g. "MONDO:0008753" or ["MONDO:0008753","MONDO:0005148","MONDO:0005155"]

        Returns:
            top_drugs (pd.DataFrame): the top drugs predicted by DTD model for given disease ids
        """

        cursor = self.connection.cursor()
        columns = ["drug_id","drug_name","disease_id","disease_name","tn_score","tp_score","unknown_score"]
        if isinstance(disease_ids, str):
            cursor.execute(f"select drug_id,drug_name,disease_id,disease_name,tn_score,tp_score,unknown_score from PREDICTION_SCORE_TABLE where disease_id='{disease_ids}';")
            res = cursor.fetchall()
            top_drugs= pd.DataFrame(res, columns=columns)
            return top_drugs
        elif isinstance(disease_ids, list):
            cursor.execute(f"select drug_id,drug_name,disease_id,disease_name,tn_score,tp_score,unknown_score from PREDICTION_SCORE_TABLE where disease_id in {tuple(set(disease_ids))};")
            res = cursor.fetchall()
            top_drugs = pd.DataFrame(res, columns=columns)
            return top_drugs
        else:
            print("The 'dataset_id' in get_top_drugs_for_disease should be a string or a list", flush=True)
            top_drugs = pd.DataFrame([], columns=columns)
            return top_drugs

    def get_top_paths_for_disease(self, disease_ids):
        """get top paths predicted by DTD model for given disease ids

        Args:
            disease_ids (str|list): a string of disease curie id or a list of disease curies, e.g. "MONDO:0008753" or ["MONDO:0008753","MONDO:0005148","MONDO:0005155"]

        Returns:
            top_paths (dict): the top paths predicted by DTD model for given disease ids
        """

        cursor = self.connection.cursor()
        columns = ["drug_id","drug_name","disease_id","disease_name","path","path_score"]
        top_paths = dict()
        if isinstance(disease_ids, str):
            cursor.execute(f"select drug_id,drug_name,disease_id,disease_name,path,path_score from PATH_RESULT_TABLE where disease_id='{disease_ids}';")
            res = cursor.fetchall()
            temp_df = pd.DataFrame(res, columns=columns)
            for drug_id, disease_id in temp_df[['drug_id','disease_id']].drop_duplicates().values:
                temp = np.array(temp_df.loc[(temp_df['drug_id']==drug_id) & (temp_df['disease_id']==disease_id),['path','path_score']].values).tolist()
                if len(temp) > 0:
                    top_paths[(drug_id, disease_id)] = temp
            return top_paths
        elif isinstance(disease_ids, list):
            cursor.execute(f"select drug_id,drug_name,disease_id,disease_name,path,path_score from PATH_RESULT_TABLE where disease_id in {tuple(set(disease_ids))};")
            res = cursor.fetchall()
            temp_df = pd.DataFrame(res, columns=columns)
            for drug_id, disease_id in temp_df[['drug_id','disease_id']].drop_duplicates().values:
                temp = np.array(temp_df.loc[(temp_df['drug_id']==drug_id) & (temp_df['disease_id']==disease_id),['path','path_score']].values).tolist()
                if len(temp) > 0:
                    top_paths[(drug_id, disease_id)] = temp
            return top_paths
        else:
            print("The 'dataset_id' in get_top_drugs_for_disease should be a string or a list", flush=True)
            return top_paths

####################################################################################################

def main():
    parser = argparse.ArgumentParser(description="Tests or builds the ExplainableDTD Database", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--log_dir", type=str, help="The path of logfile folder", default=os.path.join(ROOTPath, "log_folder"))
    parser.add_argument("--log_name", type=str, help="log file name", default="build_sql_database.log")
    parser.add_argument('--build', action="store_true", required=False, help="If set, (re)build the index from scratch", default=False)
    parser.add_argument('--test', action="store_true", required=False, help="If set, run a test of database by doing several lookups", default=False)
    parser.add_argument('--path_to_score_results', type=str, required=True, help="Path to a folder containing the prediction score results of all diseases")
    parser.add_argument('--path_to_path_results', type=str, required=True, help="Path to a folder containing the path results of all diseases")
    parser.add_argument('--database_name', type=str, required=False, help="Database name", default="ExplainableDTD.db")
    parser.add_argument('--outdir', type=str, required=False, help="Path to a folder where the database is generated", default="./")
    args = parser.parse_args()

    if not args.build and not args.test:
        parser.print_help()
        sys.exit(2)

    EDTDdb = BuildExplainableDTD(args, args.path_to_score_results, args.path_to_path_results, database_name=args.database_name, outdir=args.outdir)

    # To (re)build
    if args.build:
        EDTDdb.create_tables()
        EDTDdb.populate_table()
        EDTDdb.create_indexes()

    # Exit here if tests are not requested
    if not args.test:
        return

    print("==== Testing for search for top drugs by disease id ====", flush=True)
    print(EDTDdb.get_top_drugs_for_disease('MONDO:0005148'))
    # print(EDTDdb.get_top_drugs_for_disease(["MONDO:0008753","MONDO:0005148","MONDO:0005155"]))

    print("==== Testing for search for top paths by disease id ====", flush=True)
    print(EDTDdb.get_top_paths_for_disease('MONDO:0005148'))
    # print(EDTDdb.get_top_paths_for_disease(["MONDO:0008753","MONDO:0005148","MONDO:0005155"]))

####################################################################################################

if __name__ == "__main__":
    main()