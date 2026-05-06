
"""
This script builds a SQLite database for mapping nodes and edges from
translator KG JSONL files (nodes.jsonl, edges.jsonl).

"""

import os
import sys
import argparse
import collections
import json
from tqdm import tqdm
import sqlite3

## Import Personal Packages
pathlist = os.getcwd().split(os.path.sep)
ROOTindex = pathlist.index("xDTD_training_pipeline")
ROOTPath = os.path.sep.join([*pathlist[:(ROOTindex + 1)]])
sys.path.append(os.path.join(ROOTPath, 'scripts'))
import utils


class xDTDMappingDB():

    def __init__(self, logger, database_name='ExplainableDTD.db', outdir=None, mode='build', db_loc=None):
        """
        Args:
            logger: logging.Logger instance.
            database_name (str): database file name (default: ExplainableDTD.db).
            outdir (str): directory where the database is created (default: ./).
            mode (str): 'build' to create from scratch, 'run' to open existing.
            db_loc (str): path to the directory containing an existing database (used with mode='run').
        """
        self.logger = logger
        self.database_name = database_name

        if mode == 'build':
            if outdir is None:
                outdir = './'
            if not os.path.exists(outdir):
                os.makedirs(outdir)
            db_path = os.path.join(outdir, self.database_name)
            self.success_con = self._connect(db_path)
        elif mode == 'run':
            if db_loc is None:
                self.logger.error("db_loc must be provided in 'run' mode.")
                raise ValueError("db_loc is required for mode='run'")
            db_path = os.path.join(db_loc, database_name)
            self.success_con = self._connect(db_path)
        else:
            raise ValueError(f"Unknown mode '{mode}'. Use 'build' or 'run'.")

    def __del__(self):
        self._disconnect()

    def _connect(self, db_path):
        self.conn = sqlite3.connect(db_path)
        self.logger.info(f"Connected to database: {db_path}")
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
        self.logger.info(f"Creating tables in {self.database_name}")

        self.conn.execute("DROP TABLE IF EXISTS NODE_MAPPING_TABLE")
        self.conn.execute("""
            CREATE TABLE NODE_MAPPING_TABLE (
                id TEXT NOT NULL,
                name TEXT,
                category TEXT,
                equivalent_identifiers TEXT,
                description TEXT,
                synonym TEXT,
                xref TEXT,
                chembl_natural_product TEXT,
                chembl_availability_type TEXT,
                chembl_black_box_warning TEXT
            )
        """)

        self.conn.execute("DROP TABLE IF EXISTS EDGE_MAPPING_TABLE")
        self.conn.execute("""
            CREATE TABLE EDGE_MAPPING_TABLE (
                subject TEXT NOT NULL,
                predicate TEXT NOT NULL,
                object TEXT NOT NULL,
                id TEXT,
                category TEXT,
                qualifier TEXT,
                publications TEXT,
                sources TEXT,
                resource_id TEXT,
                resource_role TEXT,
                knowledge_level TEXT,
                agent_type TEXT,
                stage_qualifier TEXT,
                original_subject TEXT,
                original_object TEXT
            )
        """)
        self.conn.commit()
        self.logger.info("Creating tables is completed")

    def populate_tables(self, nodes_jsonl_path, edges_jsonl_path):
        if self.success_con is not True:
            return

        BATCH_SIZE = 50000
        NODE_INSERT = "INSERT INTO NODE_MAPPING_TABLE VALUES (?,?,?,?,?,?,?,?,?,?)"
        EDGE_INSERT = "INSERT INTO EDGE_MAPPING_TABLE VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)"

        self.conn.execute("PRAGMA journal_mode = WAL")
        self.conn.execute("PRAGMA synchronous = OFF")
        self.conn.execute("PRAGMA cache_size = -2000000")

        # --- Insert nodes ---
        self.logger.info("Inserting into NODE_MAPPING...")
        batch = []
        node_count = 0
        with open(nodes_jsonl_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="inserting into NODE_MAPPING_TABLE"):
                d = json.loads(line)
                row = (
                    d['id'],
                    d.get('name'),
                    json.dumps(d['category']) if 'category' in d else None,
                    json.dumps(d['equivalent_identifiers']) if 'equivalent_identifiers' in d else None,
                    d.get('description'),
                    json.dumps(d['synonym']) if 'synonym' in d else None,
                    json.dumps(d['xref']) if 'xref' in d else None,
                    str(d['chembl_natural_product']) if 'chembl_natural_product' in d else None,
                    d.get('chembl_availability_type'),
                    d.get('chembl_black_box_warning'),
                )
                batch.append(row)
                node_count += 1
                if len(batch) >= BATCH_SIZE:
                    self.conn.executemany(NODE_INSERT, batch)
                    self.conn.commit()
                    batch = []
        if batch:
            self.conn.executemany(NODE_INSERT, batch)
            self.conn.commit()
        self.logger.info(f"Inserted {node_count} rows into NODE_MAPPING_TABLE")

        # --- Insert edges ---
        self.logger.info("Inserting edges into EDGE_MAPPING_TABLE...")
        batch = []
        edge_count = 0
        with open(edges_jsonl_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="Inserting edges into EDGE_MAPPING_TABLE"):
                d = json.loads(line)
                sources = d.get('sources', [])
                resource_ids = '|'.join(s.get('resource_id', '') for s in sources)
                resource_roles = '|'.join(s.get('resource_role', '') for s in sources)
                row = (
                    d['subject'],
                    d['predicate'],
                    d['object'],
                    d.get('id'),
                    json.dumps(d['category']) if 'category' in d else None,
                    d.get('qualifier'),
                    json.dumps(d['publications']) if 'publications' in d else None,
                    json.dumps(sources) if sources else None,
                    resource_ids or None,
                    resource_roles or None,
                    d.get('knowledge_level'),
                    d.get('agent_type'),
                    d.get('stage_qualifier'),
                    d.get('original_subject'),
                    d.get('original_object'),
                )
                batch.append(row)
                edge_count += 1
                if len(batch) >= BATCH_SIZE:
                    self.conn.executemany(EDGE_INSERT, batch)
                    self.conn.commit()
                    batch = []
        if batch:
            self.conn.executemany(EDGE_INSERT, batch)
            self.conn.commit()
        self.logger.info(f"Inserted {edge_count} rows into EDGE_MAPPING_TABLE")

        self.conn.execute("PRAGMA synchronous = NORMAL")
        self.logger.info("Populating tables is completed")

    def create_indexes(self):
        if self.success_con is not True:
            return
        self.logger.info("Creating indexes on NODE_MAPPING_TABLE...")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_NODE_MAPPING_TABLE_id ON NODE_MAPPING_TABLE(id)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_NODE_MAPPING_TABLE_name ON NODE_MAPPING_TABLE(name)")

        self.logger.info("Creating indexes on EDGE_MAPPING_TABLE...")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_EDGE_MAPPING_TABLE_triple ON EDGE_MAPPING_TABLE(subject, predicate, object)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_EDGE_MAPPING_TABLE_subject ON EDGE_MAPPING_TABLE(subject)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_EDGE_MAPPING_TABLE_object ON EDGE_MAPPING_TABLE(object)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_EDGE_MAPPING_TABLE_predicate ON EDGE_MAPPING_TABLE(predicate)")
        self.conn.commit()
        self.logger.info("Creating indexes is completed")

    def get_node_info(self, node_id=None, node_name=None):
        NodeInfo = collections.namedtuple('NodeInfo', [
            'id', 'name', 'category', 'equivalent_identifiers', 'description',
            'synonym', 'xref', 'chembl_natural_product', 'chembl_availability_type',
            'chembl_black_box_warning'
        ])
        cursor = self.conn.cursor()
        if node_id is not None:
            cursor.execute("SELECT * FROM NODE_MAPPING_TABLE WHERE id = ?", (node_id,))
        elif node_name is not None:
            cursor.execute("SELECT * FROM NODE_MAPPING_TABLE WHERE name = ? COLLATE NOCASE", (node_name,))
        else:
            return None
        result = cursor.fetchone()
        return NodeInfo._make(result) if result else None

    def get_edge_info(self, subject, predicate, object_id):
        EdgeInfo = collections.namedtuple('EdgeInfo', [
            'subject', 'predicate', 'object', 'id', 'category', 'qualifier',
            'publications', 'sources', 'resource_id', 'resource_role',
            'knowledge_level', 'agent_type', 'stage_qualifier',
            'original_subject', 'original_object'
        ])
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT * FROM EDGE_MAPPING_TABLE WHERE subject = ? AND predicate = ? AND object = ?",
            (subject, predicate, object_id)
        )
        return [EdgeInfo._make(record) for record in cursor.fetchall()]


def main():
    parser = argparse.ArgumentParser(
        description="Build or query the ExplainableDTD database",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--build', action="store_true", help="Build the database from JSONL files", default=False)
    parser.add_argument('--test', action="store_true", help="Run a quick test of the database", default=False)
    parser.add_argument('--nodes_jsonl', type=str, help="Path to nodes.jsonl file")
    parser.add_argument('--edges_jsonl', type=str, help="Path to edges.jsonl file")
    parser.add_argument('--database_name', type=str, default="ExplainableDTD.db", help="Name of the database file")
    parser.add_argument('--outdir', type=str, default=".", help="Path to the output directory")
    parser.add_argument('--log_dir', type=str, default=os.path.join(ROOTPath, "log_folder"), help="The path of logfile folder")
    parser.add_argument('--log_name', type=str, default="build_mapping_db.log", help="Log file name")
    args = parser.parse_args()

    logger = utils.get_logger(os.path.join(args.log_dir, args.log_name))

    if not args.build and not args.test:
        parser.print_help()
        sys.exit(2)

    if args.build:
        if args.nodes_jsonl is None or args.edges_jsonl is None:
            logger.error("--nodes_jsonl and --edges_jsonl are required when using --build")
            sys.exit(1)
        nodes_jsonl = args.nodes_jsonl
        edges_jsonl = args.edges_jsonl
        db = xDTDMappingDB(logger, database_name=args.database_name, outdir=args.outdir, mode='build')
        logger.info("==== Creating tables ====")
        db.create_tables()
        logger.info("==== Populating tables ====")
        db.populate_tables(nodes_jsonl, edges_jsonl)
        logger.info("==== Creating indexes ====")
        db.create_indexes()
        sys.exit(0)

    if args.test:
        db = xDTDMappingDB(logger, database_name=args.database_name, outdir=args.outdir, mode='run', db_loc=args.outdir)
        logger.info("==== Testing node lookup ====")
        logger.info(db.get_node_info(node_id='CHEBI:10'))
        logger.info(db.get_node_info(node_name='Nalidixic acid'))
        logger.info("==== Testing edge lookup ====")
        logger.info(db.get_edge_info(subject='NCBIGene:18993', predicate='biolink:expressed_in', object_id='UBERON:0001016'))


if __name__ == "__main__":
    main()
