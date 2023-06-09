import ast
import json
import os
import pathlib
import sqlite3
import sys
from collections import defaultdict
from typing import Optional, Union, List, Set

pathlist = os.getcwd().split(os.path.sep)
ROOTindex = pathlist.index("xDTD_training_pipeline")
ROOTPath = os.path.sep.join([*pathlist[:(ROOTindex + 1)]])

class NodeSynonymizer:

    def __init__(self):
        
        ## Load the contents of two config files (e.g., config_dbs.json and config_secrets.json)
        with open(os.path.join(ROOTPath, 'config_dbs.json'), 'rb') as file_in:
            config_dbs = json.load(file_in)
            node_synonymizer_path = config_dbs["database_downloads"]["node_synonymizer"]
            self.database_name = node_synonymizer_path.split('/')[-1]
        
        synonymizer_dir = os.path.join(ROOTPath, 'data')
        self.database_path = f"{synonymizer_dir}/{self.database_name}"
        self.placeholder_lookup_values_str = "**LOOKUP_VALUES_GO_HERE**"

        if not pathlib.Path(self.database_path).exists():
            raise ValueError(f"Synonymizer specified in config_dbs file does not exist locally, even after "
                             f"running the database manager! It should be at: {self.database_path}")
        else:
            self.db_connection = sqlite3.connect(self.database_path)

    def __del__(self):
        if hasattr(self, "db_connection"):
            self.db_connection.close()

    # --------------------------------------- EXTERNAL MAIN METHODS ----------------------------------------------- #

    def get_canonical_curies(self, curies: Optional[Union[str, Set[str], List[str]]] = None,
                             names: Optional[Union[str, Set[str], List[str]]] = None,
                             return_all_categories: bool = False) -> dict:

        # Convert any input values to Set format
        curies_set = self._convert_to_set_format(curies)
        names_set = self._convert_to_set_format(names)
        results_dict = dict()

        if curies_set:
            # Query the synonymizer sqlite database for these identifiers
            sql_query_template = f"""
                        SELECT N.id, N.cluster_id, C.name, C.category
                        FROM nodes as N
                        INNER JOIN clusters as C on C.cluster_id == N.cluster_id
                        WHERE N.id in ('{self.placeholder_lookup_values_str}')"""
            matching_rows = self._run_sql_query_in_batches(sql_query_template, curies_set)

            # Transform the results into the proper response format
            results_dict = {row[0]: self._create_preferred_node_dict(preferred_id=row[1],
                                                                     preferred_category=row[3],
                                                                     preferred_name=row[2])
                            for row in matching_rows}

        if names_set:
            # Query the synonymizer sqlite database for these names
            sql_query_template = f"""
                        SELECT N.id, N.name, N.cluster_id, C.name, C.category
                        FROM nodes as N
                        INNER JOIN clusters as C on C.cluster_id == N.cluster_id
                        WHERE N.name in ('{self.placeholder_lookup_values_str}')"""
            matching_rows = self._run_sql_query_in_batches(sql_query_template, names_set)

            # For each input name, pick the cluster that nodes with that exact name most often belong to
            names_to_best_cluster_id = self._count_clusters_per_name(matching_rows, name_index=1, cluster_id_index=2)

            # Create some helper maps
            cluster_ids_to_node_id = {row[2]: row[0] for row in matching_rows}  # Doesn't matter that this gives ONE node per cluster
            node_ids_to_rows = {row[0]: row for row in matching_rows}
            names_to_cluster_rows = {name: node_ids_to_rows[cluster_ids_to_node_id[cluster_id]]
                                     for name, cluster_id in names_to_best_cluster_id.items()}

            # Transform the results into the proper response format
            results_dict_names = {name: self._create_preferred_node_dict(preferred_id=cluster_id,
                                                                         preferred_category=names_to_cluster_rows[name][4],
                                                                         preferred_name=names_to_cluster_rows[name][3])
                                  for name, cluster_id in names_to_best_cluster_id.items()}

            # Merge these results with any results for input curies
            results_dict.update(results_dict_names)

        # Tack on all categories, if asked for (infrequent enough that it's ok to have an extra query for this)
        if return_all_categories:
            cluster_ids = {canonical_info["preferred_curie"]
                           for canonical_info in results_dict.values()}
            sql_query_template = f"""
                        SELECT N.cluster_id, N.category
                        FROM nodes as N
                        WHERE N.cluster_id in ('{self.placeholder_lookup_values_str}')"""
            matching_rows = self._run_sql_query_in_batches(sql_query_template, cluster_ids)

            # Count up how many members this cluster has with different categories
            clusters_by_category_counts = defaultdict(lambda: defaultdict(int))
            for cluster_id, member_category in matching_rows:
                member_category = self._add_biolink_prefix(member_category)
                clusters_by_category_counts[cluster_id][member_category] += 1

            # Add the counts to our response
            for canonical_info in results_dict.values():
                cluster_id = canonical_info["preferred_curie"]
                category_counts = clusters_by_category_counts[cluster_id]
                canonical_info["all_categories"] = dict(category_counts)

        # Add None values for any unrecognized input values
        unrecognized_input_values = (curies_set.union(names_set)).difference(results_dict)
        for unrecognized_value in unrecognized_input_values:
            results_dict[unrecognized_value] = None

        return results_dict

    def get_equivalent_nodes(self, curies: Optional[Union[str, Set[str], List[str]]] = None,
                             names: Optional[Union[str, Set[str], List[str]]] = None,
                             include_unrecognized_entities: bool = True) -> dict:

        # Convert any input values to Set format
        curies_set = self._convert_to_set_format(curies)
        names_set = self._convert_to_set_format(names)
        results_dict = dict()

        if curies_set:
            # Query the synonymizer sqlite database for these identifiers (in batches, if necessary)
            sql_query_template = f"""
                        SELECT N.id, C.member_ids
                        FROM nodes as N
                        INNER JOIN clusters as C on C.cluster_id == N.cluster_id
                        WHERE N.id in ('{self.placeholder_lookup_values_str}')"""
            matching_rows = self._run_sql_query_in_batches(sql_query_template, curies_set)

            # Transform the results into the proper response format
            results_dict = {row[0]: ast.literal_eval(row[1]) for row in matching_rows}

        if names_set:
            # Query the synonymizer sqlite database for these names
            sql_query_template = f"""
                        SELECT N.id, N.name, C.cluster_id, C.member_ids
                        FROM nodes as N
                        INNER JOIN clusters as C on C.cluster_id == N.cluster_id
                        WHERE N.name in ('{self.placeholder_lookup_values_str}')"""
            matching_rows = self._run_sql_query_in_batches(sql_query_template, names_set)

            # For each input name, pick the cluster that nodes with that exact name most often belong to
            names_to_best_cluster_id = self._count_clusters_per_name(matching_rows, name_index=1, cluster_id_index=2)

            # Create some helper maps
            cluster_ids_to_node_id = {row[2]: row[0] for row in matching_rows}  # Doesn't matter that this gives ONE node per cluster
            node_ids_to_rows = {row[0]: row for row in matching_rows}
            names_to_cluster_rows = {name: node_ids_to_rows[cluster_ids_to_node_id[cluster_id]]
                                     for name, cluster_id in names_to_best_cluster_id.items()}

            # Transform the results into the proper response format
            results_dict_names = {name: ast.literal_eval(cluster_row[3])
                                  for name, cluster_row in names_to_cluster_rows.items()}

            # Merge these results with any results for input curies
            results_dict.update(results_dict_names)

        if include_unrecognized_entities:
            # Add None values for any unrecognized input curies
            unrecognized_curies = (curies_set.union(names_set)).difference(results_dict)
            for unrecognized_curie in unrecognized_curies:
                results_dict[unrecognized_curie] = None

        return results_dict

    def get_normalizer_results(self, entities: Optional[Union[str, Set[str], List[str]]]) -> dict:

        # Convert any input curies to Set format
        entities_set = self._convert_to_set_format(entities)

        # First get all equivalent IDs for each input curie
        equivalent_curies_dict = self.get_equivalent_nodes(curies=entities_set, include_unrecognized_entities=False)
        unrecognized_entities = entities_set.difference(equivalent_curies_dict)
        # If we weren't successful at looking up some entities as curies, try looking them up as names
        if unrecognized_entities:
            equivalent_curies_dict_names = self.get_equivalent_nodes(names=unrecognized_entities, include_unrecognized_entities=False)
            equivalent_curies_dict.update(equivalent_curies_dict_names)

        # Then get info for all of those equivalent nodes
        all_node_ids = set().union(*equivalent_curies_dict.values())
        sql_query_template = f"""
                    SELECT N.id, N.cluster_id, N.name, N.category, N.major_branch, N.name_sri, N.category_sri, N.name_kg2pre, N.category_kg2pre, C.name
                    FROM nodes as N
                    INNER JOIN clusters as C on C.cluster_id == N.cluster_id
                    WHERE N.id in ('{self.placeholder_lookup_values_str}')"""
        matching_rows = self._run_sql_query_in_batches(sql_query_template, all_node_ids)
        nodes_dict = {row[0]: {"identifier": row[0],
                               "category": self._add_biolink_prefix(row[3]),
                               "label": row[2],
                               "major_branch": row[4],
                               "in_sri": row[6] is not None,
                               "name_sri": row[5],
                               "category_sri": self._add_biolink_prefix(row[6]),
                               "in_kg2pre": row[8] is not None,
                               "name_kg2pre": row[7],
                               "category_kg2pre": self._add_biolink_prefix(row[8]),
                               "cluster_id": row[1],
                               "cluster_preferred_name": row[9]} for row in matching_rows}

        # Transform the results into the proper response format
        results_dict = dict()
        for input_entity, equivalent_curies in equivalent_curies_dict.items():
            cluster_id = nodes_dict[next(iter(equivalent_curies))]["cluster_id"]  # All should have the same cluster ID
            cluster_rep = nodes_dict[cluster_id]
            results_dict[input_entity] = {"id": {"identifier": cluster_id,
                                                 "name": cluster_rep["cluster_preferred_name"],
                                                 "category": cluster_rep["category"],
                                                 "SRI_normalizer_name": cluster_rep["name_sri"],
                                                 "SRI_normalizer_category": cluster_rep["category_sri"],
                                                 "SRI_normalizer_curie": cluster_id if cluster_rep["category_sri"] else None},
                                          "categories": defaultdict(int),
                                          "nodes": [nodes_dict[equivalent_curie] for equivalent_curie in equivalent_curies]}

        # Do some post-processing (tally up category counts and remove no-longer-needed 'cluster_id' property)
        for concept_info_dict in results_dict.values():
            for equivalent_node in concept_info_dict["nodes"]:
                concept_info_dict["categories"][equivalent_node["category"]] += 1
                if "cluster_id" in equivalent_node:
                    del equivalent_node["cluster_id"]
                if "cluster_preferred_name" in equivalent_node:
                    del equivalent_node["cluster_preferred_name"]

        # Add None values for any unrecognized input curies
        unrecognized_curies = entities_set.difference(results_dict)
        for unrecognized_curie in unrecognized_curies:
            results_dict[unrecognized_curie] = None

        return results_dict

    # ---------------------------------------- INTERNAL HELPER METHODS -------------------------------------------- #

    @staticmethod
    def _convert_to_str_format(list_or_set: Union[Set[str], List[str]]) -> str:
        preprocessed_list = [item.replace("'", "''") for item in list_or_set if item]  # Need to escape ' characters for SQL
        list_str = "','".join(preprocessed_list)  # SQL wants ('id1', 'id2') format for string lists
        return list_str

    @staticmethod
    def _convert_to_set_format(some_value: Optional[Union[str, Set[str], List[str]]]) -> set:
        if some_value:
            if isinstance(some_value, set):
                return some_value
            elif isinstance(some_value, list):
                return set(some_value)
            elif isinstance(some_value, str):
                return {some_value}
            else:
                raise ValueError(f"Input is not an allowable data type (list, set, or string)!")
        else:
            return set()

    @staticmethod
    def _add_biolink_prefix(category: Optional[str]) -> Optional[str]:
        if category:
            return f"biolink:{category}"
        else:
            return category

    @staticmethod
    def _count_clusters_per_name(rows: list, name_index: int, cluster_id_index: int) -> dict:
        names_to_cluster_counts = defaultdict(lambda: defaultdict(int))
        for row in rows:
            name = row[name_index]
            cluster_id = row[cluster_id_index]
            names_to_cluster_counts[name][cluster_id] += 1
        names_to_best_cluster_id = {name: max(cluster_counts, key=cluster_counts.get)
                                    for name, cluster_counts in names_to_cluster_counts.items()}
        return names_to_best_cluster_id

    @staticmethod
    def _divide_into_chunks(some_set: Set[str], chunk_size: int) -> List[List[str]]:
        some_list = list(some_set)
        return [some_list[start:start + chunk_size] for start in range(0, len(some_list), chunk_size)]

    def _create_preferred_node_dict(self, preferred_id: str, preferred_category: str, preferred_name: Optional[str]) -> dict:
        return {
            "preferred_curie": preferred_id,
            "preferred_name": preferred_name,
            "preferred_category": self._add_biolink_prefix(preferred_category)
        }

    def _run_sql_query_in_batches(self, sql_query_template: str, lookup_values: Set[str]) -> list:
        """
        Sqlite has a max length allowed for SQL statements, so we divide really long curie/name lists into batches.
        """
        lookup_values_batches = self._divide_into_chunks(lookup_values, 5000)
        all_matching_rows = []
        for lookup_values_batch in lookup_values_batches:
            sql_query = sql_query_template.replace(self.placeholder_lookup_values_str,
                                                   self._convert_to_str_format(lookup_values_batch))
            matching_rows = self._execute_sql_query(sql_query)
            all_matching_rows += matching_rows
        return all_matching_rows

    def _execute_sql_query(self, sql_query: str) -> list:
        cursor = self.db_connection.cursor()
        cursor.execute(sql_query)
        matching_rows = cursor.fetchall()
        cursor.close()
        return matching_rows


def main():
    test_curies = ["DOID:14330", "MONDO:0005180", "CHEMBL.COMPOUND:CHEMBL112", "UNICORN"]
    test_names = ["Acetaminophen", "Unicorn", "ACETAMINOPHEN", "Parkinson disease"]
    synonymizer = NodeSynonymizer()
    results = synonymizer.get_canonical_curies(test_curies)
    results = synonymizer.get_equivalent_nodes(test_curies)
    results = synonymizer.get_normalizer_results(test_curies)
    results = synonymizer.get_normalizer_results(test_curies + test_names)
    results = synonymizer.get_canonical_curies(names=test_names)
    results = synonymizer.get_equivalent_nodes(names=test_names)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
