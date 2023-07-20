import sqlite3
import json
import biothings_client
import requests
from tqdm import tqdm

class DrugConflator:
    def __init__(self, node_synonymizer_path = "data/node_synonymizer_v1.1_KG2.8.0.1.sqlite", mychem_data_path = "data/mychem_rxcui.json", rxnav_url = "https://rxnav.nlm.nih.gov/REST", normalizer_url = 'https://nodenormalization-sri.renci.org/1.3'):
        """
        This class to identify "essentially the same" drugs based on RXCUI identifiers.
        Parameters
        node_synonymizer_path[str]: path to the node synonymizer database
        rxnav_url[str]: URI of the RXNAV API endpoint
        normalizer_url[str]: URI of the normalizer API endpoint
        """

        self.node_synonymizer_path = node_synonymizer_path
        self.normalizer_url = normalizer_url
        self.rxnav_url = rxnav_url
        # self.mc = biothings_client.get_client("chem")
        self.session = requests.Session()
        
        # ## pre-download mychem data
        # res = self.mc.query("_exists_:unii.rxcui", fields=["unii.rxcui", "chembl.molecule_chembl_id", "umls.cui", "umls.mesh", "drugcentral.xrefs.kegg_drug", "drugbank.id",
        #                                       "unii.ncit", "chebi.id", "drugcentral.xrefs.vandf", "unichem.hmdb", "drugcentral.xrefs.drugcentral", "unii.unii"], fetch_all=True)
        # mychem_rxcui = []
        # for x in res:
        #     mychem_rxcui += [x]
        with open(mychem_data_path, 'r') as f:
            mychem_rxcui = json.load(f)
        
        self.rxcui_dict = {}
        for item in tqdm(mychem_rxcui, desc="Generating rxcui dictionary"):
            if 'unii' in item and 'rxcui' in item['unii']:
                rxcui = item['unii']['rxcui']
                if 'ncit' in item['unii']:
                    self.rxcui_dict.update(self._generate_dict_result(item['unii']['ncit'], "NCIT:", rxcui))
                if 'unii' in item['unii']:
                    self.rxcui_dict.update(self._generate_dict_result(item['unii']['unii'], "UNII:", rxcui))          
            else:
                continue
            
            if 'chebi' in item:
                if isinstance(item['chebi'], dict):
                    self.rxcui_dict.update(self._generate_dict_result(item['chebi']['id'], "", rxcui))
                else:
                    for chebi_item in item['chebi']:
                        self.rxcui_dict.update(self._generate_dict_result(chebi_item['id'], "", rxcui))
            if 'chembl' in item:
                if isinstance(item['chembl'], dict):
                    self.rxcui_dict.update(self._generate_dict_result(item['chembl']['molecule_chembl_id'], "CHEMBL.COMPOUND:", rxcui))
                else:
                    for chembl_item in item['chembl']:
                        self.rxcui_dict.update(self._generate_dict_result(chembl_item['molecule_chembl_id'], "CHEMBL.COMPOUND:", rxcui))
            if 'drugbank' in item:
                if isinstance(item['drugbank'], dict):
                    self.rxcui_dict.update(self._generate_dict_result(item['drugbank']['id'], "DRUGBANK:", rxcui))
                else:
                    for drugbank_item in item['drugbank']:
                        self.rxcui_dict.update(self._generate_dict_result(drugbank_item['id'], "DRUGBANK:", rxcui))
            if 'drugcentral' in item and 'xrefs' in item['drugcentral']:
                if 'kegg_drug' in item['drugcentral']['xrefs']:                    
                    self.rxcui_dict.update(self._generate_dict_result(item['drugcentral']['xrefs']['kegg_drug'], "KEGG.DRUG:", rxcui))
                if 'vandf' in item['drugcentral']['xrefs']:
                    self.rxcui_dict.update(self._generate_dict_result(item['drugcentral']['xrefs']['vandf'], "VANDF:", rxcui))
                if 'drugcentral' in item['drugcentral']['xrefs']:
                    self.rxcui_dict.update(self._generate_dict_result(item['drugcentral']['xrefs']['drugcentral'], "DrugCentral:", rxcui))
            if 'unichem' in item:
                if 'hmdb' in item['unichem']:
                    self.rxcui_dict.update(self._generate_dict_result(item['unichem']['hmdb'], "HMDB:", rxcui))
            if 'umls' in item:
                if 'cui' in item['umls']:
                    self.rxcui_dict.update(self._generate_dict_result(item['umls']['cui'], "UMLS:", rxcui))
                if 'mesh' in item['umls']:
                    self.rxcui_dict.update(self._generate_dict_result(item['umls']['mesh'], "MESH:", rxcui))

    
    @staticmethod
    def _generate_dict_result(res, prefix_name, rxcui):
        temp_dict = {}
        if isinstance(res, str):
            temp_dict[f"{prefix_name}{res}"] = rxcui
        elif isinstance(res, list):
            temp_dict.update({f"{prefix_name}{x}":rxcui for x in res})
            
        return temp_dict
    
    def _get_all_equivalent_info_from_node_normalizer(self, curie):
        """
        This internal function calls the node normalizer and returns the equivalent identifiers and their names
        Parameters
        curie[str]: a curie identifier (e.g. "CHEBI:15365", "RXNORM:1156278")
        
        Returns
        A list of two sublists: [identifiers, Names]
        """

        body = {
                'curies': [
                    curie
                ],
                'conflate': "true"
                }
        headers = {'Content-Type':'application/json'}
        identifiers = []
        labels = []
        try:
            response = self.session.post(url=f"{self.normalizer_url}/get_normalized_nodes", headers=headers, json=body)
        except:
            return []
        if response.status_code == 200:
            json_response = response.json()
            if json_response[curie]:
                for item in json_response[curie]['equivalent_identifiers']:
                    if 'identifier' in item and item['identifier'] and item['identifier'] != '':
                        identifiers.append(item['identifier'])
                    if 'label' in item and item['label'] and item['label'] != '':
                        labels.append(item['label'].lower())
                        
                return [list(set(identifiers)), list(set(labels))]
            else:
                return []
        else:
            return []

    def _get_all_equivalent_info_from_synonymizer(self, curie):
        """
        This internal function calls the node synnoymizer and returns the equivalent identifiers and their names
        Parameters
        curie[str]: a curie identifier (e.g. "CHEBI:15365", "RXNORM:1156278")
        
        Returns
        A list of two sublists: [identifiers, Names]
        """
        ns_con = sqlite3.connect(self.node_synonymizer_path)

        identifiers = []
        labels = []
        ns_cur = ns_con.cursor()                                 
        sql_query_template = f"""
                    SELECT N.id, N.cluster_id, N.name, N.category, C.name
                    FROM nodes as N
                    INNER JOIN clusters as C on C.cluster_id == N.cluster_id
                    WHERE N.id in ('{curie}')"""
        try:
            culster_ids = [x[1] for x in ns_cur.execute(sql_query_template).fetchall()]
        except:
            culster_ids = []
        if len(culster_ids) > 0:
            if len(culster_ids) == 1:
                sql_query_template = f"""
                            SELECT N.id, N.cluster_id, N.name, N.category, C.name
                            FROM nodes as N
                            INNER JOIN clusters as C on C.cluster_id == N.cluster_id
                            WHERE N.cluster_id in ('{culster_ids[0]}')"""
            else:
                sql_query_template = f"""
                            SELECT N.id, N.cluster_id, N.name, N.category, C.name
                            FROM nodes as N
                            INNER JOIN clusters as C on C.cluster_id == N.cluster_id
                            WHERE N.cluster_id in {tuple(culster_ids)}"""
            res = ns_cur.execute(sql_query_template).fetchall()
            for item in res:
                identifiers.append(item[0])
                if item[2] and item[2] != '':
                    labels.append(item[2].lower())
                elif item[4] and item[4] != '':
                    labels.append(item[4].lower())
                else:
                    pass

            return [list(set(identifiers)), list(set(labels))]
        else:
            return []

    @staticmethod
    def _parse_rxcui_json(json_response):
        """
        Parse JSON response from rxnav API
        """
        selected_types = ['IN', 'MIN', 'PIN', 'BN', 'SCDC', 'SBDC', 'SCD', 'GPCK', 'SBD', 'BPCK', 'SCDG', 'SBDG']
        return list(set([y['rxcui'] for x in json_response['allRelatedGroup']['conceptGroup'] if x['tty'] in selected_types and 'conceptProperties' in x  for y in x['conceptProperties']]))

    @staticmethod
    def _compute_drug_similarity(list1, list2, method='mc'):
        """
        This internal function computes the drug similarity
        
        Returns:
        A float score between 0 and 1
        """
        
        def _jaccard_similarity(list1, list2):
            s1 = set(list1)
            s2 = set(list2)
            return len(s1.intersection(s2)) / len(s1.union(s2))
        
        def _max_containment(list1, list2):
            s1 = set(list1)
            s2 = set(list2)
            return len(s1.intersection(s2)) / min(len(s1), len(s2))

        if method == 'mc':
            return _max_containment(list1, list2)
        elif method == 'js':
            return _jaccard_similarity(list1, list2)
        else:
            return None


    def get_rxnorm_from_rxnav(self, curie_list = None, name_list = None):
        """
        This function queries the rxnorm APIs to get the related rxcui ids for a given curie list and a given string name.
        It accepts a list of curies and a list of names as input and returns a list of rxcuis.
        Specifically, it queries the following APIs:
        For curie ids:
            API: https://rxnav.nlm.nih.gov/REST/rxcui.json?idtype=yourIdtype&id=yourId
            For idtype, we only consider the following: ATC, Drugbank, GCN_SEQNO(NDDF), HIC_SEQN(NDDF), MESH, UNII_CODE(UNII), VUID(VANDF)
        For names:
            API: https://rxnav.nlm.nih.gov/REST/approximateTerm?term=value&maxEntries=4
            The 'value' is the name of given drug
        By using these two kinds of APIs, the function will get some rxcui ids. With these key rxcui ids, another API:
            https://rxnav.nlm.nih.gov/REST/rxcui/id/allrelated.json will be called to get more related rxcui ids.
            
            
        Parameters
        curie_list[list]: a list of curie ids (e.g., ['CHEBI:136036','MESH:C026430','CAS:38609-97-1','PUBCHEM.COMPOUND:38072'])
        name_list[list]: a list of curie names (e.g., ['cridanimod', '10-carboxymethyl-9-acridanone', 'cridanimod (inn)'])
        
        Returns
        A list of rxcui ids
        """
        
        rxcui_list = []
        selected_prefixes = ['ATC', 'MESH', 'DRUGBANK', 'NDDF', 'RXNORM', 'UNII', 'VANDF']
        prefix_mapping = {'ATC': 'ATC', 'MESH': 'MESH', 'DRUGBANK': 'Drugbank', 'NDDF': 'GCN_SEQNO|HIC_SEQN', 'UNII': 'UNII_CODE', 'VANDF': 'VUID'}
        if curie_list and len(curie_list) > 0:
            ## filter unrelated curies
            curie_list = [curie for curie in curie_list if curie.split(':')[0] in selected_prefixes]
            if len(curie_list) > 0:
                for curie in curie_list:
                    prefix = curie.split(':')[0]
                    value = curie.split(':')[1]
                    if prefix == 'RXNORM':
                        rxcui_list += [value]
                    else:
                        prefix_list = prefix_mapping[prefix].split('|')
                        for prefix in prefix_list:
                            url = f"{self.rxnav_url}/rxcui.json?idtype={prefix}&id={value}"
                            response = self.session.get(url)
                            if response.status_code == 200:
                                try:
                                    rxcui_list += response.json()['idGroup']['rxnormId']
                                except KeyError:
                                    pass
            else:
                pass
            
        if name_list and len(name_list) > 0:
            for name in name_list:
                url = f"{self.rxnav_url}/approximateTerm.json?term={name}&maxEntries=1"
                response = self.session.get(url)
                if response.status_code == 200:
                    try:
                        rxcui_list += list(set([x['rxcui'] for x in response.json()['approximateGroup']['candidate']]))
                    except KeyError:
                        pass
        
        if len(rxcui_list) > 0:
            final_result = []
            for rxcui in rxcui_list:
                url = f"{self.rxnav_url}/rxcui/{rxcui}/allrelated.json"
                response = self.session.get(url)
                if response.status_code == 200:
                    final_result += self._parse_rxcui_json(response.json())
            return list(set(final_result))
        else:
            return []

    def get_rxnorm_from_mychem(self ,curie_list = None):
        """
        This function calls mychem.info API and queries the unii.rxcui field for a given curie list.
        
        Parameters
        curie_list[list]: a list of curie ids (e.g., ['CHEBI:136036','MESH:C026430','CAS:38609-97-1','PUBCHEM.COMPOUND:38072'])
        
        Returns
        A list of rxcui ids
        """

        rxcui_list = []
        
        # filter unrelated curies
        # selected_prefixes = ['CHEMBL.COMPOUND', 'UMLS', 'KEGG.DRUG', 'DRUGBANK', 'NCIT', 'CHEBI', 'VANDF', 'HMDB', 'DrugCentral', 'UNII']
        # query_template_dict = {
        #     'CHEMBL.COMPOUND': "chembl.molecule_chembl_id:{value}",
        #     'UMLS': "umls.cui:{value}",
        #     'KEGG.DRUG': "drugcentral.xrefs.kegg_drug:{value}",
        #     'DRUGBANK': "drugbank.id:{value}",
        #     'NCIT': "unii.ncit:{value}",
        #     'CHEBI': "chebi.id:{key}\\:{value}",
        #     'VANDF': "drugcentral.xrefs.vandf:{value}",
        #     'HMDB': "unichem.hmdb:{value}i",
        #     'DrugCentral': "drugcentral.xrefs.drugcentral:{value}",
        #     'UNII': "unii.unii:{value}"
        # }
        
        # if curie_list and len(curie_list) > 0:
        #     curie_list = [curie for curie in curie_list if curie.split(':')[0] in selected_prefixes]
        #     for curie in curie_list:
        #         try:
        #             query = query_template_dict[curie.split(':')[0]].format(key=curie.split(':')[0], value=curie.split(':')[1])
        #             # fetch_all=True option returns all hits as an iterator
        #             res = self.mc.query(query, fields="unii.rxcui", size=1, fetch_all=True)
        #             for item in res:
        #                 if isinstance(item['unii'], list):
        #                     rxcui_list += [uni['rxcui'] for uni in item['unii'] if 'rxcui' in uni]
        #                 else:
        #                     try:
        #                         rxcui_list.append(item['unii']['rxcui'])
        #                     except KeyError:
        #                         pass
        #         except:
        #             pass
                            
        #     return list(set(rxcui_list))
        if curie_list and len(curie_list) > 0:
            return list(set([self.rxcui_dict[curie] for curie in curie_list if curie in self.rxcui_dict]))
        else:
            return []
        
    def get_equivalent_curies_and_name(self, curie):
        """
        This function is used to call the node normalizer and node synonymizer to get the equivalent curies and english name based on a given curie
        
        Parameters
        curie[str]: a curie identifier (e.g. "CHEBI:15365", "RXNORM:1156278")
        
        Returns
        A list of two sublists: [identifiers, Names]
        """
        
        identifiers = []
        labels = []
        
        # get equivalent curies and english name from node normalizer
        res_node_normalizer = self._get_all_equivalent_info_from_node_normalizer(curie)
        if len(res_node_normalizer) > 0:
            identifiers += res_node_normalizer[0]
            labels += res_node_normalizer[1]
            
        # get equivalent curies and english name from node synonymizer
        res_synonymizer = self._get_all_equivalent_info_from_synonymizer(curie)
        if len(res_synonymizer) > 0:
            identifiers += res_synonymizer[0]
            labels += res_synonymizer[1]
        
        return [list(set(identifiers)), list(set(labels))]

    def get_rxcui_results(self, curie, use_curie_id = True, use_curie_name = True, use_rxnav = True, use_mychem = True):
        """
        This function calls the 'get_equivalent_curies_and_name' function to get the equivalent curies and names of the given drug
        Following which we query the RxNav database with the identifer and the english name for the rxcui value
        Following which we query mychem.info for the rxcui value
        
        Parameters
        curie[str]: a curie identifier (e.g. "CHEBI:15365", "RXNORM:1156278")
        use_curie_id[bool]: whether to use the curie identifier to query the RxCUI value
        use_curie_name[bool]: whether to use the english name to query the RxCUI value
        use_rxnav[bool]: whether to query the RxNav database
        use_mychem[bool]: whether to query the mychem database
        
        Returns
        A list of rxcui ids
        """
        
        result = []
        ## Get equivalent curies and names
        equivalent_info = self.get_equivalent_curies_and_name(curie)
        
        ## Get rxcui from RxNav
        if use_curie_id and len(equivalent_info[0]) > 0:
            curie_list = equivalent_info[0]
        else:
            curie_list = None
        if use_curie_name and len(equivalent_info[1]) > 0:
            name_list = equivalent_info[1]
        else:
            name_list = None
            
        if use_rxnav:
            result += self.get_rxnorm_from_rxnav(curie_list = curie_list, name_list = name_list)
        
        if use_mychem:
            result += self.get_rxnorm_from_mychem(curie_list = curie_list)
                

        return list(set(result))

    def are_conflated(self, curie1, curie2, use_curie_id = True, use_curie_name = True, use_rxnav = True, use_mychem = True, method = 'mc', threshold = 0.0, return_format = 'score'):
        """
        This function is used to determine whether two given drug curies are essentially the same
        
        Parameters
        curie1[str]: a curie identifier (e.g. "CHEBI:15365")
        curie2[str]: a curie identifier (e.g. "RXNORM:1156278")
        use_curie_id[bool]: whether to use the curie identifier to query the RxCUI value
        use_curie_name[bool]: whether to use the english name to query the RxCUI value
        use_rxnav[bool]: whether to query the RxNav database
        use_mychem[bool]: whether to query the mychem database
        method[str]: the method used to evaluate how close the two drugs are. (Default is 'mc'. Options: 'mc': 'max containment'; 'js': jaccard similarity)
        threshold[float]: the threshold used to determine whether two drugs are conflated. (Default is 0.0)
        return_format[str]: the format of the return value. (Default is 'score'. Options: 'score': return a score; 'boolean': return a boolean value)
        
        Returns
        A score or a boolean value indicating whether the two drugs are conflated
        """
        
        ## check if curie1 is valid
        if not isinstance(curie1, str):
            print(f"Curie1 must be a curie identifier", flush=True)
            return None
        
        ## check if curie2 is valid
        if not isinstance(curie2, str):
            print(f"Curie2 must be a curie identifier", flush=True)
            return None

        ## check if method is valid
        if method not in ['mc', 'js']:
            print(f"Method must be either 'mc' or 'js'", flush=True)
            return None
        
        ## check if return_format is valid
        if return_format not in ['score', 'boolean']:
            print(f"Return format must be either 'score' or 'boolean'", flush=True)
            return None

        curie1_rxcui_list = self.get_rxcui_results(curie1, use_curie_id = use_curie_id, use_curie_name = use_curie_name, use_rxnav = use_rxnav, use_mychem = use_mychem)
        if len(curie1_rxcui_list) == 0:
            print(f"WARNING: Curie1 does not have any rxcui value", flush=True)
            if return_format == 'score':
                return 0.0
            else:
                return False
        
        curie2_rxcui_list = self.get_rxcui_results(curie2, use_curie_id = use_curie_id, use_curie_name = use_curie_name, use_rxnav = use_rxnav, use_mychem = use_mychem)
        if len(curie2_rxcui_list) == 0:
            print(f"WARNING: Curie2 does not have any rxcui value", flush=True)
            if return_format == 'score':
                return 0.0
            else:
                return False
            
        if method == 'mc':
            score = self._compute_drug_similarity(curie1_rxcui_list, curie2_rxcui_list, method='mc')
        else:
            score = self._compute_drug_similarity(curie1_rxcui_list, curie2_rxcui_list, method='js')
            
        if return_format == 'score':
            return score
        else:
            return score >= threshold

if __name__ == "__main__":
    ## Test Examples
    test_curies = ["CHEBI:15365", "RXNORM:1156278"]
    ## Set up drug conflator class
    dc = DrugConflator()
    
    result = [[curie, dc.get_rxcui_results(curie)] for curie in test_curies]
    for item in result:
        print(f"query_curie: {item[0]}, rxcui: {item[1]}", flush=True)

    ## A few examples to test the conflator
    dc.are_conflated("CHEBI:15365", "RXNORM:1156278")
    dc.are_conflated("CHEMBL.COMPOUND:CHEMBL25", "CHEBI:15365", method = 'js')
    dc.are_conflated("CHEMBL.COMPOUND:CHEMBL25", "CHEBI:15365", method = 'js', threshold = 0.5, return_format='boolean')
    dc.are_conflated("CHEMBL.COMPOUND:CHEMBL25", "CHEBI:15365", method = 'mc')
    dc.are_conflated("CHEMBL.COMPOUND:CHEMBL25", "CHEBI:15365", method = 'mc', threshold = 0.5, return_format='boolean')