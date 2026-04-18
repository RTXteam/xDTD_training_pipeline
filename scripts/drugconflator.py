import json
import requests
from tqdm import tqdm


# Each spec: (path_of_keys_to_parent, field_name_inside_parent, curie_prefix, parent_may_be_list_of_dicts)
_FIELD_SPECS = [
    (['unii'], 'ncit', 'NCIT:', False),
    (['unii'], 'unii', 'UNII:', False),
    (['chebi'], 'id', '', True),
    (['chembl'], 'molecule_chembl_id', 'CHEMBL.COMPOUND:', True),
    (['drugbank'], 'id', 'DRUGBANK:', True),
    (['drugcentral', 'xrefs'], 'kegg_drug', 'KEGG.DRUG:', False),
    (['drugcentral', 'xrefs'], 'vandf', 'VANDF:', False),
    (['drugcentral', 'xrefs'], 'drugcentral', 'DrugCentral:', False),
    (['unichem'], 'hmdb', 'HMDB:', False),
    (['umls'], 'cui', 'UMLS:', False),
    (['umls'], 'mesh', 'MESH:', False),
]


class DrugConflator:
    def __init__(self, mychem_data_path="data/mychem_rxcui.json",
                 rxnav_url="https://rxnav.nlm.nih.gov/REST",
                 normalizer_url='https://nodenormalization-sri.renci.org/1.5'):
        """
        Identify "essentially the same" drugs based on RXCUI identifiers.

        Parameters
        ----------
        mychem_data_path : str
            Path to the pre-downloaded mychem RXCUI JSON file.
        rxnav_url : str
            URI of the RxNav API endpoint.
        normalizer_url : str
            URI of the Node Normalizer API endpoint.
        """
        self.normalizer_url = normalizer_url
        self.rxnav_url = rxnav_url
        self.session = requests.Session()

        with open(mychem_data_path, 'r') as f:
            mychem_rxcui = json.load(f)

        self.rxcui_dict = {}
        for item in tqdm(mychem_rxcui, desc="Generating rxcui dictionary"):
            unii = item.get('unii')
            if not isinstance(unii, dict) or 'rxcui' not in unii:
                continue
            rxcui = unii['rxcui']

            for keys, field, prefix, may_be_list in _FIELD_SPECS:
                parent = item
                for k in keys:
                    parent = parent.get(k) if isinstance(parent, dict) else None
                    if parent is None:
                        break
                if parent is None:
                    continue

                if may_be_list and isinstance(parent, list):
                    for sub in parent:
                        if field in sub:
                            self._add_to_rxcui_dict(sub[field], prefix, rxcui)
                elif isinstance(parent, dict) and field in parent:
                    self._add_to_rxcui_dict(parent[field], prefix, rxcui)

    def _add_to_rxcui_dict(self, value, prefix, rxcui):
        """Insert one or more prefixed curie→rxcui mappings into self.rxcui_dict."""
        if isinstance(value, str):
            self.rxcui_dict[f"{prefix}{value}"] = rxcui
        elif isinstance(value, list):
            for v in value:
                self.rxcui_dict[f"{prefix}{v}"] = rxcui

    def _get_all_equivalent_info_from_node_normalizer(self, curie):
        """
        Call the Node Normalizer and return equivalent identifiers and labels.

        Returns
        -------
        list
            ``[identifiers, labels]`` — two deduplicated lists, or ``[]`` on failure.
        """
        body = {'curies': [curie], 'conflate': "true"}
        try:
            response = self.session.post(
                url=f"{self.normalizer_url}/get_normalized_nodes",
                headers={'Content-Type': 'application/json'},
                json=body,
            )
        except requests.RequestException:
            return []

        if response.status_code != 200:
            return []
        entry = response.json().get(curie)
        if not entry:
            return []

        identifiers = set()
        labels = set()
        for item in entry.get('equivalent_identifiers', []):
            ident = item.get('identifier')
            if ident:
                identifiers.add(ident)
            label = item.get('label')
            if label:
                labels.add(label.lower())
        return [list(identifiers), list(labels)]

    @staticmethod
    def _parse_rxcui_json(json_response):
        """Parse JSON response from RxNav ``allrelated`` endpoint."""
        selected_types = {'IN', 'MIN', 'PIN', 'BN', 'SCDC', 'SBDC',
                          'SCD', 'GPCK', 'SBD', 'BPCK', 'SCDG', 'SBDG'}
        return list({
            y['rxcui']
            for x in json_response['allRelatedGroup']['conceptGroup']
            if x['tty'] in selected_types and 'conceptProperties' in x
            for y in x['conceptProperties']
        })

    @staticmethod
    def _compute_drug_similarity(list1, list2, method='mc'):
        """
        Compute similarity between two RXCUI lists.

        Returns
        -------
        float or None
            Score between 0 and 1, or ``None`` if *method* is unknown.
        """
        s1, s2 = set(list1), set(list2)
        overlap = len(s1 & s2)
        if method == 'mc':
            return overlap / min(len(s1), len(s2))
        if method == 'js':
            return overlap / len(s1 | s2)
        return None

    def get_rxnorm_from_rxnav(self, curie_list=None, name_list=None):
        """
        Query RxNav APIs to obtain related RXCUI ids for curie identifiers and/or names.

        Parameters
        ----------
        curie_list : list[str] or None
            CURIE identifiers (e.g. ``['MESH:C026430', 'DRUGBANK:DB00945']``).
        name_list : list[str] or None
            Drug names (e.g. ``['aspirin']``).

        Returns
        -------
        list[str]
            Deduplicated RXCUI ids.
        """
        rxcui_list = []
        prefix_mapping = {
            'ATC': 'ATC', 'MESH': 'MESH', 'DRUGBANK': 'Drugbank',
            'NDDF': 'GCN_SEQNO|HIC_SEQN', 'UNII': 'UNII_CODE', 'VANDF': 'VUID',
        }

        if curie_list:
            for curie in curie_list:
                prefix, _, value = curie.partition(':')
                if prefix == 'RXNORM':
                    rxcui_list.append(value)
                elif prefix in prefix_mapping:
                    for id_type in prefix_mapping[prefix].split('|'):
                        url = f"{self.rxnav_url}/rxcui.json?idtype={id_type}&id={value}"
                        response = self.session.get(url)
                        if response.status_code == 200:
                            try:
                                rxcui_list += response.json()['idGroup']['rxnormId']
                            except KeyError:
                                pass

        if name_list:
            for name in name_list:
                url = f"{self.rxnav_url}/approximateTerm.json?term={name}&maxEntries=1"
                response = self.session.get(url)
                if response.status_code == 200:
                    try:
                        rxcui_list += [x['rxcui'] for x in response.json()['approximateGroup']['candidate']]
                    except KeyError:
                        pass

        if not rxcui_list:
            return []

        final_result = set()
        for rxcui in set(rxcui_list):
            url = f"{self.rxnav_url}/rxcui/{rxcui}/allrelated.json"
            response = self.session.get(url)
            if response.status_code == 200:
                final_result.update(self._parse_rxcui_json(response.json()))
        return list(final_result)

    def get_rxnorm_from_mychem(self, curie_list=None):
        """
        Look up RXCUI values from the pre-loaded mychem dictionary.

        Parameters
        ----------
        curie_list : list[str] or None
            CURIE identifiers.

        Returns
        -------
        list[str]
            Deduplicated RXCUI ids.
        """
        if not curie_list:
            return []
        return list({self.rxcui_dict[c] for c in curie_list if c in self.rxcui_dict})

    def get_equivalent_curies_and_name(self, curie):
        """
        Get equivalent CURIE identifiers and English names via the Node Normalizer.

        Returns
        -------
        list
            ``[identifiers, names]`` — two lists (may be empty).
        """
        return self._get_all_equivalent_info_from_node_normalizer(curie) or [[], []]

    def get_rxcui_results(self, curie, use_curie_id=True, use_curie_name=True,
                          use_rxnav=True, use_mychem=True):
        """
        Obtain all related RXCUI ids for a drug curie by combining Node Normalizer,
        RxNav, and mychem lookups.

        Returns
        -------
        list[str]
            Deduplicated RXCUI ids.
        """
        equivalent_info = self.get_equivalent_curies_and_name(curie)
        curie_list = equivalent_info[0] if use_curie_id and equivalent_info[0] else None
        name_list = equivalent_info[1] if use_curie_name and equivalent_info[1] else None

        result = []
        if use_rxnav:
            result += self.get_rxnorm_from_rxnav(curie_list=curie_list, name_list=name_list)
        if use_mychem:
            result += self.get_rxnorm_from_mychem(curie_list=curie_list)
        return list(set(result))

    def are_conflated(self, curie1, curie2, use_curie_id=True, use_curie_name=True,
                      use_rxnav=True, use_mychem=True, method='mc', threshold=0.0,
                      return_format='score'):
        """
        Determine whether two drug CURIEs are essentially the same drug.

        Parameters
        ----------
        curie1, curie2 : str
            Drug CURIE identifiers.
        method : str
            ``'mc'`` (max containment) or ``'js'`` (Jaccard similarity).
        threshold : float
            Score threshold for the boolean check.
        return_format : str
            ``'score'`` returns the similarity float; ``'boolean'`` returns True/False.

        Returns
        -------
        float or bool or None
            Similarity score, boolean, or ``None`` on invalid input.
        """
        if not isinstance(curie1, str) or not isinstance(curie2, str):
            print("Both curie1 and curie2 must be curie identifiers", flush=True)
            return None
        if method not in ('mc', 'js'):
            print("Method must be either 'mc' or 'js'", flush=True)
            return None
        if return_format not in ('score', 'boolean'):
            print("Return format must be either 'score' or 'boolean'", flush=True)
            return None

        kwargs = dict(use_curie_id=use_curie_id, use_curie_name=use_curie_name,
                      use_rxnav=use_rxnav, use_mychem=use_mychem)
        curie1_rxcui = self.get_rxcui_results(curie1, **kwargs)
        if not curie1_rxcui:
            print("WARNING: Curie1 does not have any rxcui value", flush=True)
            return 0.0 if return_format == 'score' else False

        curie2_rxcui = self.get_rxcui_results(curie2, **kwargs)
        if not curie2_rxcui:
            print("WARNING: Curie2 does not have any rxcui value", flush=True)
            return 0.0 if return_format == 'score' else False

        score = self._compute_drug_similarity(curie1_rxcui, curie2_rxcui, method=method)
        return score if return_format == 'score' else score >= threshold


if __name__ == "__main__":
    ## ── Test Examples ───────────────────────────────────────────────────────────────
    test_curies = ["CHEBI:15365", "RXNORM:1156278"]
    ## ── Initialize DrugConflator ───────────────────────────────────────────────────────
    dc = DrugConflator()

    for curie in test_curies:
        print(f"query_curie: {curie}, rxcui: {dc.get_rxcui_results(curie)}", flush=True)

    ## ── Test Conflation ───────────────────────────────────────────────────────────────
    dc.are_conflated("CHEBI:15365", "RXNORM:1156278")
    dc.are_conflated("CHEMBL.COMPOUND:CHEMBL25", "CHEBI:15365", method='js')
    dc.are_conflated("CHEMBL.COMPOUND:CHEMBL25", "CHEBI:15365", method='js', threshold=0.5, return_format='boolean')
    dc.are_conflated("CHEMBL.COMPOUND:CHEMBL25", "CHEBI:15365", method='mc')
    dc.are_conflated("CHEMBL.COMPOUND:CHEMBL25", "CHEBI:15365", method='mc', threshold=0.5, return_format='boolean')
