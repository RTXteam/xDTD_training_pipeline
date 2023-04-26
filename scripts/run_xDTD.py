import os, sys
path_list = os.path.abspath(__file__).split('/')
index = path_list.index('creative_DTD_endpoint')
script_path = '/'.join(path_list[:(index+1)] + ['scripts'])
sys.path.append(script_path)
from creativeDTD import creativeDTD
import pandas as pd
import argparse
data_path = '/'.join(path_list[:(index+1)] + ['data'])
model_path = '/'.join(path_list[:(index+1)] + ['models'])
from tqdm import tqdm

## load packages
import os, sys

## Import Personal Packages
pathlist = os.getcwd().split(os.path.sep)
ROOTindex = pathlist.index("xDTD_training_pipeline")
ROOTPath = os.path.sep.join([*pathlist[:(ROOTindex + 1)]])
sys.path.append(os.path.join(ROOTPath, 'scripts'))
import utils

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='Run Model')
    parser.add_argument("--log_dir", type=str, help="The path of logfile folder", default=os.path.join(ROOTPath, "log_folder"))
    parser.add_argument("--log_name", type=str, help="log file name", default="run_xDTD.log")
	parser.add_argument('--disease_set', type=str, required=True, help='Path to disase set')
	parser.add_argument('--out_dir', type=str, required=True, help='Output Directory')
	parser.add_argument('--use_gpu', action='store_true', default=False)
	parser.add_argument('--gpu', type=int, required=False, default=0, help='Which gpu to use')
	parser.add_argument('--N_drugs', type=int, required=False, default=25, help='Number of predicted drugs')
	parser.add_argument('--N_paths', type=int, required=False, default=25, help='Number of predicted paths')
	parser.add_argument('--batch_size', type=int, required=False, default=200, help='Batch size')
	parser.add_argument('--data_path', type=str, required=False, default=os.path.join(ROOTPath, 'data'), help='Path to data folder')
	parser.add_argument('--model_path', type=str, required=False, default=os.path.join(ROOTPath, 'models'), help='Path to model folder')
	
	## knowledge graph and environment parameters
    parser.add_argument('--entity_dim', type=int, help='Dimension of entity embedding', default=100)
    parser.add_argument('--relation_dim', type=int, help='Dimension of relation embedding', default=100)
    parser.add_argument('--entity_type_dim', type=int, help='Dimension of entity type embedding', default=100)
    parser.add_argument('--max_path', type=int, help='Maximum length of path', default=3)
    parser.add_argument('--bandwidth', type=int, help='Maximum number of neighbors', default=3000)
    parser.add_argument('--bucket_interval', type=int, help='adjacency list bucket size to save memory (default: 50)', default=50)
    parser.add_argument('--state_history', type=int, help='state history length', default=1)
    parser.add_argument("--emb_dropout_rate", type=float, help="Knowledge entity and relation embedding vector dropout rate (default: 0)", default=0)

    # discriminator parameters
    parser.add_argument('--disc_hidden', type=int, nargs='*', help='Path discriminator hidden dim parameter', default=[512, 512])
    parser.add_argument('--disc_dropout_rate', type=float, help='Path discriminator dropout rate', default=0.3)

    # metadiscriminator parameters
    parser.add_argument('--metadisc_hidden', type=int, nargs='*', help='Meta discriminator hidden dim parameters', default=[512, 256])
    parser.add_argument('--metadisc_dropout_rate', type=float, help='Meta discriminator dropout rate', default=0.3)

    # AC model parameters
    parser.add_argument('--ac_hidden', type=int, nargs='*', help='ActorCritic hidden dim parameters', default=[512, 512])
    parser.add_argument('--actor_dropout_rate', type=float, help='actor dropout rate', default=0.3)
    parser.add_argument('--critic_dropout_rate', type=float, help='critic dropout rate', default=0.3)
    parser.add_argument('--act_dropout', type=float, help='action dropout rate', default=0.3)
    parser.add_argument('--gamma', type=float, help='reward discount factor', default=0.99)
    parser.add_argument('--target_update', type=float, help='update ratio of target network', default=0.05)
 
	main_args = parser.parse_args()

    logger = utils.get_logger(os.path.join(main_args.log_dir,main_args.log_name))
    logger.info(args)
    args.logger = logger

	output_path = main_args.out_dir

	## set up output folder
	if not os.path.exists(main_args.out_dir):
		os.makedirs(main_args.out_dir)
	if not os.path.exists(os.path.join(output_path, 'prediction_scores')):
		os.makedirs(os.path.join(output_path, 'prediction_scores'))
	if not os.path.exists(os.path.join(output_path, 'path_results')):
		os.makedirs(os.path.join(output_path, 'path_results'))

	## create xDTD object
	dtd = creativeDTD(args, data_path, model_path)

	## get disease info
	disease_set = pd.read_csv(main_args.disease_set, sep='\t', header=0)
	disease_list = list(disease_set['id'])
	existing_disease_list = [os.path.splitext(file_name)[0].replace('top25drugs_','') for file_name in os.listdir(os.path.join(output_path, 'prediction_scores'))]
	disease_list = list(set(disease_list).difference(set(existing_disease_list)))

	for index, disease in enumerate(tqdm(disease_list)):
		## set up query disease curie
		has_disease = dtd.set_query_disease(disease)
		if has_disease:
			## predict top N drugs for query disease
			predicted_drugs = dtd.predict_top_N_drugs(main_args.N_drugs)
			## save topN predictions as txt file
			predicted_drugs.to_csv(os.path.join(output_path, 'prediction_scores', f'top{main_args.N_drugs}drugs_{disease_list[index]}.txt'), sep='\t', index=None)
			## predict top M paths for N drugs with query disease pairs
			predicted_paths = dtd.predict_top_M_paths(main_args.N_paths)
			path_table = []
			for pair in predicted_paths:
				for path in predicted_paths[pair]:
					try:
						path_table += [[pair[0], pair[1], path[0], path[1]]]
					except:
						continue
			path_table = pd.DataFrame(path_table, columns = ['drug_id', 'disease_id', 'path', 'path_score'])
			path_table = path_table.merge(predicted_drugs[['drug_id','drug_name','disease_id','disease_name']], left_on=['drug_id','disease_id'], right_on=['drug_id','disease_id'], how='left')
			path_table = path_table[['drug_id', 'drug_name', 'disease_id', 'disease_name', 'path', 'path_score']].reset_index(drop=True)
			if len(path_table) > 0:
				path_table.to_csv(os.path.join(output_path, 'path_results', f'top{main_args.N_paths}paths_{disease_list[index]}.txt'), sep='\t', index=None)
