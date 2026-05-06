"""Run xDTD predictions """

import argparse
import os
import sys

import polars as pl
from tqdm import tqdm

pathlist = os.getcwd().split(os.path.sep)
ROOTindex = pathlist.index("xDTD_training_pipeline")
ROOTPath = os.path.sep.join([*pathlist[:(ROOTindex + 1)]])
sys.path.append(os.path.join(ROOTPath, 'scripts'))
import utils
from creativeDTD import creativeDTD


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run Model')
    parser.add_argument("--log_dir", type=str, help="The path of logfile folder", default=os.path.join(ROOTPath, "log_folder"))
    parser.add_argument("--log_name", type=str, help="log file name", default="step23_run_xDTD.log")
    parser.add_argument('--disease_set', type=str, required=True, help='Path to disease set')
    parser.add_argument('--out_dir', type=str, required=True, help='Output Directory')
    parser.add_argument('--use_gpu', action='store_true', default=False)
    parser.add_argument('--gpu', type=int, default=0, help='Which gpu to use')
    parser.add_argument('--N_drugs', type=int, default=25, help='Number of predicted drugs')
    parser.add_argument('--N_paths', type=int, default=25, help='Number of predicted paths')
    parser.add_argument('--batch_size', type=int, default=200, help='Batch size')
    parser.add_argument('--data_path', type=str, default=os.path.join(ROOTPath, 'data'), help='Path to data folder')
    parser.add_argument('--model_path', type=str, default=os.path.join(ROOTPath, 'models'), help='Path to model folder')

    ## Knowledge graph and environment parameters
    parser.add_argument('--entity_dim', type=int, default=100, help='Dimension of entity embedding')
    parser.add_argument('--relation_dim', type=int, default=100, help='Dimension of relation embedding')
    parser.add_argument('--entity_type_dim', type=int, default=100, help='Dimension of entity type embedding')
    parser.add_argument('--max_path', type=int, default=3, help='Maximum length of path')
    parser.add_argument('--bandwidth', type=int, default=3000, help='Maximum number of neighbors')
    parser.add_argument('--bucket_interval', type=int, default=50, help='Adjacency list bucket size')
    parser.add_argument('--state_history', type=int, default=1, help='State history length')
    parser.add_argument("--emb_dropout_rate", type=float, default=0, help="Entity/relation embedding dropout rate")

    ## Discriminator / actor-critic parameters
    parser.add_argument('--disc_hidden', type=int, nargs='*', default=[512, 512], help='Path discriminator hidden dims')
    parser.add_argument('--disc_dropout_rate', type=float, default=0.3, help='Path discriminator dropout rate')
    parser.add_argument('--metadisc_hidden', type=int, nargs='*', default=[512, 256], help='Meta discriminator hidden dims')
    parser.add_argument('--metadisc_dropout_rate', type=float, default=0.3, help='Meta discriminator dropout rate')
    parser.add_argument('--ac_hidden', type=int, nargs='*', default=[512, 512], help='ActorCritic hidden dims')
    parser.add_argument('--actor_dropout_rate', type=float, default=0.3, help='Actor dropout rate')
    parser.add_argument('--critic_dropout_rate', type=float, default=0.3, help='Critic dropout rate')
    parser.add_argument('--act_dropout', type=float, default=0.3, help='Action dropout rate')
    parser.add_argument('--gamma', type=float, default=0.99, help='Reward discount factor')
    parser.add_argument('--target_update', type=float, default=0.05, help='Target network update ratio')

    ## Other parameters
    parser.add_argument('--threshold', type=float, default=0.6, help='Threshold to filter drug predictions')
    parser.add_argument('--factor', type=float, default=0.9, help='Decay factor for path probability score')

    main_args = parser.parse_args()
    main_args.pretrain_model_path = os.path.join(main_args.model_path, 'xgboost_model_3class', 'xgboost_model.pt')

    logger = utils.get_logger(os.path.join(main_args.log_dir, main_args.log_name))
    logger.info(main_args)
    main_args.logger = logger

    output_path = main_args.out_dir
    os.makedirs(os.path.join(output_path, 'prediction_scores'), exist_ok=True)
    os.makedirs(os.path.join(output_path, 'path_results'), exist_ok=True)

    dtd = creativeDTD(main_args, main_args.data_path, main_args.model_path)

    disease_set = pl.read_csv(main_args.disease_set, separator='\t')
    all_diseases = disease_set['id'].to_list()
    existing = {
        os.path.splitext(f)[0].replace(f'top{main_args.N_drugs}drugs_', '')
        for f in os.listdir(os.path.join(output_path, 'prediction_scores'))
    }
    diseases_to_run = [d for d in all_diseases if d not in existing]

    for disease in tqdm(diseases_to_run):
        if not dtd.set_query_disease(disease):
            continue

        if not dtd.filter_drugs_with_paths():
            continue

        predicted_drugs = dtd.predict_top_N_drugs(main_args.N_drugs, main_args.threshold)
        if predicted_drugs is None:
            continue

        predicted_drugs.write_csv(
            os.path.join(output_path, 'prediction_scores', f'top{main_args.N_drugs}drugs_{disease}.txt'),
            separator='\t',
        )

        predicted_paths = dtd.predict_top_M_paths(main_args.N_paths)
        rows = []
        if predicted_paths:
            for pair, paths in predicted_paths.items():
                for path in paths:
                    if len(path) >= 2:
                        rows.append([pair[0], pair[1], path[0], path[1]])

        path_table = pl.DataFrame(rows, schema=['drug_id', 'disease_id', 'path', 'path_score'], orient='row')
        if path_table.height > 0:
            path_table = (
                path_table
                .join(
                    predicted_drugs.select(['drug_id', 'drug_name', 'disease_id', 'disease_name']),
                    on=['drug_id', 'disease_id'], how='left',
                )
                .select(['drug_id', 'drug_name', 'disease_id', 'disease_name', 'path', 'path_score'])
            )
            path_table.write_csv(
                os.path.join(output_path, 'path_results', f'top{main_args.N_paths}paths_{disease}.txt'),
                separator='\t',
            )
