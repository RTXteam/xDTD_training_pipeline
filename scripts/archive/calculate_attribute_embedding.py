## Import Standard Packages
import sys, os
import argparse
import json
import pickle
from transformers import AutoModel, AutoTokenizer
import torch
import numpy as np
import polars as pl
from sklearn.decomposition import PCA
from tqdm import trange

## Import Personal Packages
pathlist = os.getcwd().split(os.path.sep)
ROOTindex = pathlist.index("xDTD_training_pipeline")
ROOTPath = os.path.sep.join([*pathlist[:(ROOTindex + 1)]])
sys.path.append(os.path.join(ROOTPath, 'scripts'))
import utils


def get_bert_embedding(texts, tokenizer, model, device):
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt", max_length=512)
    inputs = inputs.to(device)
    with torch.no_grad():
        embeddings = model(**inputs, output_hidden_states=True, return_dict=True).pooler_output
    return embeddings.detach().to("cpu").numpy()


def load_data(node_info_path, logger):
    """Load node info TSV and build (id -> index) mapping and text list.

    Text for each node is: "<name> <leaf_cat1> <leaf_cat2> ..."
    where leaf categories are the most specific non-mixin biolink categories.
    """
    df = pl.read_csv(node_info_path, separator='\t')
    ids = df['id'].to_list()
    names = df['name'].to_list()
    all_cats = df['all_categories'].to_list()

    id2index = {}
    texts = []
    for i, (nid, name, cats_json) in enumerate(zip(ids, names, all_cats)):
        id2index[nid] = i
        categories = json.loads(cats_json)
        leaves = utils.get_leaf_categories(categories)
        texts.append(f"{name} {' '.join(leaves)}")

    logger.info(f"Loaded {len(texts)} nodes from {node_info_path}")
    return id2index, texts


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", type=str, help="The path of logfile folder", default=os.path.join(ROOTPath, "log_folder"))
    parser.add_argument("--log_name", type=str, help="log file name", default="step12_calculate_attribute_embedding.log")
    parser.add_argument('--gpu', type=int, help='gpu device (default: 0)', default=0)
    parser.add_argument("--use_gpu", action="store_true", help="Whether use GPU or not", default=False)
    parser.add_argument('--node_info', type=str, help='Path to a file containing node information', default=os.path.join(ROOTPath, "data", "filtered_graph_nodes_info.txt"))
    parser.add_argument('--seed', type=int, help='Random seed (default: 1023)', default=1023)
    parser.add_argument('--pca_components', type=int, help='Number of components for PCA', default=100)
    parser.add_argument("--batch_size", type=int, help="Batch size of bert embedding calculation", default=10)
    parser.add_argument("--output_folder", type=str, help="The path of output folder", default=os.path.join(ROOTPath, "data"))
    args = parser.parse_args()

    logger = utils.get_logger(os.path.join(args.log_dir, args.log_name))
    logger.info(args)
    utils.set_random_seed(args.seed)

    id2index, texts = load_data(args.node_info, logger)
    index2id = {v: k for k, v in id2index.items()}

    embedding_dir = os.path.join(args.output_folder, "text_embedding")
    os.makedirs(embedding_dir, exist_ok=True)

    with open(os.path.join(embedding_dir, "ca.json"), "w") as f:
        json.dump(id2index, f)
    with open(os.path.join(embedding_dir, "index2id.json"), "w") as f:
        json.dump(index2id, f)

    ## Device setup
    if args.use_gpu and torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu}')
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.set_device(args.gpu)
    else:
        if args.use_gpu:
            logger.warning('No GPU detected. Falling back to CPU.')
        device = torch.device('cpu')

    ## Load model
    model_name = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.to(device)

    ## Compute BERT embeddings in batches
    n_texts = len(texts)
    ori_embedding = np.zeros((n_texts, 768))
    logger.info(f"Computing BERT embeddings for {n_texts} nodes on {device} (batch_size={args.batch_size})")

    for start in trange(0, n_texts, args.batch_size, desc="BERT embedding"):
        end = min(start + args.batch_size, n_texts)
        ori_embedding[start:end] = get_bert_embedding(texts[start:end], tokenizer, model, device)

    ## PCA dimensionality reduction
    logger.info(f"Fitting PCA with {args.pca_components} components")
    pca = PCA(n_components=args.pca_components)
    pca_embedding = pca.fit_transform(ori_embedding)

    ## Save embeddings
    id2embedding = {n_id: pca_embedding[idx] for n_id, idx in id2index.items()}
    output_path = os.path.join(embedding_dir, "embedding_biobert_namecat.pkl")
    with open(output_path, 'wb') as f:
        pickle.dump(id2embedding, f, protocol=pickle.HIGHEST_PROTOCOL)
    logger.info(f"Saved {len(id2embedding)} embeddings to {output_path}")
