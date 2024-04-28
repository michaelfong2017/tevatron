# import torch
# from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig
# from peft import PeftModel, PeftConfig

# def get_model(peft_model_name):
#     config = PeftConfig.from_pretrained(peft_model_name)
#     bnb_config = BitsAndBytesConfig(
#         load_in_4bit=True,
#         bnb_4bit_use_double_quant=True,
#         bnb_4bit_quant_type="nf4",
#         bnb_4bit_compute_dtype=torch.bfloat16
#     )
#     base_model = AutoModel.from_pretrained(config.base_model_name_or_path, quantization_config=bnb_config, device_map={"":0})
#     model = PeftModel.from_pretrained(base_model, peft_model_name)
#     model = model.merge_and_unload()
#     model.eval()
#     return model

# # Load the tokenizer and model
# tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-hf')
# model = get_model('castorini/repllama-v1-7b-lora-passage')

# # Define query and passage inputs
# query = "What is llama?"
# title = "Llama"
# passage = "The llama is a domesticated South American camelid, widely used as a meat and pack animal by Andean cultures since the pre-Columbian era."
# query_input = tokenizer(f'query: {query}</s>', return_tensors='pt')
# passage_input = tokenizer(f'passage: {title} {passage}</s>', return_tensors='pt')

# # Run the model forward to compute embeddings and query-passage similarity score
# with torch.no_grad():
#     # compute query embedding
#     query_outputs = model(**query_input)
#     query_embedding = query_outputs.last_hidden_state[0][-1]
#     query_embedding = torch.nn.functional.normalize(query_embedding, p=2, dim=0)

#     # compute passage embedding
#     passage_outputs = model(**passage_input)
#     passage_embeddings = passage_outputs.last_hidden_state[0][-1]
#     passage_embeddings = torch.nn.functional.normalize(passage_embeddings, p=2, dim=0)

#     # compute similarity score
#     score = torch.dot(query_embedding, passage_embeddings)
#     print(score)

# import pandas as pd
# from transformers import AutoTokenizer, AutoModel

# # Specify the path to the pickle file
# pkl_file_path = "beir_embedding_scifact/corpus_scifact.0.pkl"

# # Read the pickle file into a Pandas DataFrame
# df = pd.read_pickle(pkl_file_path)

# tokens = df[0][0]

# # Load the model and tokenizer
# model_name = "castorini/repllama-v1-7b-lora-passage"
# tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-hf')

# # Convert the array to tokens
# # tokens = tokenizer.convert_ids_to_tokens(ids)

# # Convert tokens back to text
# text = tokenizer.convert_tokens_to_string(tokens)

# print(text)

from beir import util, LoggingHandler
from beir.retrieval import models
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES

import torch
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel, PeftConfig

import logging
import pathlib, os

#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
#### /print debug information to stdout

#### Download scifact.zip dataset and unzip the dataset
dataset = "scifact"
url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(dataset)
out_dir = os.path.join(pathlib.Path(__file__).parent.absolute(), "datasets")
data_path = util.download_and_unzip(url, out_dir)

#### Provide the data_path where scifact has been downloaded and unzipped
corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="test")

def get_model(peft_model_name):
    config = PeftConfig.from_pretrained(peft_model_name)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    base_model = AutoModel.from_pretrained(config.base_model_name_or_path, quantization_config=bnb_config, device_map={"":0})
    model = PeftModel.from_pretrained(base_model, peft_model_name)
    model = model.merge_and_unload()
    model.eval()
    return model

#### Load the SBERT model and retrieve using cosine-similarity
model = DRES(get_model("castorini/repllama-v1-7b-lora-passage"), batch_size=16)
retriever = EvaluateRetrieval(model, score_function="dot") # or "cos_sim" for cosine similarity
results = retriever.retrieve(corpus, queries)

#### Evaluate your model with NDCG@k, MAP@K, Recall@K and Precision@K  where k = [1,3,5,10,100,1000] 
ndcg, _map, recall, precision = retriever.evaluate(qrels, results, retriever.k_values)

print(ndcg, _map, recall, precision)