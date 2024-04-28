import torch
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel, PeftConfig

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

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-hf')
model = get_model('castorini/repllama-v1-7b-lora-passage')

# Define query and passage inputs
query = "What is llama?"
title = "Llama"
passage = "The llama is a domesticated South American camelid, widely used as a meat and pack animal by Andean cultures since the pre-Columbian era."
query_input = tokenizer(f'query: {query}</s>', return_tensors='pt')
passage_input = tokenizer(f'passage: {title} {passage}</s>', return_tensors='pt')

# Run the model forward to compute embeddings and query-passage similarity score
with torch.no_grad():
    # compute query embedding
    query_outputs = model(**query_input)
    query_embedding = query_outputs.last_hidden_state[0][-1]
    query_embedding = torch.nn.functional.normalize(query_embedding, p=2, dim=0)

    # compute passage embedding
    passage_outputs = model(**passage_input)
    passage_embeddings = passage_outputs.last_hidden_state[0][-1]
    passage_embeddings = torch.nn.functional.normalize(passage_embeddings, p=2, dim=0)

    # compute similarity score
    score = torch.dot(query_embedding, passage_embeddings)
    print(score)
