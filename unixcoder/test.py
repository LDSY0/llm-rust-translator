
from transformers import AutoTokenizer, AutoModel
import torch
from torch.nn.functional import cosine_similarity

model_name = "microsoft/unixcoder-base-nine"  
tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
model = AutoModel.from_pretrained(model_name, local_files_only=True)




def get_code_embeddings(code: str):
    inputs = tokenizer(code, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        embeddings = model(**inputs).last_hidden_state.mean(dim=1)  
    return embeddings

def compute_similarity(code1: str, code2: str):
    embedding1 = get_code_embeddings(code1)
    embedding2 = get_code_embeddings(code2)

    cos_sim = torch.nn.functional.cosine_similarity(embedding1, embedding2)
    return cos_sim.item()

