
from transformers import AutoTokenizer, AutoModel
import torch
from torch.nn.functional import cosine_similarity

model_name = "microsoft/unixcoder-base-nine"  
# model_name = "/home/duangliang/unixcoder-base-nine/unixcoder-base-nine" # currently use local model for convenience
tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=False)
model = AutoModel.from_pretrained(model_name, local_files_only=False)

model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)



def get_code_embeddings(code: str):
    inputs = tokenizer(code, return_tensors="pt", truncation=True, padding=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        embeddings = model(**inputs).last_hidden_state.mean(dim=1)  
    return embeddings.cpu()

def compute_similarity(code1: str, code2: str):
    embedding1 = get_code_embeddings(code1)
    embedding2 = get_code_embeddings(code2)

    cos_sim = cosine_similarity(embedding1, embedding2)
    return cos_sim.item()



class RagEmbedder:
    def __init__(self, model_name, max_length=512, enable_amp=True):
        self.device = device
        self.tokenizer = tokenizer
        self.model = model
        self.max_length = max_length
        self.enable_amp = enable_amp
        self.amp_dtype = torch.float16 if self.device.type == "cuda" else torch.float32

        if self.device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf31 = True

    def _preprocess(self, code):
        if isinstance(code, str):
            return [code]
        return code
    
    @torch.no_grad()
    def encode(self, code, batch_size=32, normalize=True):
        code_list = self._preprocess(code)
        all_embeddings = []

        for i in range(0, len(code_list), batch_size):
            batch = code_list[i:i+batch_size]
            inputs = self.tokenizer(batch, return_tensors="pt", padding="longest", truncation=True, 
                                    max_length=self.max_length, add_special_tokens=True).to(self.device)
            
            with torch.autocast(device_type=self.device.type, dtype=self.amp_dtype, enabled=self.enable_amp):
                outputs = self.model(**inputs)
                embeddings = outputs.last_hidden_state.mean(dim=1)

            if normalize:
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            all_embeddings.append(embeddings.cpu())
        return torch.cat(all_embeddings, dim=0)
    
    def similarity(self, code1, code2):
        embedding1 = self.encode(code1)
        embedding2 = self.encode(code2)
        return torch.nn.functional.cosine_similarity(embedding1, embedding2).item()
    
    @property
    def embedding_dim(self):
        return self.model.config.hidden_size
    

embedder = RagEmbedder("microsoft/unixcoder-base-nine")

def encode(code, batch_size=32, normalize=True):
    return embedder.encode(code, batch_size, normalize)

def similarity(code1, code2):
    return embedder.similarity(code1, code2)
