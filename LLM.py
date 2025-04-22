from openai import OpenAI
import concurrent.futures
import sys
import json
import re
from tenacity import retry, stop_after_attempt, wait_fixed, wait_random_exponential



def extract_code(content):
    

    translated_result = content

    translated_code = None

    patterns = [r'```rust(.*?)```',r'```Rust(.*?)```']

    for pattern in patterns:
        if translated_code == None:
            try:
                translated_code = re.findall(pattern, translated_result, re.DOTALL)[0].strip()
                
            except:
                translated_code = None
        else:
            break
    
    if translated_code == None:
        translated_code = translated_result
    

    return translated_code


class LLM():

    SUPPORT_MODEL_LIST = [
        "claude-3-5-sonnet-20240620",
        "claude-3-7-sonnet-20250219",
        "deepseek-v3"
    ]
     
    def __init__(
        self, 
        api_keys : list,
        model_name: str = None
    ):
        if model_name is None:
            if not self.SUPPORT_MODEL_LIST:
                raise ValueError("SUPPORT_MODEL_LIST is empty, must explicitly specify model_name")
            
            model_name = self.SUPPORT_MODEL_LIST[0]

        if model_name not in self.SUPPORT_MODEL_LIST:
            raise ValueError(f"Unsupported model: {model_name}. Supported models are: {', '.join(self.SUPPORT_MODEL_LIST)}")
        
        self.clients = [OpenAI(api_key=api_key, base_url="https://sg.uiuiapi.com/v1") for api_key in api_keys]
        self.model_name = model_name


    def generation_in_parallel(self, generator, model_name=None):
        used_model = model_name or self.model_name
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(self.clients)) as executor:
            future_to_path = {
                executor.submit(self.generation, message, self.clients[i % len(self.clients)], model_name=used_model): (path, message, query_func, original_function_name)
                for i, (path, message, query_func, original_function_name) in enumerate(generator)
            }
            
            for future in concurrent.futures.as_completed(future_to_path):
                path, message, query_func, original_function_name = future_to_path[future]
                response = future.result()

                translated_code = extract_code(response)

                with open(path, 'w', encoding='utf-8', errors='ignore') as output_file:
                    output_file.write(f"<message>\n{message}\n</message>\n")
                    output_file.write(f"<response>\n{response}\n</response>\n")
                    output_file.write(f"<function>\n{query_func}\n</function>\n<translated function>\n{translated_code}</translated function>")

    @retry(wait=wait_random_exponential(min=1, max=10), stop=stop_after_attempt(6))
    def generation(self, content, client, temperature=0, model_name=None):
        used_model = model_name if model_name else self.model_name
        if used_model not in self.SUPPORT_MODEL_LIST:
            raise ValueError(f"Unsupported model: {model_name}. Supported models are: {', '.join(self.SUPPORT_MODEL_LIST)}")
        
        response = client.chat.completions.create(
            model=used_model, 
            messages=[
                {
                    "role": "user", 
                    "content": content
                }
            ],
            temperature=temperature
        )
        if response.choices[0].message.content:
            return response.choices[0].message.content 
        else:
            raise ValueError("Empty response from API")
    
    def generation_code_in_parallel(self, message, model_name=None):
        used_model = model_name or self.model_name
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(self.clients)) as executor:
            future_to_client = {
                executor.submit(self.generation, message, client, model_name=used_model): client
                for client in self.clients
            }

            translation_code = []
            for future in concurrent.futures.as_completed(future_to_client):
                try:
                    response = future.result()

                    code = extract_code(response)
                    translation_code.append(code)
                except Exception as e:
                    print(f"Error processing request: {str(e)}")
                
            return translation_code
    
    def support_model_list(self):
        return self.SUPPORT_MODEL_LIST


# your own api key
api_keys = [
    "sk-3wqcbv2TkXsFaDbpZVtTV0LsD93LI50GEK9HJaJjWb6GJYJ6"
]

llm = LLM(api_keys, "claude-3-5-sonnet-20240620")

def generation(message, model_name=None):
    return llm.generation(message, model_name=model_name)

def generation_in_parallel(message, model_name=None):
    return llm.generation_in_parallel(message, model_name=model_name)

def generation_code_in_parallel(message, model_name=None):
    return llm.generation_code_in_parallel(message, model_name=model_name)