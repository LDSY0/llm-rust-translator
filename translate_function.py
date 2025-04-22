import os
import logging
import time
import sys
import re
import json
from LLM import generation, generation_in_parallel, generation_code_in_parallel

from retrieval import retrieve_translation_pairs

from itertools import islice

logging.basicConfig(filename=f"translate_throughLLM.log", level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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

def read_user_message(corpus_lang, corpus_func, query_lang="Rust"):
    logging.info(f"[MessageBuilder] start constructing {corpus_lang}->{query_lang} translation prompt message")

    # currently only one example is used
    translation_pair_examples_lists = retrieve_translation_pairs(corpus_func, 1)

    translation_pair_examples = ""
    if len(translation_pair_examples_lists) > 0:
        translation_pair_examples = f"""[source code]
{translation_pair_examples_lists[0]["source"]}
[translation result]
{translation_pair_examples_lists[0]["translation"]}
"""

    logging.info(f"[MessageBuilder] Retrieved {len(translation_pair_examples_lists)} translation pair examples")

    translation_examples_func = " "
    query_func_signature = " "
    related_function_and_datatype = " "
    use_package = " "

    translation_message = ""
                
    translation_message = f"""
Here are some rust codes for reference.
"""
    translation_message += translation_examples_func
                
    translation_message += """
Here are some translation example for reference.
"""
    translation_message += translation_pair_examples

    message = f"""
Translate the focal {corpus_lang} function to {query_lang} 

Here are the basic details about the function under translation
<focal {corpus_lang} function>
[source code of function]
{corpus_func}
</focal {corpus_lang} function>
<{query_lang} function signature>
{query_func_signature}
</{query_lang} function signature>
<{query_lang} function dependencies, and data type declarations>\n{related_function_and_datatype}\n</{query_lang} function dependencies and data type declarations>
<{query_lang} function dependency libraries>{use_package}\n</{query_lang} function dependency libraries>

Please translate the function following the given steps 
1. Confirm the functionality to be implemented by the current function
2. Distinguish the dependencies differences between the source and target programming languages
2.1 The focal function may have differences in dependencies between the source language and the programming language. For example:
    - Variable Differences: For example, variable names may differ, or the structure of the data types corresponding to the variables may vary.
    - Function Differences: For instance, function names might be different, or a function that exists in the source language might not be available in the target language.
    - Data Type Differences: There may be differences in custom data types between the source language and the target language.
2.2 Enumerate all used dependencies including function, data type and variable within this function in target programming languages(if any)
3. Enumerate all used local variables and their mut status and data types in previous translated code snippets(if any)
4. Distinguish the syntax differences between the source and target programming languages(if any). For example:
    - In C, Java, and Python, variables declared are mutable by default, while in Rust, variables are immutable by default. To make a variable mutable in Rust, the `mut` keyword must be explicitly used. 
    - In C and Java, null pointer checks are performed at runtime for variables, whereas in Rust, such checks are not required for variables that are not of the `Option` type. 
    - Memory safety checks required in C are not required in Rust. 
    - Data type checking required in Python is not required in Rust.
5. Translate the focal function based on the functionality implemented by the function, the dependencies used, the translated code snippets and the syntax differences. 
    
The translation process must adhere to the following rules:
    - Do not perform a simple one-to-one translation; instead, consider the functional consistency of the code. The translated result only needs to achieve the same functionality as the original language's function.
    - ** Ensure that the local variables used genuinely exist in the <translated code snippets>, the dependencies used are actually present in the provided <{query_lang} function dependencies, and data type declarations>, and the syntax used is valid in the target language. **
    - For an `if` code block, if the condition being checked is unnecessary in Rust according to the syntax differences, the `if` block can be omitted entirely, and the translated result should be an empty line. Examples include null pointer checks in C and Java, type checks in Python, or memory safety checks for structs in C.

{translation_message}

```{query_lang}
** (only reply with the translated result of the focal function) **
```
"""
    print(message)

    logging.info(f"[MessageBuilder] message construction completed")

    return message




def llm_translation_with_rag(source_lang, source_code, target_lang="Rust", target_model=None):
    model_name = target_model or "claude-3-5-sonnet-20240620"
    
    try:
        message = read_user_message(source_lang, source_code, target_lang)

        logging.info(f"[{model_name}] start {source_lang}->{target_lang} parallel code translation")
        
        start_time = time.time()
        translation = generation_code_in_parallel(message, model_name=target_model)
        duration = time.time() - start_time
        logging.info(f"[{model_name}] completed translation | time consumed: {duration:.2f}s")
        return translation
    except Exception as e:
        logging.error(f"[{model_name}] translation failed | error type: {type(e).__name__} | details: {str(e)}", exc_info=True)
        raise






# if __name__ == "__main__":
#     source_dir = sys.argv[1]
#     target_dir = sys.argv[2]
#     llm = sys.argv[3]
#     dependencies_path = sys.argv[4]
#     rag_path_function = sys.argv[5]
#     translation_pair_path = sys.argv[6]
    
#     message_generator = read_message(source_dir, target_dir, llm, dependencies_path, rag_path_function, translation_pair_path)
#     generation_in_parallel(message_generator)