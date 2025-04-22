# from LLM import generation_code_in_parallel
# from check_rust_compilation import check_rust_compilation
# from translate_function import llm_translation_with_rag
# from repair import process_translation

# # message = "请用 Rust 写一个 Fibonacci 函数"
# # codes = generation_code_in_parallel(message)


# # for i, code in enumerate(codes):
# #     print(f"[Client {i + 1}]\n{code}\n{'=' * 40}")
# #     success, stdout, stderr = check_rust_compilation(code)
# #     if success:
# #         print("SUCCESS")

# rust_code = """
# fn main() {
#     prin!("Hello, world!");
# }
# """

# c_code = """printf("Hello world!");"""

# # success, stdout, stderr = check_rust_compilation(rust_code)

# # print(stderr)

# # response=process_translation(corpus_lang="C",corpus_func=c_code,previous_response=rust_code,error_message=stderr)
# # print(response)


# def Trans(source_lang, source_code, target_lang="Rust", target_model=None):
#     response_code = llm_translation_with_rag(source_lang=source_lang, source_code=source_code,target_model=target_model)
#     for in range(5):
#         success, stdout, stderr = check_rust_compilation(rust_code)
#         if success:
#             break
#         else:
#             response_code=process_translation(corpus_lang=source_lang,cropus_func=source_code,previous_response=response_code,error_message=stderr)
#     return response_code

# Trans(source_lang="C",source_code=c_code)
from typing import Tuple, Optional, List
from LLM import generation_code_in_parallel
from check_rust_compilation import check_rust_compilation
from translate_function import llm_translation_with_rag
from repair import process_translation

MAX_RETRIES = 5

def translate_and_repair(
    source_lang: str,
    source_code: str,
    target_lang: str = "Rust",
    target_model: Optional[str] = None
) -> Tuple[Optional[str], bool]:
    """
    执行代码翻译并自动修复编译错误
    
    :param source_lang: 源代码语言 (e.g. "C")
    :param source_code: 需要翻译的源代码
    :param target_lang: 目标语言 (默认为Rust)
    :param target_model: 可选的目标模型参数
    :return: (最终代码, 是否成功)
    """
    # 初始翻译（明确处理列表返回值）
    translated_candidates: List[str] = llm_translation_with_rag(
        source_lang=source_lang,
        source_code=source_code,
        target_model=target_model
    )
    
    if not translated_candidates:
        print("Initial translation failed")
        return None, False

    # 选择第一个候选代码进行验证
    current_code: str = translated_candidates[0]
    last_valid_code: Optional[str] = None

    for attempt in range(MAX_RETRIES):
        print(f"Attempt {attempt + 1}/{MAX_RETRIES}")

        # 检查编译（确保传入字符串）
        success, stdout, stderr = check_rust_compilation(current_code)
        
        if success:
            print("Compilation successful!")
            return current_code, True
            
        print(f"Compilation failed. Error:\n{stderr}")
        
        # 保留最后一个有效代码（如果有）
        if "warning" in stderr.lower():
            last_valid_code = current_code
            
        # 尝试修复（确保返回字符串）
        repaired_code: Optional[str] = process_translation(
            corpus_lang=source_lang,
            corpus_func=source_code,  # 修正参数名拼写错误
            previous_response=current_code,
            error_message=stderr
        )
        
        if not repaired_code or repaired_code == current_code:
            print("No improvement in repair")
            break
            
        current_code = repaired_code

    return (last_valid_code or current_code), (last_valid_code is not None)

# 示例用法
if __name__ == "__main__":
    sample_c_code = """printf("Hello world!");"""
    
    final_code, success = translate_and_repair(
        source_lang="C",
        source_code=sample_c_code
    )
    
    if success:
        print("\nFinal valid code:")
        print(final_code)
    else:
        print("\nFailed to produce valid code after retries")
        if final_code:
            print("Last attempted code:")
            print(final_code)