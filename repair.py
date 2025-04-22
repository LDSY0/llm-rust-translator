import os
import logging
import json
import time
import sys
import re
from typing import Optional

from LLM import generation_code_in_parallel  # 只导入需要的函数
from openai import AzureOpenAI  # 保留实际使用的导入

# 常量定义
TRANSLATION_TAG_PATTERN = r'<translated function>(.*?)</translated function>'
RUST_CODE_PATTERNS = [
    r'```rust(.*?)```',
    r'```Rust(.*?)```',
    r'<rust function>(.*?)</rust function>',
    r'<rust function translation>(.*?)</rust function translation>',
    r'<rust translated function>(.*?)</rust translated function>'
]
INVALID_CONTENT = {"Too long", "None"}
ENCODING = 'utf-8'

# 日志配置
logging.basicConfig(
    filename="translate_repair.log",
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def read_translated_function(file_path: str) -> Optional[bytes]:
    """读取并解析翻译后的Rust代码文件"""
    try:
        with open(file_path, 'r', encoding=ENCODING, errors='ignore') as f:
            content = f.read().strip()

        if content in INVALID_CONTENT:
            return None

        # 提取翻译结果部分
        translated_result = re.search(
            TRANSLATION_TAG_PATTERN,
            content,
            re.DOTALL
        )
        if not translated_result:
            logging.warning(f"No translation tag found in {file_path}")
            return None

        # 尝试多种模式匹配Rust代码
        rust_code = None
        for pattern in RUST_CODE_PATTERNS:
            match = re.search(pattern, translated_result.group(1), re.DOTALL)
            if match:
                rust_code = match.group(1).strip()
                break
        
        # 回退使用完整内容
        rust_code = rust_code or translated_result.group(1).strip()
        return rust_code.encode(ENCODING)

    except Exception as e:
        logging.error(f"Error processing {file_path}: {str(e)}", exc_info=True)
        return None

def process_translation(
    corpus_lang: str,
    corpus_func: str,
    previous_response: str,
    error_message: str,
    query_lang: str = "Rust"
) -> Optional[str]:
    """处理翻译修复请求"""
    try:
        # 使用多行字符串构造模板
        message_template = f"""\
You were asked to translate the given {corpus_lang} function to {query_lang} according to the \
{query_lang} specifications. Some errors occurred when executing your code. Please fix them.

<previous response>
{previous_response}
</previous response>

<error message>
{error_message}
</error message>

<{corpus_lang} function>
{corpus_func}
</{corpus_lang} function>

** (only reply with the translated result of the focal function) **
"""

        logging.debug(f"Sending request:\n{message_template}")
        print(message_template)
        response = generation_code_in_parallel(message_template)
        
        if response:
            logging.info(f"Received response: {response[:200]}...")  # 记录部分响应
            return response
        return None

    except Exception as e:
        logging.error(f"Processing failed: {str(e)}", exc_info=True)
        return None