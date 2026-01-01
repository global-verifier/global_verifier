import re
try:
    from bs4 import BeautifulSoup, Comment
except ImportError:  # pragma: no cover
    BeautifulSoup = None
    Comment = None

def extract_visible_text(html_content):
    """
    从HTML内容中提取可见文本
    """
    if BeautifulSoup is None or Comment is None:
        raise ImportError("bs4 is required for extract_visible_text(). Please install beautifulsoup4.")
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # 移除包含Instruction的元素（通过id="instruction-text"定位） 用于 webshop 环境 TODO: this is webshop specific, need to be generalized
    instruction_div = soup.find(id='instruction-text')
    if instruction_div:
        instruction_div.decompose()
    
    # 移除script和style标签及其内容
    for tag in soup(['script', 'style', 'noscript']):
        tag.decompose()
    
    # 移除HTML注释
    comments = soup.find_all(string=lambda text: isinstance(text, Comment))
    for comment in comments:
        comment.extract()
    
    # 提取文本
    text = soup.get_text(separator=' ', strip=True)
    
    # 清理多余的空白字符，去除所有换行符（包括字面量\n和真实换行符）
    text = text.replace('\\n', ' ')  # 去除字面量 \n
    text = text.replace('\n', ' ')  # 去除真实换行符
    text = re.sub(r'[ \t]+', ' ', text)  # 多个空格/制表符合并为单个空格
    text = text.strip()
    
    return text


def frozenlake_goal_positions(desc):
    """
    输入 FrozenLake 的 desc，输出所有 'G' 的坐标列表 [(r, c), ...]。

    兼容常见 desc 格式：
    - list[str]，例如 ["SFFF", "FHFH", "FFFH", "HFFG"]
    - gymnasium 内部 desc（通常为 numpy array / list，元素为 bytes）
    """
    if desc is None:
        raise ValueError("desc cannot be None")

    goals = []
    for r, row in enumerate(desc):
        # row may be a string (e.g. "SFFFHG") or an iterable of bytes (gym's internal desc)
        if isinstance(row, bytes):
            row_iter = row.decode("utf-8")
        else:
            row_iter = row

        if isinstance(row_iter, str):
            for c, ch in enumerate(row_iter):
                if ch == "G":
                    goals.append((r, c))
        else:
            for c, cell in enumerate(row_iter):
                ch = cell.decode("utf-8") if isinstance(cell, bytes) else str(cell)
                if ch == "G":
                    goals.append((r, c))

    return goals

def format_full_llama_prompt(system_prompt, user_prompt):
        prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
{system_prompt} 
<|eot_id|>
<|start_header_id|>user<|end_header_id|>
{user_prompt}
<|eot_id|> 
<|start_header_id|>assistant<|end_header_id|>
"""
        return prompt

def format_full_mistral_prompt(system_prompt, user_prompt):
    prompt = f"<s>[INST] {system_prompt}\n\n{user_prompt} [/INST]"
    return prompt

def format_full_qwen_prompt(system_prompt, user_prompt):
        prompt = (
            f"<|im_start|>system\n{system_prompt}\n<|im_end|>\n"
            f"<|im_start|>user\n{user_prompt}\n<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        return prompt

def format_full_internlm_prompt(system_prompt, user_prompt):
    """
    InternLM3-8B-Instruct 采用 ChatML 格式。
    参考: https://huggingface.co/internlm/internlm3-8b-instruct
    """
    prompt = (
        f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        f"<|im_start|>user\n{user_prompt}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    return prompt

def format_full_deepseek_prompt(system_prompt, user_prompt):
    """
    DeepSeek-Coder-V2-Lite-Instruct 格式。
    参考: https://huggingface.co/deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct
    """
    # DeepSeek Coder V2 通常以 <｜begin▁of▁sentence｜> 开始，使用 User/Assistant 角色标记
    prompt = f"<｜begin▁of▁sentence｜>{system_prompt}\n\nUser: {user_prompt}\n\nAssistant: "
    return prompt

def format_full_openai_prompt(system_prompt, user_prompt):
    prompt = f"{system_prompt}\n\n{user_prompt}"
    return prompt

def choose_format_full_prompt(model_name):
    # choose the format_full_xxx_prompt function
    if "llama" in model_name:
        return format_full_llama_prompt
    elif "mistral" in model_name:
        return format_full_mistral_prompt
    elif "qwen" in model_name:
        return format_full_qwen_prompt
    elif "internlm" in model_name:
        return format_full_internlm_prompt
    elif "deepseek" in model_name:
        return format_full_deepseek_prompt
    else:
        return format_full_openai_prompt

