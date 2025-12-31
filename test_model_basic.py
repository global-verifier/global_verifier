import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "6"


# 确保能引用到 global_verifier 目录
sys.path.append(os.getcwd())

from explorer_model.internlm_explorer_model import InternLMExplorerModel
from explorer_model.deepseek_explorer_model import DeepSeekExplorerModel
from env_adaptors.adopter_util import format_full_internlm_prompt, format_full_deepseek_prompt
from config import model_path
from openai import OpenAI

# OpenAI API 配置（来源：用户提供）
OPENAI_BASE_URL = "https://hk.yi-zhan.top/v1"
OPENAI_MODEL = "gpt-3.5-turbo-instruct"


def test_openai():
    print("=== Testing OpenAI API (gpt-3.5-turbo-instruct) ===")
    client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)
    prompt = "Say hello world in one short line."
    try:
        resp = client.completions.create(model=OPENAI_MODEL, prompt=prompt, max_tokens=16)
        text = resp.choices[0].text.strip()
        print(f"Prompt:\n{prompt}")
        print(f"Response: {text}")
    except Exception as e:
        print(f"OpenAI API Error: {e}")


def test_internlm():
    print("=== Testing InternLM3 ===")
    path = model_path["internlm3-8b"]  # 你的本地路径或 HF ID
    try:
        model = InternLMExplorerModel(path)
        system = "You are a helpful assistant. Answer 1 number only, no other text."
        user = "What is 1+1?"
        prompt = format_full_internlm_prompt(system, user)
        print(f"Prompt:\n{prompt}")
        response = model.get_next_action(prompt)  # 如果有显卡可以取消注释测试生成
        print(f"Response: {response}")
        print("InternLM loaded successfully.")
    except Exception as e:
        print(f"InternLM Error: {e}")


def test_deepseek():
    print("\n=== Testing DeepSeek Coder V2 ===")
    path = model_path["deepseek-v2"]  # 你的本地路径或 HF ID
    try:
        model = DeepSeekExplorerModel(path)
        system = "You are a coding assistant."
        user = "Print hello world in python."
        prompt = format_full_deepseek_prompt(system, user)
        print(f"Prompt:\n{prompt}")
        response = model.get_next_action(prompt)  # 如果有显卡可以取消注释测试生成
        print(f"Response: {response}")
        print("DeepSeek loaded successfully.")
    except Exception as e:
        print(f"DeepSeek Error: {e}")


if __name__ == "__main__":
    # 注意：显存占用较大，OpenAI 走云端，其他模型需显存。
    test_openai()
    # test_internlm()
    # test_deepseek()