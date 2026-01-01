from .base_explorer_model import BaseExplorerModel
from openai import OpenAI
import os
from config import api


class OpenAIExplorerModel(BaseExplorerModel):
    def __init__(self, model_name, max_new_tokens: int = 64):
        """
        初始化 OpenAI 模型。
        支持 Chat 模型 (gpt-4o, gpt-3.5-turbo) 和 Instruct 模型 (gpt-3.5-turbo-instruct)
        """
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("Environment variable OPENAI_API_KEY is not set.")
            
        self.client = OpenAI(api_key=api_key, base_url=api["base_url"])
        self.model_name = model_name
        # self.model_name = "gpt-4o"
        self.max_new_tokens = max_new_tokens

    def digest_goal(self, goal: str) -> None:
        pass

    def get_next_action(self, get_action_prompt: str) -> str:
        try:
            # 针对 gpt-3.5-turbo-instruct 使用 Completions API
            if self.model_name == "gpt-3.5-turbo-instruct":
                response = self.client.completions.create(
                    model=self.model_name,
                    prompt=get_action_prompt, # 直接传入拼接好的字符串
                    max_tokens=self.max_new_tokens,
                    temperature=0.9,
                    top_p=1.0,
                )
                action = response.choices[0].text.strip()
            
            # 针对其他 Chat 模型 (GPT-4o, GPT-3.5-turbo 等) 使用 Chat API
            else:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "user", "content": get_action_prompt}
                    ],
                    max_tokens=self.max_new_tokens,
                    temperature=0.9,
                    top_p=1.0,
                )
                action = response.choices[0].message.content.strip()

            # 清理动作文本（防止模型输出换行后的解释）
            if '\n' in action:
                 action = action.split('\n')[0]
            
            return action.strip()
            
        except Exception as e:
            print(f"OpenAI API Error: {e}")
            return ""