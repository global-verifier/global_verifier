from .base_explorer_model import BaseExplorerModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from .model_util import extract_deepseek_response
import torch

class DeepSeekExplorerModel(BaseExplorerModel):
    def __init__(self, model_path: str, max_new_tokens: int = 64):
        """
        初始化 DeepSeek Coder V2 Explorer Model。
        DeepSeek Coder V2 是 MoE 架构，通常需要 trust_remote_code=True。
        """
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype="auto",
            device_map="auto",
            trust_remote_code=True
        )
        # Disable cache to avoid DynamicCache seen_tokens issues on some transformer versions
        self.model.config.use_cache = False
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, 
            trust_remote_code=True
        )
        self.max_new_tokens = max_new_tokens

    def digest_goal(self, goal: str) -> None:
        pass

    def get_next_action(self, get_action_prompt: str) -> str:
        """
        获取下一个动作。
        get_action_prompt 应该符合 DeepSeek 的格式，以 Assistant: 结尾。
        """
        model_inputs = self.tokenizer(
            [get_action_prompt],
            return_tensors="pt",
        ).to(self.model.device)

        with torch.no_grad():
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=self.max_new_tokens,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id,
                use_cache=False,
                do_sample=True,
                temperature=0.8,
                top_p=0.95,
            )

        # 去除 prompt 部分的 token
        generated_ids = [
            output_ids[len(input_ids):]
            for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        
        # 解码回复
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=False)[0]

        # 拼接完整的文本用于提取
        full_text = get_action_prompt + response
        action = extract_deepseek_response(full_text)
        return action