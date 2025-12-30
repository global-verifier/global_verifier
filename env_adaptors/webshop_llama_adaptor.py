from .webshop_adaptor import WebshopAdaptor
from .adaptor_prompt_factory import build_webshop_user_prompt, WEBSHOP_SYSTEM_PROMPT

class WebshopLlamaAdaptor(WebshopAdaptor):
    def __init__(self, env_name, enable_confirm_purchase=None, session=None):
        super().__init__(
            env_name,
            enable_confirm_purchase=enable_confirm_purchase,
            session=session,
        )

    def get_action_prompt(self, retrieved_experiences=None):
        """生成用于LLM获取下一个action的prompt"""
        user_prompt = build_webshop_user_prompt(
            state=self.get_state(),
            instruction=self.get_instruction(),
            action_status=self.get_available_actions(),
            action_path=self.get_action_path(),
            retrieved_experiences=retrieved_experiences,
        )
        # Construct the prompt
        prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
{WEBSHOP_SYSTEM_PROMPT} 
<|eot_id|>
<|start_header_id|>user<|end_header_id|>
{user_prompt}
<|eot_id|> 
<|start_header_id|>assistant<|end_header_id|>
"""
        return prompt
