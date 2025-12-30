from .webshop_adaptor import WebshopAdaptor
from .adaptor_prompt_factory import build_webshop_user_prompt, WEBSHOP_SYSTEM_PROMPT

class WebshopMistralAdaptor(WebshopAdaptor):
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
        
        # Construct the prompt in Mistral / Mistral format
        prompt = f"<s>[INST] {WEBSHOP_SYSTEM_PROMPT}\n\n{user_prompt} [/INST]"
        return prompt