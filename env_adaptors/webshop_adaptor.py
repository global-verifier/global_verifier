import sys
import os
import gym
from base_env_adaptor import BaseEnvAdaptor
from env_config import webshop_config
from adopter_util import extract_visible_text

_webshop_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'webshop')
_webshop_path = os.path.abspath(_webshop_path)
if _webshop_path not in sys.path:
    sys.path.insert(0, _webshop_path)

# Import to register the environment with gym
from web_agent_site.envs import WebAgentTextEnv

SID_PLACEHOLDER = '<session_id>'

class WebshopAdaptor(BaseEnvAdaptor):
    def __init__(self, env_name):
        super().__init__(env_name)
        self.env = gym.make(webshop_config['id'], observation_mode=webshop_config['observation_mode'], num_products=webshop_config['num_products'])
        self.url_id = None

    def initialize_env(self):
        self.env.reset()
        self.url_id = self.env.state['url'].split('/')[-1]

    def get_instruction(self):
        """获取环境的 instruction，提取 "Instruction: " 之后的内容"""
        instruction_text = self.env.get_instruction_text()
        return instruction_text.split("Instruction: ", 1)[1].strip()

    def get_state(self):
        state = {}
        # get clean url
        full_url = self.env.state['url']
        clean_url = full_url.replace(self.url_id, SID_PLACEHOLDER)
        state['url'] = clean_url

        # get the html
        full_html = self.env.state['html']
        display_text = extract_visible_text(full_html)
        state['html'] = display_text

        # TODO: whether to put available actions to the state
        
        return state

    def get_available_actions(self):
        available_actions = self.env.get_available_actions()
        return available_actions
    
    def step(self, action):
        self.env.step(action)

    # Tobe implemented in the subclass
