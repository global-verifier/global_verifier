import sys
import os
import gym
from .base_env_adaptor import BaseEnvAdaptor
from .env_config import webshop_config
from .adopter_util import extract_visible_text
import re
import random
from utils import get_timestamp_ms

_webshop_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'webshop')
_webshop_path = os.path.abspath(_webshop_path)
if _webshop_path not in sys.path:
    sys.path.insert(0, _webshop_path)

# Import to register the environment with gym
from web_agent_site.envs import WebAgentTextEnv

SID_PLACEHOLDER = '<session_id>'
QUERY_PLACEHOLDER = '<query>'

class WebshopAdaptor(BaseEnvAdaptor):
    def __init__(self, env_name):
        super().__init__(env_name)
        seed = webshop_config.get('random_seed')
        if seed is not None:
            random.seed(seed)
        self.env = gym.make(
            webshop_config['id'],
            observation_mode = webshop_config['observation_mode'],
            num_products = webshop_config['num_products'],
            human_goals = webshop_config['human_goals'],
        )
        self.url_id = None
        self.instruction = None
        # history records
        self.st = None
        self.prev_action = None
        self.st1 = None
        self.action_path = []

    # Getters
    def get_env_description(self):
        return f"""-----
Init new environment:
[ENV] Webshop
[URL ID]: {self.url_id}
[INSTRUCTION]: {self.instruction}
-----"""
    
    def get_instruction(self):
        return self.instruction

    def get_state(self):
        state = {}
        # get clean url - normalize it by replacing session_id and search query
        full_url = self.env.state['url']
        clean_url = self._normalize_url_query(full_url, self.url_id)
        state['url'] = clean_url

        # get the html
        full_html = self.env.state['html']
        display_text = extract_visible_text(full_html)
        state['html'] = display_text

        # TODO: whether to put available actions to the state
        
        return state
    
    def _normalize_url_query(self, url, url_id):
        """
        Normalize URL by replacing session_id and search query with placeholders.
        
        This enables experience matching across different sessions and search queries.
        
        URL patterns:
        - search_results: http://host/search_results/<session_id>/<search_query>/<page_num>
        - item_page: http://host/item_page/<session_id>/<product_id>/<search_query>/<page_num>/<options>
        """
        # Step 1: Replace session_id with placeholder
        # Replace /session_id/ or /session_id at the end to avoid replacing digits in product IDs
        normalized_url = re.sub(r'/' + re.escape(url_id) + r'(?=/|$)', '/' + SID_PLACEHOLDER, url)
        
        # Step 2: Replace search query with placeholder
        parts = normalized_url.split('/')
        
        # search_results URL: replace the search query segment (index 5) with placeholder
        if len(parts) > 4 and parts[3] == 'search_results':
            # parts: ['http:', '', 'host', 'search_results', '<session_id>', '<search_query>', '<page_num>']
            if len(parts) > 5:
                parts[5] = QUERY_PLACEHOLDER
        
        # item_page URL: replace the search query segment (index 6) with placeholder
        elif len(parts) > 5 and parts[3] == 'item_page':
            # parts: ['http:', '', 'host', 'item_page', '<session_id>', '<product_id>', '<search_query>', '<page_num>', '<options>']
            if len(parts) > 6:
                parts[6] = QUERY_PLACEHOLDER
        
        return '/'.join(parts)

    def get_available_actions(self):
        available_actions = self.env.get_available_actions()
        return available_actions

    def get_action_path(self):
        return self.action_path.copy()

    def get_experience(self):
        if self.st is None or self.prev_action is None or self.st1 is None:
            raise ValueError("[webshop_adaptor] In get_experience(), the history is not set, one of st0, a or st1 is None")
        experience = {
            "id": f"{get_timestamp_ms()}_{self.url_id}_{'-'.join(self.action_path)}",
            "action_path": self.get_action_path(),  # Use copy() to avoid reference issue
            "st": self.st,
            "action": self.prev_action,
            "st1": self.st1,
        }
        return experience

    # Setters
    def _set_instruction(self):
        """获取环境的 instruction，提取 "Instruction: " 之后的内容"""
        instruction_text = self.env.get_instruction_text()
        return instruction_text.split("Instruction: ", 1)[1].strip()

    # Consultants    
    def is_valid_action(self, action):
        action_status = self.get_available_actions()
        action = action.strip()
        # pattern: content[content]
        pattern = r"^[^\[\]\s]+?\[[^\[\]]+?\]$"
        if not bool(re.match(pattern, action)):
            return False
        action = action[:-1].split("[")
        action_type = action[0]
        action_content = action[1]
        if action_type == "search":
            if action_status["has_search_bar"]:
                return True
            return False
        elif action_type == "click":
            if action_content in action_status["clickables"]:
                return True
            return False
        raise ValueError(f"Unrecognized action: {action}")

    def is_finished_state(self, state):
        action_status = self.get_available_actions()
        if not action_status["has_search_bar"] and len(action_status["clickables"]) == 0:
            return True
        return False
    
    # Modifiers
    def initialize_env(self):
        if webshop_config['session'] is not None:
            self.env.reset(session=webshop_config['session'])
        else:
            self.env.reset()
        self.url_id = self.env.state['url'].split('/')[-1]
        self.instruction = self._set_instruction()
        # set new history
        self.st = None
        self.prev_action = None
        self.st1 = self.get_state()
        self.action_path = []

    def step(self, action):
        # record history
        self.st = self.st1
        self.prev_action = action
        self.action_path.append(action)
        # make history
        self.env.step(action)
        # observe history
        self.st1 = self.get_state()

    def format_action(self, action):
        return action.strip().lower()

    def extract_reward_score(self) -> float:
        html = self.env.state.get("html")
        match = re.search(r'<h3[^>]*id="reward".*?<pre>\s*([0-9.]+)\s*</pre>', html, flags=re.DOTALL)
        if not match:
            raise ValueError(f"Reward score not found in the HTML")
        return float(match.group(1))

    # Tobe implemented in the subclass
    def get_action_prompt(self, instruction, state, retrieved_experiences=None):
        raise NotImplementedError
