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
            enable_confirm_purchase = webshop_config['enable_confirm_purchase'],
        )
        self.reproduce_method = "action_path"
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
        - confirm_purchase: http://host/confirm_purchase/<session_id>/<product_id>/<search_query>/<page_num>/<options>
        - item_sub_page: http://host/item_sub_page/<session_id>/<product_id>/<search_query>/<page_num>/<sub_page>/<options>
        """
        # Step 1: Replace session_id with placeholder
        # Replace /session_id/ or /session_id at the end to avoid replacing digits in product IDs
        normalized_url = re.sub(r'/' + re.escape(url_id) + r'(?=/|$)', '/' + SID_PLACEHOLDER, url)
        
        # Step 2: Replace search query with placeholder
        parts = normalized_url.split('/')
        
        # search_results URL: query at index 5
        # parts: ['http:', '', 'host', 'search_results', '<session_id>', '<search_query>', '<page_num>']
        if len(parts) > 5 and parts[3] == 'search_results':
            parts[5] = QUERY_PLACEHOLDER
        
        # item_page, confirm_purchase, item_sub_page: query at index 6
        # parts: ['http:', '', 'host', '<page_type>', '<session_id>', '<product_id>', '<search_query>', ...]
        elif len(parts) > 6 and parts[3] in ('item_page', 'confirm_purchase', 'item_sub_page'):
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
            "reproduce_method": self.reproduce_method,
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
        """
        规范化模型输出；若缺少右括号（常见错误：`click[red`），自动补齐以避免因格式问题卡死。
        """
        action = action.strip().lower()
        if action and '[' in action and ']' not in action and action.count('[') == 1:
            action = action + ']'
        return action

    def extract_reward_score(self) -> float:
        html = self.env.state.get("html")
        match = re.search(r'<h3[^>]*id="reward".*?<pre>\s*([0-9.]+)\s*</pre>', html, flags=re.DOTALL)
        if not match:
            raise ValueError(f"Reward score not found in the HTML")
        return float(match.group(1))

    @staticmethod
    def check_finished_and_get_score(state: dict) -> tuple:
        """
        Check if the state is a finished state (done page) and extract the score if so.
        
        Args:
            state: A dict containing 'url' and 'html' keys
            
        Returns:
            A tuple of (is_finished, score):
            - is_finished: True if this is the done/finished page
            - score: The reward score if finished, None otherwise
        """
        url = state.get('url', '')
        
        # Check if this is a finished page by parsing URL structure
        # done URL: http://host/done/<session_id>/<product_id>/<options>
        # parts: ['http:', '', 'host', 'done', '<session_id>', '<product_id>', '<options>']
        parts = url.split('/')
        if len(parts) < 4 or parts[3] != 'done':
            return False, None
        
        # Extract score from HTML
        html = state.get('html', '')
        # Pattern: "Your score (min 0.0, max 1.0) <score>"
        match = re.search(r'Your score \(min 0\.0, max 1\.0\)\s*([0-9.]+)', html)
        if match:
            score = float(match.group(1))
            return True, score
        
        # If URL indicates done but score not found, still return finished but with None score
        raise ValueError(f"Score not found in the HTML")

    # Tobe implemented in the subclass
    def get_action_prompt(self, instruction, state, retrieved_experiences=None):
        raise NotImplementedError
