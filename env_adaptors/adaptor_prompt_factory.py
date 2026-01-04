import json
import urllib.parse

# -- System Prompt --
WEBSHOP_SYSTEM_PROMPT = "You are an intelligent exploration agent navigating a web shop. Your goal is to understand the task instruction and buy the correct product with the highest possible score of (1.0). Respond with only the action you want to execute, without any additional explanation or formatting."
FROZENLAKE_SYSTEM_PROMPT = "You are an intelligent exploration agent navigating a frozen lake. Your goal is to reach the destination with highest possible score while avoiding holes. Analyze the current position and decide the next move. Respond with only the action number (0, 1, 2, or 3) without any additional explanation or formatting."
MOUNTAINCAR_SYSTEM_PROMPT = """You are an intelligent control agent for the Mountain Car environment.

ENVIRONMENT:
- The car is in a valley between two hills
- Position range: -1.2 (left edge) to 0.6 (right edge)
- Valley bottom (lowest point): position -0.5
- Goal: reach position 0.5 (top of right hill)
- The car starts near the valley bottom

PHYSICS HINT:
- The car may need to build momentum before climbing uphill
- Going uphill slows the car down, going downhill speeds it up

RULES:
1. Learn from past experiences to find the optimal strategy
2. Respond with only the action number (0, 1, or 2) without explanation"""

# -- MountainCar Helper Functions --
def mountaincar_get_position_description(position: float) -> str:
    """Convert position value to human-readable description (matches old adaptor methods)."""
    # Position ranges from -1.2 (far left) to 0.6 (beyond goal)
    # Goal is at 0.5
    if position >= 0.5:
        desc = "AT GOAL (flag reached!)"
    elif position >= 0.45:
        desc = "NEAR GOAL (almost there!)"
    elif position >= 0.3:
        desc = "RIGHT hill (upper slope)"
    elif position >= 0.1:
        desc = "RIGHT hill (middle slope)"
    elif position >= -0.1:
        desc = "RIGHT hill (lower slope)"
    elif position >= -0.3:
        desc = "VALLEY (right side)"
    elif position >= -0.5:
        desc = "VALLEY bottom (deepest point)"
    elif position >= -0.7:
        desc = "LEFT hill (middle)"
    elif position >= -0.9:
        desc = "LEFT hill (lower)"
    else:
        desc = "FAR LEFT (left edge)"

    # Add distance to goal info
    distance_to_goal = 0.5 - position
    if distance_to_goal > 0:
        desc += f" [{distance_to_goal:.2f} to goal]"
    else:
        desc += " [GOAL REACHED!]"

    return desc


def mountaincar_get_velocity_description(velocity: float) -> str:
    """Convert velocity value to human-readable description (matches old adaptor methods)."""
    if velocity < -0.04:
        return "Moving LEFT (fast)"
    elif velocity < -0.01:
        return "Moving LEFT (moderate)"
    elif velocity < -0.002:
        return "Moving LEFT (slow)"
    elif velocity <= 0.002:
        return "STATIONARY (no momentum)"
    elif velocity < 0.01:
        return "Moving RIGHT (slow)"
    elif velocity < 0.04:
        return "Moving RIGHT (moderate)"
    else:
        return "Moving RIGHT (fast)"


def mountaincar_get_progress_description(position: float) -> str:
    """Describe overall progress towards goal (matches old adaptor methods)."""
    if position >= 0.5:
        return "SUCCESS - Goal reached"
    elif position >= 0.4:
        return "Very close - One more push"
    elif position >= 0.3:
        return "Making good progress"
    elif position >= 0:
        return "Right side of valley"
    elif position >= -0.5:
        return "Left side of valley"
    else:
        return "On left hill"

# -- MountainCar User Prompt --
def build_mountaincar_user_prompt(
    *,
    episode_length: int,
    episode_reward: float,
    state: dict,
    retrieved_experiences=None,
) -> str:
    """Build MountainCar user prompt content (without model-specific wrappers)."""
    if retrieved_experiences is None:
        retrieved_experiences = []

    position = state["position"]
    velocity = state["velocity"]
    position_desc = mountaincar_get_position_description(position)
    velocity_desc = mountaincar_get_velocity_description(velocity)
    progress_desc = mountaincar_get_progress_description(position)

    user_prompt = f"""
Current Episode Steps: {episode_length}
Current Total Reward: {episode_reward}

Current State:
  Position: {position_desc} (value: {position:.3f})
  Velocity: {velocity_desc} (value: {velocity:.4f})
  Progress: {progress_desc}

Available Actions:
  0 = Push LEFT (accelerate towards left hill)
  1 = NO PUSH (coast with current momentum)
  2 = Push RIGHT (accelerate towards right hill/goal)

"""

    if retrieved_experiences:
        user_prompt += f"""\n--- Historical Experience from This Exact State ---
You have been at this state before. Here are {len(retrieved_experiences)} previous experience(s):

"""
        for idx, exp in enumerate(retrieved_experiences, 1):
            action = exp.get('action', 'N/A')
            st1 = exp.get('st1', {})
            reachable = exp.get('reachable', None)
            path_length = exp.get('path_length', None)
            summary = exp.get('voyager_summary')
            gen_score = exp.get('generative_score')
            probability = exp.get('probability')

            user_prompt += f"""Experience {idx}:
  Action taken: {action}
  Result state: pos={st1.get('position', 0):.3f}, vel={st1.get('velocity', 0):.4f}
  Can reach goal: {reachable if reachable is not None else 'Unknown'}
  Steps to goal: {path_length if path_length is not None else 'Unknown'}
"""
            if probability is not None:
                user_prompt += f"""  Probability: {probability:.2f}
"""
            user_prompt += "\n"
            if summary:
                user_prompt += f"""  Summary for this step is: {summary}
"""
            if gen_score is not None:
                user_prompt += f"""  LLM analyzed score for this action is: {gen_score}
"""

        reachable_exps = [e for e in retrieved_experiences if e.get('reachable', False)]
        if reachable_exps:
            best_action = reachable_exps[0]['action']
            user_prompt += f"""If an action can reach the goal, you MUST choose that action.
Action {best_action} can reach the goal. You MUST choose {best_action}.

"""
        else:
            user_prompt += """Choose the action that may lead to the goal.

"""

    user_prompt += """Respond with only ONE action number (0, 1, or 2).
"""
    return user_prompt

# -- Webshop User Prompt --
def build_webshop_user_prompt(
    state: dict,
    instruction: str,
    action_status: dict,
    action_path,
    retrieved_experiences=None,
    show_failed_result: bool = False,
) -> str:
    """Build the user prompt content (without ChatML wrappers).

    This function is intentionally self-free so it can be reused and tested easily.
    """
    if retrieved_experiences is None:
        retrieved_experiences = []

    # Construct the user prompt
    is_search = action_status["has_search_bar"]
    if is_search:
        available_actions = "[search]"
    else:
        available_actions = "[click]"

    user_prompt = f"""Task Instruction: {instruction}

Current URL: {state['url']}

Current Webpage Display Text: {state['html']}

Available Actions: {available_actions}

"""
    if not is_search:
        user_prompt += f"""Clickables: {action_status['clickables']}\n"""
        if "buy now" in action_status['clickables']:
            user_prompt += f"""If you think you have meet the requirement, you can click the 'buy now' button to buy the product."""
        user_prompt += f"""You can only click one button at a time."""

    # Add current episode's action history to avoid repeating ineffective actions
    if action_path:
        user_prompt += f"""
Actions you have already taken in this episode: {action_path}
Try not repeat action already in the action path.
Do NOT repeat actions that didn't change the state. If you already clicked an option, try a different one.
"""

    # Add historical experiences if available
    if retrieved_experiences:
        user_prompt += f"""\n--- Historical Experience from Similar States ---
You have visited this state before. Here are {len(retrieved_experiences)} previous experience(s):

"""
        for idx, exp in enumerate(retrieved_experiences, 1):
            action_taken = exp.get('action', 'N/A')
            next_url = exp.get('st1', {}).get('url', 'N/A')
            max_score = exp.get('max_score', None)
            summary = exp.get('voyager_summary')
            gen_score = exp.get('generative_score')
            probability = exp.get('probability')
            user_prompt += f"""Experience {idx}:
  Action taken: {action_taken}
  Result URL: {next_url}
"""
            if probability is not None:
                user_prompt += f"""  Probability: {probability:.2f}
"""
            # 修改逻辑开始
            if max_score is None or max_score == 0:
                if show_failed_result:
                    user_prompt += f"""  Result: Score 0.0 (Failed). \n"""
                else:
                    user_prompt += f""" \n"""
            elif max_score == 1:
                user_prompt += f"""  Result: Score 1.0 (PERFECT).
  Max score this action can achieve is 1.0. You SHOULD take the same action: {action_taken}.

"""
            else:
                # 针对 0.75 等非满分情况
                user_prompt += f"""  Result: Score {max_score} (SUBOPTIMAL).
  Max score is {max_score}, which is NOT 1.0. This means '{action_taken}' is NOT the best choice (e.g. wrong color or size).
  Do NOT repeat '{action_taken}'. You MUST select a different option to aim for score 1.0!

"""
            # 修改逻辑结束

            if summary:
                user_prompt += f"""  Summary for this step is: {summary}
"""
            if gen_score is not None:
                user_prompt += f"""  LLM analyzed score for this action is: {gen_score}
"""

        if "confirm_purchase" in str(state.get("url", "")):
            blacklisted_actions = []
            for exp in retrieved_experiences:
                max_score = exp.get("max_score", None)
                action = exp.get("action", None)
                if max_score in (0, 0.0) and isinstance(action, str) and action.startswith("click["):
                    if action not in blacklisted_actions:
                        blacklisted_actions.append(action)
            if blacklisted_actions:
                user_prompt += """IMPORTANT: BLACKLISTED ACTIONS
The following actions have been confirmed to result in a score of 0.0 in this specific state. You MUST NOT select them:

"""
                for action in blacklisted_actions:
                    user_prompt += f"""{action} (Score: 0.0) Please choose a different action from the remaining Clickables to explore potential success.
"""
                user_prompt += "\n"

    # 对 item_page 做选项黑名单，避免重复选择已选项
    if "item_page" in str(state.get("url", "")):
        selected_option_blacklist = []
        selected_part = str(state.get("url", "")).rsplit("/", 1)[-1]
        decoded_part = urllib.parse.unquote(selected_part)
        selected_dict = json.loads(decoded_part)
        assert isinstance(selected_dict, dict), "selected_dict is not a dict"
        for value in selected_dict.values():
            action = f"click[{str(value).lower()}]"
            if action not in selected_option_blacklist:
                selected_option_blacklist.append(action)
        if selected_option_blacklist:
            user_prompt += """IMPORTANT: CURRENTLY SELECTED OPTIONS
The following options are already selected on this item page. Do NOT click them again; pick a different option to progress:

"""
            for action in selected_option_blacklist:
                user_prompt += f"""{action} (already selected). Choose another option from Clickables.
"""
            user_prompt += "\n"

    # 增强结尾的指令
    user_prompt += """IMPORTANT STRATEGY:
1. SCORE 1.0 IS THE ONLY GOAL.
2. If a previous action got 1.0 -> REPEAT IT.
3. If a previous action got anything less than 1.0 (e.g. 0.5, 0.75) -> IT IS WRONG. DO NOT REPEAT IT. CHOOSE A DIFFERENT OPTION.
---

"""

    user_prompt += """You goal is to buy the most suitable product that satisfies the task instruction and get the maximum score (1.0).
Please consider selecitng required options before buying.
Based on the current state and task instruction, what is the next action you should take?

If you want to search, response with "search[<search_query>]"
If you want to click, response follow the format: "click[name of the clickable]"
"""
    return user_prompt


# -- FrozenLake User Prompt --
def build_frozenlake_user_prompt(
    *,
    state: dict,
    available_actions,
    destinations,
    goal_rewards,
    map_rows: int,
    map_cols: int,
    retrieved_experiences=None,
) -> str:
    """Build FrozenLake user prompt content (without model-specific wrappers)."""
    if retrieved_experiences is None:
        retrieved_experiences = []

    cur_pos = state["cur_pos"]
    tile_type = state["tile_type"]

    destinations_str = ", ".join([f"({r}, {c})" for r, c in destinations])

    # Build per-destination guidance: how many steps to move in each direction
    # from current position to reach each destination.
    dest_detail_lines = []
    cur_r, cur_c = cur_pos
    for i, (r, c) in enumerate(destinations, start=1):
        up_steps = max(cur_r - r, 0)
        down_steps = max(r - cur_r, 0)
        left_steps = max(cur_c - c, 0)
        right_steps = max(c - cur_c, 0)
        moves = []
        if up_steps > 0:
            moves.append(f"{up_steps} line up")
        if down_steps > 0:
            moves.append(f"{down_steps} line down")
        if left_steps > 0:
            moves.append(f"{left_steps} column left")
        if right_steps > 0:
            moves.append(f"{right_steps} column right")

        move_str = ", ".join(moves) + " step(s)" if moves else "Already at destination (0 moves)."
        assert len(goal_rewards) > 0
        if len(goal_rewards) == 1:
            dest_detail_lines.append(
                f"- Destination {i}: ({r}, {c})."
            )
        else:
            dest_detail_lines.append(
                f"- Destination {i}: ({r}, {c}). It is {move_str} compare to current position ({cur_r}, {cur_c}). "
                f"You do NOT have to go directly; always avoid holes. Explore more and take detours to find a safe way around."
            )
    destinations_detail_str = "\n".join(dest_detail_lines) if dest_detail_lines else "- (none)"

    user_prompt = f"""
Map Size: {map_rows} rows x {map_cols} columns
Current Position: {cur_pos}

There are a total of {len(destinations)} destination(s).
Destinations: {destinations_str}
For each destination, the steps below describe how many moves are needed in each direction from your CURRENT position:
{destinations_detail_str}

The highest possible score achievable from these destinations is 1.0.

Current Tile Type: {tile_type} (S=Start, F=Frozen, H=Hole, G=Goal)

Available Actions: {available_actions}
  0 = Move Left (Column - 1)
  1 = Move Down (Row + 1)
  2 = Move Right (Column + 1)
  3 = Move Up (Row - 1)

"""

    # Add historical experiences if available
    if retrieved_experiences:
        user_prompt += f"""\n--- Historical Experience from Similar States ---
You have been at this position before. Here are {len(retrieved_experiences)} previous experience(s):

"""
        forbidden_actions = []
        goal_actions = []

        for exp in retrieved_experiences:
            action_taken = exp.get('action', 'N/A')
            next_pos = exp.get('st1', {}).get('cur_pos', 'N/A')
            next_tile = exp.get('st1', {}).get('tile_type', 'N/A')
            achieved_score = exp.get('st1', {}).get('score', 0)
            summary = exp.get('voyager_summary')
            max_score = exp.get('max_score')
            gen_score = exp.get('generative_score')
            probability = exp.get('probability')

            tile_warning = ""
            if next_tile == 'H':
                tile_warning = f" HOLE - AVOID ACTION {action_taken}!"
                forbidden_actions.append(action_taken)
            elif next_tile == 'G':
                if achieved_score >= 1.0:
                    tile_warning = f" MAX REWARD GOAL (Score: {achieved_score}) - TAKE ACTION {action_taken}!"
                    goal_actions.append(action_taken)
                else:
                    tile_warning = f" SUB-OPTIMAL GOAL (Score: {achieved_score}) - Can you find a better one?"

            user_prompt += f"""  Action taken: {action_taken}
  Result Position: {next_pos}
  Result Tile: {next_tile}{tile_warning}
"""
            if probability is not None:
                user_prompt += f"""  Probability: {probability:.2f}
"""
            if max_score is not None and max_score > 0:
                user_prompt += f"""  Max score achievable from this path: {max_score}
"""
            if summary:
                user_prompt += f"""  Summary for this step is: {summary}
"""
            if gen_score is not None:
                user_prompt += f"""  LLM analyzed score for this action is: {gen_score}
"""
            user_prompt += "\n"

        if forbidden_actions or goal_actions:
            user_prompt += "\n"
        if goal_actions:
            user_prompt += f"""!!! BEST CHOICE: Action(s) {goal_actions} will reach the HIGHEST SCORE GOAL directly. Choose this!

"""
        if forbidden_actions:
            forbidden_list = ", ".join(map(str, dict.fromkeys(forbidden_actions)))
            user_prompt += f"""!!! CRITICAL WARNING: Action(s) {forbidden_list} lead to HOLES and will END THE GAME.
YOU MUST NOT choose {forbidden_list} from this position!

"""
  
        user_prompt += """Consider these experiences when deciding your next action.
---

"""


    assert len(goal_rewards) > 0
    if len(goal_rewards) == 1:
        user_prompt += """Based on the current position, task instruction, and past experiences, what is the next action you should take?

REMEMBER: If certain actions lead to holes, you MUST avoid them. Choose the safest action that moves toward the destination.

Respond with only the action number (0, 1, 2, or 3).
"""
    else:
        user_prompt += """Based on the current position and the destination coordinates:
1. CALCULATE the difference between Current Position and Destinations (Row difference and Column difference).
2. IDENTIFY which action reduces this difference (e.g., if Goal Row > Current Row, you need to increase Row).
3. DO NOT simply repeat a "safe" action if it moves you AWAY from the destination. Explore unexplored areas.
4. DO NOT satisfy with low scores (e.g. 0.5). Aim for the HIGHEST SCORE (1.0). Explore unexplored areas to search for best score.
5. AVOID actions strictly forbidden (leading to Holes). Never take actions that lead to holes.

Which action effectively moves you closer to the HIGHEST SCORE goal? Respond with only the action number (0, 1, 2, or 3).
"""

    return user_prompt
