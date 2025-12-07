from .mountainCar_adaptor import MountainCarAdaptor
import re

LLAMA3_MOUNTAINCAR_SYSTEM_PROMPT = """You are an intelligent control agent for the Mountain Car environment. Your goal is to drive an underpowered car to reach the flag at the top of the right hill.

CRITICAL RULES:
1. The car's engine is TOO WEAK to climb the hill directly
2. You MUST build momentum by driving back and forth
3. Strategy: Go LEFT first to gain momentum, then accelerate RIGHT to reach the goal
4. Goal position is at 0.5 (top of right hill)
5. Learn from past experiences to find the optimal momentum-building strategy
6. Respond with only the action number (0, 1, or 2) without explanation"""

class MountainCarLlamaAdaptor(MountainCarAdaptor):
    def __init__(self, env_name):
        super().__init__(env_name)

    def get_action_prompt(self, retrieved_experiences=None):
        if retrieved_experiences is None:
            retrieved_experiences = []
            
        # Get current state information
        available_actions = self.get_available_actions()
        state = self.get_state()
        
        # Build state description
        position_desc = self._get_position_description(state['position_bin'], state['position'])
        velocity_desc = self._get_velocity_description(state['velocity_bin'], state['velocity'])
        progress_desc = self._get_progress_description(state['position'])
        
        user_prompt = f"""
Current Episode Steps: {self.episode_length}
Current Total Reward: {self.episode_reward}

Current State:
  Position: {position_desc} (bin {state['position_bin']}, actual: {state['position']:.3f})
  Velocity: {velocity_desc} (bin {state['velocity_bin']}, actual: {state['velocity']:.4f})
  Progress: {progress_desc}

Available Actions:
  0 = Push LEFT (accelerate towards left hill)
  1 = NO PUSH (coast with current momentum)
  2 = Push RIGHT (accelerate towards right hill/goal)

"""
        
        # Add historical experiences if available
        if retrieved_experiences:
            # Categorize experiences by outcome
            good_actions = []  # Actions that led to progress
            neutral_actions = []  # Actions with no significant change
            
            for exp in retrieved_experiences:
                action = exp.get('action', 'N/A')
                st = exp.get('st', {})
                st1 = exp.get('st1', {})
                
                # Analyze if action led to progress
                pos_before = st.get('position', 0)
                pos_after = st1.get('position', 0)
                vel_before = st.get('velocity', 0)
                vel_after = st1.get('velocity', 0)
                
                # Check if action improved situation (moved right or gained useful momentum)
                if self._is_good_transition(pos_before, pos_after, vel_before, vel_after):
                    good_actions.append((action, st, st1))
                else:
                    neutral_actions.append((action, st, st1))
            
            user_prompt += f"""\n--- Historical Experience from Similar States ---
Based on {len(retrieved_experiences)} previous attempt(s) from similar position/velocity:

"""
            
            if good_actions:
                user_prompt += f"""✅ EFFECTIVE ACTIONS (led to progress):
"""
                for action, st, st1 in good_actions[:3]:  # Show top 3
                    action_name = ["LEFT", "COAST", "RIGHT"][action]
                    pos_change = st1.get('position', 0) - st.get('position', 0)
                    vel_after = st1.get('velocity', 0)
                    user_prompt += f"""  Action {action} ({action_name}) → Position change: {pos_change:+.3f}, New velocity: {vel_after:+.4f}
"""
                user_prompt += f"""
💡 RECOMMENDED: Actions {set(a for a, _, _ in good_actions)} showed positive results from similar states.

"""
            
            if neutral_actions and not good_actions:
                # Only show neutral if no good actions found
                user_prompt += f"""⚪ NEUTRAL ACTIONS (no clear progress):
"""
                for action, st, st1 in neutral_actions[:3]:  # Show top 3
                    action_name = ["LEFT", "COAST", "RIGHT"][action]
                    pos_change = st1.get('position', 0) - st.get('position', 0)
                    user_prompt += f"""  Action {action} ({action_name}) → Position change: {pos_change:+.3f}
"""
                user_prompt += f"""
"""
            
            user_prompt += """Consider these experiences when deciding your next action.
---

"""

        user_prompt += """Task: Choose the BEST action to reach the goal.

🎯 STRATEGY GUIDANCE (think step-by-step):

1. ASSESS YOUR PROGRESS:
   - If you're VERY CLOSE to goal (upper right hill): Push RIGHT to finish!
   - If you're on the RIGHT SIDE of valley: Keep pushing RIGHT to maintain progress!
   
2. UNDERSTAND MOMENTUM:
   - If moving RIGHT with good speed: Push RIGHT to maintain/amplify momentum!
   - If moving LEFT with good speed: Continue LEFT to build potential energy on left hill
   
3. KNOW WHEN TO SWITCH:
   - If you've reached FAR LEFT (on left hill): Time to SWITCH to RIGHT and use that energy!
   - If you're stuck near VALLEY BOTTOM with little movement: Push LEFT first to gain momentum
   
4. GENERAL PRINCIPLE:
   - The car needs to swing LEFT to gain energy, then use that energy to climb RIGHT
   - Once you have rightward momentum, KEEP IT and push RIGHT towards goal
   - Don't randomly switch - commit to a direction until you've built momentum

⚠️ CRITICAL: The swing strategy means going LEFT first, but you MUST eventually switch to RIGHT!
Don't get stuck oscillating randomly or staying left forever.

Respond with only ONE action number (0, 1, or 2).
"""
        
        # Construct the prompt in Llama3 format
        prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
{LLAMA3_MOUNTAINCAR_SYSTEM_PROMPT} 
<|eot_id|>
<|start_header_id|>user<|end_header_id|>
{user_prompt}
<|eot_id|> 
<|start_header_id|>assistant<|end_header_id|>
"""
        return prompt

    def _get_position_description(self, pos_bin, position):
        """Convert position bin to human-readable description."""
        descriptions = [
            "FAR LEFT (left edge)",
            "LEFT hill (lower)",
            "LEFT hill (middle)",
            "VALLEY bottom (deepest point)",
            "VALLEY (right side)",
            "RIGHT hill (lower slope)",
            "RIGHT hill (middle slope)",
            "RIGHT hill (upper slope)",
            "NEAR GOAL (almost there!)",
            "AT GOAL (flag reached!)",
            "BEYOND GOAL"
        ]
        desc = descriptions[min(pos_bin, len(descriptions)-1)]
        
        # Add distance to goal info
        distance_to_goal = 0.5 - position
        if distance_to_goal > 0:
            desc += f" [{distance_to_goal:.2f} to goal]"
        else:
            desc += " [GOAL REACHED!]"
        
        return desc
    
    def _get_velocity_description(self, vel_bin, velocity):
        """Convert velocity bin to human-readable description."""
        if velocity < -0.04:
            return "Moving LEFT (fast)"
        elif velocity < -0.01:
            return "Moving LEFT (moderate)"
        elif velocity < 0:
            return "Moving LEFT (slow)"
        elif velocity == 0:
            return "STATIONARY (no momentum)"
        elif velocity < 0.01:
            return "Moving RIGHT (slow)"
        elif velocity < 0.04:
            return "Moving RIGHT (moderate)"
        else:
            return "Moving RIGHT (fast)"
    
    def _get_progress_description(self, position):
        """Describe overall progress towards goal."""
        if position >= 0.5:
            return "🎯 SUCCESS! Goal reached!"
        elif position >= 0.4:
            return "🔥 Very close! One more push!"
        elif position >= 0.3:
            return "📈 Making good progress"
        elif position >= 0:
            return "➡️ Right side of valley"
        elif position >= -0.5:
            return "⬅️ Left side of valley"
        else:
            return "🏔️ On left hill"
    
    def _is_good_transition(self, pos_before, pos_after, vel_before, vel_after):
        """
        Check if a state transition represents good progress.
        Good progress = moving towards goal or building useful momentum.
        """
        # Reached goal
        if pos_after >= 0.5:
            return True
        
        # Made progress towards goal (moved right)
        if pos_after > pos_before and pos_after > -0.3:
            return True
        
        # Building momentum on left side (good strategy)
        if pos_before < -0.5 and abs(vel_after) > abs(vel_before):
            return True
        
        # Strong rightward velocity
        if vel_after > 0.03:
            return True
        
        return False
    
    def format_action(self, action):
        """
        Format the action from LLM output to valid action integer.
        Extracts exactly one digit (0, 1, or 2) from the response.
        """
        action = action.strip()
        # Find all numbers that are 0, 1, or 2
        matches = re.findall(r'[0-2]', action)
        
        if len(matches) == 0:
            raise ValueError(f"Could not extract valid action (0, 1, or 2) from: {action}")
        elif len(matches) > 1:
            raise ValueError(f"Multiple actions found ({matches}) in: {action}")
        else:
            return int(matches[0])

