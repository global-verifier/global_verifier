"""
模型对比实验实现示例
参考 WebShop 的设计模式
"""
import json
from dataclasses import dataclass
from typing import List, Dict
import time

# ==================== 1. 使用配置类 ====================
@dataclass
class ModelConfig:
    path: str
    device: str = "cuda"
    dtype: str = "float16"
    
# ==================== 2. 统一接口 ====================
class BaseAgent:
    """所有 agent 的基础接口"""
    def predict(self, state: str, available_actions: List[str]) -> str:
        """预测下一步动作"""
        raise NotImplementedError
    
    def reset(self):
        """重置 agent 状态"""
        pass

# ==================== 3. 实验管理器 ====================
class ExperimentManager:
    """管理对比实验 - 参考 WebShop 的 SimServer 设计"""
    def __init__(self, models: Dict[str, BaseAgent], env):
        self.models = models
        self.env = env
        self.results = []
        
    def run_comparison(self, test_goals: List[Dict], num_episodes: int = 100):
        """运行对比实验"""
        print(f"开始对比实验，共 {len(self.models)} 个模型")
        
        for model_name, agent in self.models.items():
            print(f"\n测试模型: {model_name}")
            model_results = self._test_model(agent, test_goals, num_episodes)
            self.results.append({
                "model": model_name,
                "results": model_results,
                "stats": self._calculate_stats(model_results)
            })
    
    def _test_model(self, agent, test_goals, num_episodes):
        """测试单个模型 - 参考 WebShop 的 test.py"""
        results = []
        for i in range(num_episodes):
            episode_result = self._run_episode(agent, test_goals[i])
            results.append(episode_result)
            
            # 定期输出进度
            if (i + 1) % 10 == 0:
                avg_reward = sum(r['reward'] for r in results) / len(results)
                print(f"  Episode {i+1}/{num_episodes}, Avg Reward: {avg_reward:.2f}")
        
        return results
    
    def _run_episode(self, agent, goal):
        """运行单个 episode"""
        obs, info = self.env.reset()
        total_reward = 0
        actions = []
        
        for step in range(100):  # max steps
            available_actions = info['valid']
            action = agent.predict(obs, available_actions)
            actions.append(action)
            
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            
            if done:
                break
        
        return {
            "reward": reward,  # 最终 reward
            "steps": len(actions),
            "actions": actions,
            "goal": goal
        }
    
    def _calculate_stats(self, results):
        """计算统计信息 - 参考 WebShop 的测试统计"""
        rewards = [r['reward'] for r in results]
        steps = [r['steps'] for r in results]
        
        success_rate = sum(1 for r in rewards if r == 1.0) / len(rewards)
        
        return {
            "avg_reward": sum(rewards) / len(rewards),
            "success_rate": success_rate,
            "avg_steps": sum(steps) / len(steps),
            "max_reward": max(rewards),
            "min_reward": min(rewards)
        }
    
    def print_comparison(self):
        """打印对比结果"""
        print("\n" + "=" * 80)
        print("模型对比结果")
        print("=" * 80)
        
        # 表头
        print(f"{'Model':<20} {'Avg Reward':<15} {'Success Rate':<15} {'Avg Steps':<15}")
        print("-" * 80)
        
        # 数据行
        for result in self.results:
            stats = result['stats']
            print(f"{result['model']:<20} "
                  f"{stats['avg_reward']:<15.2f} "
                  f"{stats['success_rate']*100:<14.1f}% "
                  f"{stats['avg_steps']:<15.1f}")
        
        print("=" * 80)
    
    def save_results(self, output_path: str):
        """保存结果到文件"""
        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        print(f"\n结果已保存到: {output_path}")

# ==================== 4. 使用示例 ====================
def main():
    """主函数 - 参考 WebShop 的 test.py"""
    
    # 1. 加载环境
    from webshop_utils import create_webshop_env
    env = create_webshop_env(observation_mode='text_rich')
    
    # 2. 加载模型并创建 agents
    models = {
        "Qwen2.5-7B": load_qwen_agent("qwen2.5"),
        "Llama3-8B": load_llama_agent("llama3"),
    }
    
    # 3. 创建实验管理器
    manager = ExperimentManager(models, env)
    
    # 4. 运行实验
    test_goals = get_test_goals()  # 获取测试目标
    manager.run_comparison(test_goals, num_episodes=100)
    
    # 5. 打印并保存结果
    manager.print_comparison()
    manager.save_results("results/comparison.json")

if __name__ == "__main__":
    main()

# ==================== 5. 设计要点 ====================
"""
关键设计思路（来自 WebShop 分析）：

1. 分离关注点
   - Agent: 只负责决策
   - Env: 只负责环境交互
   - Manager: 负责协调和记录

2. 可扩展性
   - 新增模型只需实现 BaseAgent
   - 配置外部化（配置文件）

3. 可观测性
   - 详细的日志记录
   - 中间结果保存
   - 进度实时显示

4. 可复现性
   - 固定随机种子
   - 保存完整结果
   - 清晰的数据流

5. 参考 WebShop 的优秀实践
   - Session 管理（分离不同实验）
   - 配置外部化（config.py）
   - 清晰的接口定义（gym 标准）
   - 详细的统计信息
"""
