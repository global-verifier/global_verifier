# 记忆增强后端集成指南

这个文档介绍了如何在 global_verifier 项目中使用来自 GMemory 的三种记忆策略。

## 概述

我们已经将 GMemory 中的三个 benchmark 集成到了 global_verifier 的后端系统中：

1. **MemoryBank** - 基于遗忘机制的记忆管理（来自 `memorybank.py`）
2. **Generative** - 基于 LLM 评分的经验检索（来自 `generative.py`）
3. **Voyager** - 基于向量相似度的基准方法（来自 `voyager.py`）

每个策略都支持所有环境（CartPole、MountainCar、Webshop）。

## 快速开始

### 基本使用

```python
from exp_backend import create_backend

# 创建带记忆增强的后端
backend = create_backend(
    env_name="CartPole",           # 环境名称
    memory_strategy="memorybank"   # 记忆策略: memorybank, generative, voyager
)

# 存储经验（需要 label 字段）
experience = {
    "id": "exp_001",
    "action_path": [0, 1, 0, 1],
    "st": {"cart_position": 0.0, "cart_velocity": 0.0},
    "action": 1,
    "st1": {"cart_position": 0.1, "cart_velocity": 0.05},
    "label": True  # True=成功, False=失败
}
backend.store_experience(experience)

# 检索相似经验
query_state = {"cart_position": 0.0, "cart_velocity": 0.0}
successful_exps, failed_exps = backend.retrieve_similar_experiences(
    query_state,
    successful_topk=3,  # 返回3个成功经验
    failed_topk=1       # 返回1个失败经验
)

print(f"找到 {len(successful_exps)} 个相似的成功经验")
print(f"找到 {len(failed_exps)} 个相似的失败经验")
```

## 三种策略对比

### 1. MemoryBank 策略

**适用场景**：最近的经验更有价值的环境

**特点**：
- 基于时间的遗忘机制
- 旧经验会按指数衰减
- 自动内存管理

**示例**：
```python
backend = create_backend(
    env_name="CartPole",
    memory_strategy="memorybank",
    forgetting_threshold=0.3  # 遗忘阈值（0-1）
)

# 存储经验时会自动添加时间戳
for exp in experiences:
    backend.store_experience(exp)

# 查看内存统计
stats = backend.get_memory_statistics()
print(f"活跃经验: {stats['active_experiences']}")
print(f"已遗忘经验: {stats['forgotten_experiences']}")
print(f"平均重要性: {stats['average_importance']:.3f}")
```

### 2. Generative 策略

**适用场景**：需要上下文理解，简单相似度不够的环境

**特点**：
- 使用 LLM 评分经验相关性
- 上下文感知的经验选择
- 需要提供 LLM 模型

**示例**：
```python
# 定义你的 LLM 接口
def my_llm_model(messages, temperature=0.1):
    # messages: [{"role": "system"/"user", "content": "..."}]
    # 返回: 字符串响应
    # 这里可以调用 OpenAI API、本地模型等
    pass

backend = create_backend(
    env_name="MountainCar",
    memory_strategy="generative",
    llm_model=my_llm_model
)

# 使用 LLM 评分检索
successful_exps, failed_exps = backend.retrieve_similar_experiences(
    query_state,
    successful_topk=3,
    use_llm_scoring=True,      # 启用 LLM 评分
    score_multiplier=2          # 检索候选数量的倍数
)
```

### 3. Voyager 策略（基准）

**适用场景**：简单环境或作为对比基准

**特点**：
- 纯向量相似度检索
- 快速高效
- 可选的 LLM 总结

**示例**：
```python
backend = create_backend(
    env_name="Webshop",
    memory_strategy="voyager",
    use_summarization=False  # 设为 True 则使用 LLM 总结
)

# 简单检索（无额外机制）
successful_exps, failed_exps = backend.retrieve_similar_experiences(
    query_state,
    successful_topk=5
)

# 获取统计信息
stats = backend.get_statistics()
print(f"成功率: {stats['success_rate']:.2%}")
```

## 与原始后端的对比

| 特性 | 原始后端 | 记忆增强后端 |
|------|---------|------------|
| 存储格式 | 仅 JSON | JSON + 向量数据库 |
| 检索方法 | 精确状态匹配 | 语义相似度 |
| 经验排序 | 无 | 根据策略排序 |
| 内存管理 | 手动 | 自动（MemoryBank）|
| LLM 集成 | 无 | 可选（Generative）|
| 冲突解决 | ✓ | ✓（继承）|

## 项目结构

```
exp_backend/
├── memory_enhanced_backend.py          # 记忆增强基类
├── memorybank_backend.py               # MemoryBank 策略
├── generative_backend.py               # Generative 策略
├── voyager_backend.py                  # Voyager 策略
├── cartPole_memory_backends.py         # CartPole 专用实现
├── mountainCar_memory_backends.py      # MountainCar 专用实现
├── webshop_memory_backends.py          # Webshop 专用实现
├── backend_factory.py                  # 工厂函数
└── MEMORY_BACKEND_README.md            # 详细英文文档
```

## 运行示例

### 1. 运行演示程序

```bash
cd global_verifier
python examples/memory_backend_demo.py
```

这个脚本会演示：
- MemoryBank 的遗忘机制
- Generative 的 LLM 评分
- Voyager 的基准检索
- 三种策略的对比

### 2. 性能对比

```bash
python examples/compare_strategies.py --env CartPole --num-experiences 100 --num-queries 20
```

这会测试并对比：
- 存储性能（经验/秒）
- 检索性能（查询/秒）
- 内存使用情况

## 安装依赖

```bash
# 基础依赖
pip install langchain-chroma chromadb sentence-transformers

# 如果需要特定的 embedding 模型
pip install transformers torch
```

## 从原始后端迁移

```python
# 旧代码
from exp_backend.cartPole_exp_backend import CartPoleExpBackend
backend = CartPoleExpBackend("CartPole", "./storage/exp_store.json", "./storage/deprecated.json")

# 新代码（保持原有行为）
from exp_backend import create_backend
backend = create_backend("CartPole", "none")  # "none" = 原始行为

# 或者使用记忆增强
backend = create_backend("CartPole", "memorybank")
```

### API 兼容性

所有记忆增强后端都保持与 `BaseExpBackend` 的 API 兼容：

- ✓ `store_experience(exp)` - 兼容
- ✓ `get_exp_by_id(exp_id)` - 兼容
- ✓ `_deprecate_experience(exp_id)` - 兼容
- ✓ `resolve_experience_conflict(...)` - 兼容

新增方法：
- `retrieve_similar_experiences(query_state, ...)` - 新功能

## 实际应用示例

### 示例 1: CartPole 任务中使用 MemoryBank

```python
from exp_backend import create_backend

# 创建后端
backend = create_backend("CartPole", "memorybank", forgetting_threshold=0.3)

# 在训练循环中使用
for episode in range(num_episodes):
    state = env.reset()
    done = False
    actions = []
    
    while not done:
        # 检索相似经验来指导决策
        similar_success, similar_fail = backend.retrieve_similar_experiences(
            state, successful_topk=3, failed_topk=1
        )
        
        # 根据经验做决策
        action = select_action(state, similar_success, similar_fail)
        next_state, reward, done, _ = env.step(action)
        
        actions.append(action)
        state = next_state
    
    # 存储这次 episode 的经验
    exp = {
        "id": f"episode_{episode}",
        "action_path": actions,
        "st": initial_state,
        "action": actions[0],
        "st1": final_state,
        "label": reward > threshold  # 成功或失败
    }
    backend.store_experience(exp)
```

### 示例 2: 对比不同策略的效果

```python
from exp_backend import create_backend

strategies = ["memorybank", "generative", "voyager"]
results = {}

for strategy in strategies:
    backend = create_backend("MountainCar", strategy, llm_model=my_llm)
    
    # 运行测试
    success_rate = run_test_episodes(backend, num_episodes=50)
    results[strategy] = success_rate
    
    print(f"{strategy}: {success_rate:.2%} 成功率")

# 选择最佳策略
best_strategy = max(results, key=results.get)
print(f"最佳策略: {best_strategy}")
```

## 常见问题

### 1. 如何选择合适的策略？

- **MemoryBank**：环境变化较快，旧经验很快过时
- **Generative**：需要深度理解上下文，有可用的 LLM
- **Voyager**：简单环境，或作为基准对比

### 2. 向量数据库占用多少空间？

- 每 10,000 个经验大约占用 100-500MB
- 数据存储在磁盘上，可以持久化

### 3. 检索速度如何？

- Voyager 最快（纯相似度）
- MemoryBank 中等（相似度 + 过滤）
- Generative 较慢（需要 LLM 评分）

### 4. 能否自定义 embedding 函数？

可以：

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-mpnet-base-v2')

def custom_embedding(texts):
    return model.encode(texts)

backend = create_backend(
    "CartPole",
    "voyager",
    embedding_func=custom_embedding
)
```

## 参考资料

- 详细英文文档: `exp_backend/MEMORY_BACKEND_README.md`
- GMemory 原始实现:
  - MemoryBank: `GMemory/mas/memory/mas_memory/memorybank.py`
  - Generative: `GMemory/mas/memory/mas_memory/generative.py`
  - Voyager: `GMemory/mas/memory/mas_memory/voyager.py`

## 技术支持

如果遇到问题：

1. 查看详细日志: `log/exp_backendLog_*.log`
2. 运行示例程序验证安装
3. 查看详细英文文档中的 Troubleshooting 部分

---

集成完成时间: 2025-12-18
集成版本: v1.0
