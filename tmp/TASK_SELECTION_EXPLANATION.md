# WebShop 任务存储与选择机制说明

## 概述

每次调用 `env.reset()` 时，WebShop 环境会随机选择一个新任务（goal），并生成对应的 instruction。本文档详细说明任务的存储位置、加载方式和选择机制。

## 1. 任务存储位置

任务数据存储在以下 JSON 文件中：

### 1.1 人类标注任务（human_goals=True）
- **文件路径**: `/home/xingkun/webshop/data/items_human_ins.json`
- **数据格式**: 
  ```json
  {
    "ASIN码": [
      {
        "asin": "ASIN码",
        "instruction": "任务指令文本",
        "attributes": ["属性1", "属性2"],
        "options": ["color: blue"],
        "instruction_attributes": ["属性列表"],
        "instruction_options": ["选项列表"],
        "assignment_id": "...",
        "worker_id": "..."
      },
      ...
    ]
  }
  ```
- **特点**: 每个产品可能有多个不同的 instruction

### 1.2 合成任务（human_goals=False）
- **产品数据**: `/home/xingkun/webshop/data/items_shuffle_1000.json`
- **任务属性**: `/home/xingkun/webshop/data/items_ins_v2_1000.json`
- **数据格式**:
  ```json
  {
    "ASIN码": {
      "attributes": ["属性列表"],
      "instruction": "任务指令文本",
      "instruction_attributes": ["属性列表"]
    }
  }
  ```
- **特点**: 每个产品有一个基础 instruction，系统会为所有选项组合生成多个任务变体

## 2. 任务加载流程

### 2.1 初始化阶段（环境创建时）

**代码位置**: `web_agent_site/envs/web_agent_text_env.py`

```python
# 1. 创建 SimServer 时加载任务
self.server = SimServer(
    self.base_url,
    self.file_path,  # 默认: items_shuffle_1000.json
    ...
)
```

**SimServer.__init__** (`web_agent_site/envs/web_agent_text_env.py:276-335`):
```python
# 1. 加载产品数据
self.all_products, self.product_item_dict, self.product_prices, _ = \
    load_products(filepath=file_path, num_products=num_products, human_goals=human_goals)

# 2. 从产品数据生成任务列表
self.goals = get_goals(self.all_products, self.product_prices, human_goals)

# 3. 打乱任务顺序（固定随机种子）
random.seed(233)
random.shuffle(self.goals)

# 4. 计算权重（用于加权随机选择）
self.weights = [goal['weight'] for goal in self.goals]
self.cum_weights = [0] + np.cumsum(self.weights).tolist()
```

### 2.2 数据加载细节

**load_products** (`web_agent_site/engine/engine.py:230-362`):
- 从 `items_shuffle_1000.json` 加载产品基本信息
- 从 `items_ins_v2_1000.json` 加载任务属性（synthetic goals）
- 从 `items_human_ins.json` 加载人类标注任务（human goals）
- 将 instruction 数据合并到产品对象中：
  - `human_goals=True`: `products[i]['instructions'] = human_attributes[asin]`
  - `human_goals=False`: `products[i]['instruction_text'] = attributes[asin].get('instruction')`

**get_goals** (`web_agent_site/engine/goal.py:16-127`):
- **get_human_goals**: 遍历所有产品的 `instructions` 列表，为每个 instruction 创建一个 goal
- **get_synthetic_goals**: 为每个产品的所有选项组合生成多个 goal 变体

## 3. 任务选择机制

### 3.1 Reset 时的任务选择

**代码位置**: `web_agent_site/envs/web_agent_text_env.py:240-260`

```python
def reset(self, session=None, instruction_text=None):
    # 1. 创建新的 session ID
    if session is not None:
        self.session = str(session)
    else:
        self.session = ''.join(random.choices(string.ascii_lowercase, k=10))
    
    # 2. 访问初始化 URL，触发任务选择
    init_url = f'{self.base_url}/{self.session}'
    self.browser.get(init_url, session_id=self.session, session_int=session_int)
    
    # 3. 获取选中的 instruction
    self.instruction_text = self.get_instruction_text()
```

### 3.2 实际选择逻辑

**代码位置**: `web_agent_site/envs/web_agent_text_env.py:504-521`

在 `SimServer.receive()` 方法中，当检测到新 session 时：

```python
def receive(self, session_id, current_url, session_int=None, **kwargs):
    # 如果 session 不存在，选择一个新任务
    if session_id not in self.user_sessions:
        # 方式1: 如果提供了 session_int，使用指定的索引
        if session_int is not None and isinstance(session_int, int):
            idx = session_int
        # 方式2: 否则使用加权随机选择
        else:
            idx = random_idx(self.cum_weights)  # 基于权重的随机选择
        
        # 获取选中的 goal
        goal = self.goals[idx]
        instruction_text = goal['instruction_text']
        
        # 存储到 session 中
        self.user_sessions[session_id] = {
            'goal': goal,
            'done': False
        }
    else:
        # 如果 session 已存在，使用之前选中的 goal
        instruction_text = self.user_sessions[session_id]['goal']['instruction_text']
```

### 3.3 随机选择算法

**代码位置**: `web_agent_site/utils.py:20-28`

```python
def random_idx(cum_weights):
    """基于累积权重的加权随机选择"""
    pos = random.uniform(0, cum_weights[-1])
    idx = bisect.bisect(cum_weights, pos)
    idx = min(idx, len(cum_weights) - 2)
    return idx
```

**权重计算**:
- **Human goals**: 所有任务的权重都是 1（均匀分布）
- **Synthetic goals**: 权重 = `sum(1. / cnt_atts[att] for att in goal['attributes']) / len(goal['attributes'])`
  - 属性出现频率越低，权重越高（稀有属性任务更容易被选中）

## 4. 关键代码位置总结

| 功能 | 文件路径 | 关键函数/方法 |
|------|---------|--------------|
| 任务数据文件 | `/home/xingkun/webshop/data/` | `items_human_ins.json`, `items_ins_v2_1000.json` |
| 产品数据加载 | `web_agent_site/engine/engine.py` | `load_products()` |
| 任务生成 | `web_agent_site/engine/goal.py` | `get_goals()`, `get_human_goals()`, `get_synthetic_goals()` |
| 任务选择 | `web_agent_site/envs/web_agent_text_env.py` | `SimServer.receive()`, `SimServer.__init__()` |
| Reset 方法 | `web_agent_site/envs/web_agent_text_env.py` | `WebAgentTextEnv.reset()` |
| 随机选择算法 | `web_agent_site/utils.py` | `random_idx()` |

## 5. 使用示例

### 5.1 默认随机选择任务
```python
env = gym.make('WebAgentTextEnv-v0', observation_mode='text', num_products=100000)
obs, info = env.reset()  # 随机选择一个任务
print(env.instruction_text)  # 查看选中的 instruction
```

### 5.2 指定任务索引
```python
env = gym.make('WebAgentTextEnv-v0', observation_mode='text', num_products=100000)
obs, info = env.reset(session=0)  # 使用 session=0 作为索引选择任务
```

### 5.3 指定 instruction 文本
```python
env = gym.make('WebAgentTextEnv-v0', observation_mode='text', num_products=100000)
obs, info = env.reset(instruction_text="Find me a blue jacket")  # 直接指定任务
```

## 6. 注意事项

1. **任务打乱**: 所有任务在初始化时使用固定随机种子（233）打乱，确保可复现性
2. **Session 管理**: 每个 session 对应一个任务，同一个 session ID 会复用相同的任务
3. **权重选择**: Synthetic goals 使用加权随机，稀有属性任务更容易被选中
4. **任务数量**: 实际可用任务数量取决于：
   - `num_products`: 限制产品数量
   - `limit_goals`: 限制任务数量（如果设置）
   - `filter_goals`: 自定义过滤函数（如果设置）

