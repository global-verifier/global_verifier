# 任务 Session 使用说明

## 修改内容

### 1. 修改了 `env_adaptors/env_config.py`

添加了 `human_goals` 配置项：

```python
webshop_config = {
    "id": "WebAgentTextEnv-v0",
    "observation_mode": "text",
    "num_products": 100000,
    "human_goals": 1,  # 使用人类标注的任务 (1=True, 0=False)
}
```

### 2. 修改了 `env_adaptors/webshop_adaptor.py`

更新了环境创建逻辑，支持传递 `human_goals` 参数：

```python
def __init__(self, env_name):
    super().__init__(env_name)
    # 构建环境参数，支持 human_goals 配置
    env_kwargs = {
        'observation_mode': webshop_config['observation_mode'],
        'num_products': webshop_config['num_products'],
    }
    if 'human_goals' in webshop_config:
        env_kwargs['human_goals'] = webshop_config['human_goals']
    self.env = gym.make(webshop_config['id'], **env_kwargs)
    self.url_id = None
    self.instruction = None
```

## 目标任务信息

**Instruction**: `i am looking for x-large, red color women faux fur lined winter warm jacket coat`

**Session 索引**: `1` (在 `human_goals=True` 模式下)

**任务详情**:
- **ASIN**: `B09KP78G37`
- **完整 Instruction**: `i am looking for x-large, red color women faux fur lined winter warm jacket coat, and price lower than 70.00 dollars`
- **Category**: `fashion`
- **Query**: `Women's Coats, Jackets & Vests`
- **Attributes**: `['winter warm']`
- **Goal Options**: `['red', 'x-large']`
- **价格上限**: `$70.00`

## 使用方法

### 方法 1: 在 Explorer 中使用

```python
from explorer import Explorer

# 创建 Explorer（现在会自动使用 human_goals=1）
e = Explorer("llama3.1", "webshop_llama")

# 使用 session=1 来访问目标任务
e.adaptor.env.reset(session=1)
# 或
e.adaptor.env.reset(session_int=1)

# 查看当前 instruction
print(e.adaptor.get_instruction())
```

### 方法 2: 直接使用环境

```python
import gym
from web_agent_site.envs import WebAgentTextEnv

# 创建环境（使用 human_goals=1）
env = gym.make('WebAgentTextEnv-v0', 
               observation_mode='text', 
               num_products=100000,
               human_goals=1)

# 重置到目标任务
obs, info = env.reset(session=1)
# 或
obs, info = env.reset(session_int=1)

# 查看 instruction
instruction = env.get_instruction_text()
print(instruction)
```

## 验证修改

运行以下命令验证配置是否正确：

```python
from explorer import Explorer

e = Explorer("llama3.1", "webshop_llama")

# 检查加载的 goals 数量（应该是 13，而不是 6910）
print(f"Total goals: {len(e.adaptor.env.server.goals)}")

# 重置到目标任务
e.adaptor.env.reset(session=1)

# 验证 instruction
instruction = e.adaptor.get_instruction()
print(f"Current instruction: {instruction}")

# 应该输出类似：
# i am looking for x-large, red color women faux fur lined winter warm jacket coat, and price lower than 70.00 dollars
```

## 注意事项

1. **Goals 数量变化**:
   - `human_goals=1`: 加载 13 个人类标注的任务
   - `human_goals=0`: 加载约 6910 个合成任务（默认）

2. **Session 索引**:
   - Session 索引是基于固定随机种子（233）打乱后的顺序
   - 在 `human_goals=True` 模式下，session=1 对应目标任务
   - 在 `human_goals=False` 模式下，这个 instruction 不存在

3. **切换模式**:
   - 如果需要切换回 synthetic goals，修改 `env_config.py` 中的 `"human_goals": 0`
   - 或者删除 `human_goals` 配置项（默认为 0）

## 查找其他任务的 Session

使用 `find_task_session_correct.py` 脚本查找其他任务的 session 索引：

```bash
python find_task_session_correct.py "your instruction text here"
```

