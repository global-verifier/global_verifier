# CartPole Environment Usage Guide

本文档说明如何使用 CartPole 环境进行探索和经验学习。

## 📁 已创建的文件

### 1. 环境适配器
- `env_adaptors/cartPole_adaptor.py` - CartPole基础适配器
- `env_adaptors/cartPole_llama_adaptor.py` - LLaMA专用适配器（含Prompt工程）

### 2. 经验后端
- `exp_backend/cartPole_exp_backend.py` - CartPole经验后端基类
- `exp_backend/cartPole_exp_vanilla_backend.py` - Vanilla算法实现

### 3. 配置文件（已更新）
- `env_adaptors/env_config.py` - 添加了 `cartpole_config`
- `exp_backend/backend_config.py` - 添加了 `cartpole_vanilla_config`
- `plugin_loader.py` - 添加了 CartPole 加载器

### 4. 测试文件
- `playground_cartPole.ipynb` - Jupyter notebook测试环境

## 🚀 快速开始

### 步骤1：更新config.py

编辑 `config.py`，将环境设置为CartPole：

```python
explorer_settings = {
    "max_steps": 500,  # CartPole可以运行更久
    "max_action_retries": 3,
    "log_dir": "./log/",
    "use_experience": True,
    "save_experience": True,
    # 修改为CartPole
    "model_name": "llama3.1",
    "env_name": "cartpole_llama",           # ⭐ 改这里
    "backend_env": "cartpole-vanilla",      # ⭐ 改这里
    "storage_path": "./storage/exp_store.json",
    "depreiciate_exp_store_path": "./storage/depreiciate_exp_store.json",
}
```

### 步骤2：运行探索

#### 方法A：使用Python脚本

```python
from explorer import Explorer

# 创建Explorer实例
e = Explorer()

# 单次探索
e.explore()

# 多次探索建立经验库
for i in range(20):
    print(f"\nExploration {i+1}/20")
    e.explore()

# 清理经验
e.refine_experience()
```

#### 方法B：使用Jupyter Notebook

打开 `playground_cartPole.ipynb` 并运行cells。

#### 方法C：使用现有脚本

```bash
cd /home/xingkun/global_verifier
conda activate frozen_lake

# 修改 explore20times.py 的配置为 cartpole
python explore20times.py
```

## 🎮 CartPole环境特点

### 状态空间
CartPole的状态被离散化为以下维度：
- `x_bin`: 小车位置（0-7个bin）
- `theta_bin`: 杆角度（0-7个bin）
- `x_dot_sign`: 小车速度方向（-1/0/1）
- `theta_dot_sign`: 杆角速度方向（-1/0/1）

### 动作空间
- `0`: 向左推车
- `1`: 向右推车

### 终止条件
- 杆角度超过 ±12°
- 小车位置超过 ±2.4
- 达到最大步数（500步）

### 奖励
- 每保持平衡一步 +1分
- 目标：获得尽可能高的总分

## ⚙️ 自定义物理参数

在 `env_adaptors/env_config.py` 中修改 `cartpole_config`：

```python
cartpole_config = {
    "id": "CartPole-v1",
    "random_seed": 0,
    # 自定义物理参数
    "force_mag": 15.0,      # 推力（默认10.0）
    "gravity": 5.0,         # 重力（默认9.8）
    "masscart": 2.0,        # 小车质量（默认1.0）
    "masspole": 0.2,        # 杆质量（默认0.1）
    "length": 0.8,          # 杆半长（默认0.5）
    "tau": 0.01,            # 时间步（默认0.02）
}
```

## 📊 Prompt工程设计

### System Prompt
```
You are an intelligent control agent for the CartPole environment.
Your goal is to balance a pole on a moving cart...
```

### User Prompt包含：
1. **当前状态描述**
   - 小车位置（人类可读）
   - 杆角度（人类可读）
   - 速度方向

2. **历史经验展示**
   - 危险动作：导致失败的动作
   - 成功动作：保持平衡的动作

3. **任务指导**
   - 优先保持杆垂直
   - 防止小车到达边缘
   - 从经验中学习

## 📈 经验检索策略

### sameSt_1Step算法
- 检索条件：相同的离散化状态
- 返回：从该状态出发的所有1步转移经验

### 经验冲突解决
- 自动检测：相同状态+动作 → 不同结果
- 主动验证：重新执行经验
- 智能废弃：保留有效经验，废弃过时经验

## 🔍 状态离散化策略

为了实现经验复用，连续状态被离散化：

```python
# 位置bins：-2.4到2.4分成7段
x_bins = [-2.4, -1.2, -0.5, 0, 0.5, 1.2, 2.4]

# 角度bins：约±12°分成7段  
theta_bins = [-0.2095, -0.1, -0.05, 0, 0.05, 0.1, 0.2095]

# 速度简化为方向：-1/0/1
```

**优点**：
- 使得相似状态可以共享经验
- 减少经验存储空间

**缺点**：
- 损失精度
- 可能导致次优策略

## 📝 日志和分析

### 运行日志
```bash
log/explorerLog_<timestamp>.log  # 详细执行日志
log/explorer_summary.csv         # 运行结果汇总
```

### 经验存储
```bash
storage/exp_store.json           # 活跃经验库
storage/depreiciate_exp_store.json  # 废弃经验库
```

### CSV字段
- `timestamp`: 运行时间
- `model_name`: 模型名称（llama3.1）
- `env_name`: 环境名称（cartpole_llama）
- `instruction`: 任务指令
- `action_path`: 动作序列
- `step_count`: 步数
- `final_score`: 最终得分

## 🎯 预期结果

### 初始阶段（0-10次探索）
- 随机探索，快速失败
- 平均得分：10-30分
- 建立基础经验库

### 学习阶段（10-50次探索）
- 开始利用经验
- 平均得分：50-150分
- 经验库逐渐优化

### 成熟阶段（50+次探索）
- 稳定利用经验
- 平均得分：200-500分
- 接近最优策略

## 🐛 常见问题

### Q: 为什么一直得分很低？
A: 
- 检查max_steps是否足够（建议500）
- 检查状态离散化是否太粗糙
- 尝试清空经验库重新学习

### Q: 经验库增长太快？
A: 
- 定期运行 `e.refine_experience()` 清理
- 调整状态离散化粒度

### Q: 如何可视化？
A: 
- CartPole支持render_mode
- 可以在adaptor中添加渲染功能

## 🔗 与其他环境对比

| 特性 | FrozenLake | Webshop | CartPole |
|------|------------|---------|----------|
| 状态空间 | 离散 | 复杂（HTML） | 连续→离散化 |
| 动作空间 | 离散(4) | 结构化字符串 | 离散(2) |
| 挑战 | 避坑+路径规划 | 语义理解+导航 | 连续控制+平衡 |
| 经验复用 | 容易 | 中等（需标准化） | 中等（需离散化） |

## 📚 下一步

1. 运行多次探索建立经验库
2. 分析成功和失败的模式
3. 优化prompt以提高性能
4. 尝试不同的离散化策略
5. 对比有无经验的性能差异

