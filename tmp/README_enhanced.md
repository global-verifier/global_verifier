# Global Verifier

用于在 WebShop 环境上对比不同语言模型性能的前端 Agent 系统。

## 功能

- ✅ 支持多个模型（Qwen2.5, Llama3, Qwen3）
- ✅ 在 WebShop 环境上测试
- ⏳ 自动对比和报告生成
- ⏳ 详细的性能指标

## 快速开始

```bash
# 1. 配置模型路径
vim config.py

# 2. 运行对比实验
python main.py --models qwen2.5 llama3 qwen3 --num_episodes 100

# 3. 查看结果
cat results/comparison.json
```

## 项目结构

```
gloal_verifier/
├── config.py              # 配置文件
├── main.py                # 主入口
├── agent/                 # Agent 实现
├── evaluator/             # 评估逻辑
└── results/               # 结果输出
```

## 使用示例

```python
from evaluator import ExperimentRunner
from config import Config

# 创建实验
runner = ExperimentRunner(Config)

# 运行对比
runner.compare_models(['qwen2.5', 'llama3'])

# 查看报告
runner.print_report()
```

## 配置说明

在 `config.py` 中配置：
- 模型路径
- 环境参数
- 评估参数

## 依赖

- transformers
- torch
- webshop (环境)
- wandb (可选，用于可视化)

## 更新日志

### v0.1 (开发中)
- 基础模型加载
- WebShop 环境集成
- 初步评估框架
