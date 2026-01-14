# gloal_verifier

## simple terminal test run of webshop
```
cd webshop
conda activate webshop

import gym
from web_agent_site.envs import WebAgentTextEnv
env = gym.make('WebAgentTextEnv-v0', observation_mode='text', num_products=100000)

env.reset()
env.state
env.step("search[jacket]")
env.step("click[search]")
```

- WebShop 依赖 `spacy` + 语言模型 `en_core_web_sm`，否则会报 `ModuleNotFoundError: No module named 'spacy'`
  - 安装（在你的 conda env 里执行）：

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

- webshop不支持链接直接navigate，看来需要硬点回访方式了
    - 记录一个完整的路径

## Status 设计
- 保存为 str 存储
- 要求是
    - 去personal token
    - 去掉没用的data
    - 保证不同agent，从同一地点出发，执行同样操作所到的state记录一致
### webshop
- {
    "available_options": "{'has_search_bar': False, 'clickables': ['back to search', 'next >', 'b09kp78g37', 'b098xt346y', 'b09r7h66fc', 'b09glvmlms', 'b09qqp3356', 'b00zdedvbi', 'b07v3wxx85', 'b09s3bn15c', 'b08dxl22jn', 'b07b9qwjw9']}",
    "state": "{
        "url":"",  # 去掉 personal token
        "html": 
    }",
    ...
}

# 数据在这里：
https://huggingface.co/datasets/YWZBrandon/webshop-data/blob/main/items_shuffle_1000.jsonr

https://github.com/troyyxk/WebShop

# 环境
# 方法 1：使用 conda 环境文件（推荐）
## 1. 复制 environment_webshop_update_py310.yml 到新机器
## 2. 创建环境
conda env create -f environment_webshop_update_py310.yml
## 3. 激活环境
conda activate webshop_update_py310

# 方法 2：使用 pip requirements（跨平台更兼容）
## 1. 创建新的 conda 环境
conda create -n webshop_update_py310 python=3.10.19 openjdk=17 -y
conda activate webshop_update_py310
# 2. 安装 pip 依赖
pip install -r requirements_webshop_update_py310.txt
## 3. 下载 spacy 模型
python -m spacy download en_core_web_sm

# 方法 3：精简版安装（只装核心依赖）
## 创建环境
conda create -n webshop_update_py310 python=3.10 openjdk=17 -y
conda activate webshop_update_py310
## 安装核心依赖
pip install flask rich pyserini==0.17.0 gym spacy thefuzz beautifulsoup4
python -m spacy download en_core_web_sm
