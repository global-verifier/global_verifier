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

