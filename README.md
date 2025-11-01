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
```

- webshop不支持链接直接navigate，看来需要硬点回访方式了
    - 记录一个完整的路径

