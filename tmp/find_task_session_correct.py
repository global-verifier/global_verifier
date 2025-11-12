#!/usr/bin/env python3
"""
正确查找特定 instruction 对应的 session 索引

根据 playground 输出，实际环境加载了 6910 goals，说明使用的是 synthetic goals (human_goals=False)
但目标 instruction 在 human_goals 中，所以需要：
1. 如果使用 human_goals=True，查找对应的索引
2. 如果使用 human_goals=False，这个 instruction 可能不存在
"""

import sys
import json
import random
from os.path import dirname, abspath, join

BASE_DIR = dirname(abspath(__file__))
WEBSHOP_DIR = join(BASE_DIR, '../webshop')
DATA_DIR = join(WEBSHOP_DIR, 'data')

# 设置随机种子（与代码中一致）
random.seed(233)

PRICE_RANGE = [10.0 * i for i in range(1, 100)]

def find_in_human_goals(target_instruction):
    """在 human_goals 中查找"""
    print("="*70)
    print("查找 human_goals=True 模式下的任务索引")
    print("="*70)
    
    # 加载产品数据
    with open(join(DATA_DIR, 'items_shuffle_1000.json'), 'r') as f:
        products = json.load(f)
    
    product_dict = {p['asin']: p for p in products if 'asin' in p and p['asin'] != 'nan' and len(p['asin']) <= 10}
    product_prices = {asin: 50.0 for asin in product_dict.keys()}
    
    # 加载 human goals
    with open(join(DATA_DIR, 'items_human_ins.json'), 'r') as f:
        human_attrs = json.load(f)
    
    goals = []
    for asin, instructions in human_attrs.items():
        if asin not in product_dict:
            continue
        item = product_dict[asin]
        for product in instructions:
            attributes = product.get('instruction_attributes', [])
            if len(attributes) == 0:
                continue
            
            price = product_prices.get(asin, 50.0)
            price_range = [p for p in PRICE_RANGE if p > price][:4]
            if len(price_range) >= 2:
                _, price_upper = sorted(random.sample(price_range, 2))
                price_text = f', and price lower than {price_upper:.2f} dollars'
            else:
                price_text = ''
            
            goal = {
                'asin': asin,
                'category': item.get('category', ''),
                'query': item.get('query', ''),
                'instruction_text': product['instruction'].strip('.') + price_text,
                'attributes': attributes,
                'goal_options': product.get('instruction_options', []),
            }
            goals.append(goal)
    
    for goal in goals:
        goal['weight'] = 1
    
    # 打乱
    random.shuffle(goals)
    
    print(f"Total human goals: {len(goals)}")
    
    # 搜索
    target = target_instruction.strip().lower()
    for idx, goal in enumerate(goals):
        instruction = goal.get('instruction_text', '')
        instruction_base = instruction.split(', and price lower than')[0].strip().lower()
        if instruction_base == target:
            return idx, goal
    
    return None, None


if __name__ == '__main__':
    target = "i am looking for x-large, red color women faux fur lined winter warm jacket coat"
    
    print(f"目标 instruction: {target}\n")
    
    idx, goal = find_in_human_goals(target)
    
    if idx is not None and goal is not None:
        print(f"\n{'='*70}")
        print(f"找到任务！Session 索引: {idx}")
        print(f"{'='*70}")
        print(f"完整 instruction_text: {goal.get('instruction_text', '')}")
        print(f"ASIN: {goal.get('asin', '')}")
        print(f"Category: {goal.get('category', '')}")
        print(f"Query: {goal.get('query', '')}")
        print(f"Attributes: {goal.get('attributes', [])}")
        print(f"Goal options: {goal.get('goal_options', [])}")
        print(f"\n{'='*70}")
        print("使用方法:")
        print(f"{'='*70}")
        print("1. 创建环境时传入 human_goals=1:")
        print("   env = gym.make('WebAgentTextEnv-v0', observation_mode='text', num_products=100000, human_goals=1)")
        print(f"\n2. 然后使用:")
        print(f"   env.reset(session={idx})")
        print(f"   或")
        print(f"   env.reset(session_int={idx})")
        print(f"\n3. 在 Explorer 中使用:")
        print(f"   e.adaptor.env.reset(session={idx})")
        print(f"   或")
        print(f"   e.adaptor.env.reset(session_int={idx})")
        print(f"{'='*70}")
    else:
        print("\n未找到匹配的任务！")
        sys.exit(1)

