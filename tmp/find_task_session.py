#!/usr/bin/env python3
"""
查找特定 instruction 对应的 session 索引

用法:
    python find_task_session.py "instruction text" [num_products]
"""

import sys
import json
import random
from collections import defaultdict
from os.path import dirname, abspath, join

# 设置路径
BASE_DIR = dirname(abspath(__file__))
WEBSHOP_DIR = join(BASE_DIR, '../webshop')
DATA_DIR = join(WEBSHOP_DIR, 'data')

# 设置随机种子（与代码中一致）
random.seed(233)

PRICE_RANGE = [10.0 * i for i in range(1, 100)]

def get_human_goals_simple(all_products, product_prices):
    """简化版的 get_human_goals，不依赖其他模块"""
    goals = []
    cnt_atts = defaultdict(int)
    cnt = 0
    
    # 创建产品字典
    product_dict = {p['asin']: p for p in all_products if 'asin' in p}
    
    # 加载人类标注任务
    human_attr_path = join(DATA_DIR, 'items_human_ins.json')
    with open(human_attr_path, 'r') as f:
        human_attributes = json.load(f)
    
    for asin, instructions in human_attributes.items():
        if asin not in product_dict:
            continue
        item = product_dict[asin]
        
        for product in instructions:
            attributes = product.get('instruction_attributes', [])
            if len(attributes) == 0:
                cnt += 1
                continue

            # 价格处理
            price = product_prices.get(asin, 50.0)
            price_range = [p for p in PRICE_RANGE if p > price][:4]
            if len(price_range) >= 2:
                _, price_upper = sorted(random.sample(price_range, 2))
                price_text = f', and price lower than {price_upper:.2f} dollars'
            else:
                price_upper = 1000000
                price_text = ''

            goal = {
                'asin': asin,
                'category': item.get('category', ''),
                'query': item.get('query', ''),
                'name': item.get('name', ''),
                'product_category': item.get('product_category', ''),
                'instruction_text': product['instruction'].strip('.') + price_text,
                'attributes': attributes,
                'price_upper': price_upper,
                'goal_options': product.get('instruction_options', []),
            }
            goals.append(goal)
            for att in attributes:
                cnt_atts[att] += 1
    
    for goal in goals:
        goal['weight'] = 1
    
    return goals


def find_task_session(instruction_text, num_products=None):
    """
    查找特定 instruction 对应的 session 索引
    
    Args:
        instruction_text: 要查找的 instruction 文本（不包含价格部分）
        num_products: 产品数量限制（默认 None，即不限制）
    
    Returns:
        (session_index, goal_dict) 或 (None, None)
    """
    # 加载产品数据
    products_path = join(DATA_DIR, 'items_shuffle_1000.json')
    print(f"Loading products from {products_path}...")
    with open(products_path, 'r') as f:
        products = json.load(f)
    
    # 限制产品数量
    if num_products is not None:
        products = products[:num_products]
    
    # 创建产品字典和价格字典（简化版，使用默认价格）
    product_dict = {p['asin']: p for p in products if 'asin' in p and p['asin'] != 'nan' and len(p['asin']) <= 10}
    product_prices = {asin: 50.0 for asin in product_dict.keys()}
    
    # 生成任务列表
    print("Generating goals...")
    goals = get_human_goals_simple(list(product_dict.values()), product_prices)
    
    # 打乱任务顺序（使用固定种子）
    random.shuffle(goals)
    
    print(f"Total goals: {len(goals)}")
    
    # 搜索目标 instruction
    target_instruction = instruction_text.strip().lower()
    
    for idx, goal in enumerate(goals):
        instruction = goal.get('instruction_text', '')
        # 移除价格部分进行比较
        instruction_base = instruction.split(', and price lower than')[0].strip().lower()
        
        if instruction_base == target_instruction:
            return idx, goal
    
    # 如果精确匹配失败，尝试模糊匹配
    print("\nExact match not found. Searching for similar instructions...")
    best_match = None
    best_score = 0
    for idx, goal in enumerate(goals):
        instruction = goal.get('instruction_text', '')
        instruction_lower = instruction.lower()
        # 检查关键词语
        target_keywords = [kw for kw in target_instruction.split() if len(kw) > 3]
        matches = sum(1 for kw in target_keywords if kw in instruction_lower)
        score = matches / len(target_keywords) if target_keywords else 0
        if score > best_score:
            best_score = score
            best_match = (idx, goal)
    
    if best_score > 0.5:
        return best_match
    
    return None, None


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python find_task_session.py \"instruction text\" [num_products]")
        print("\nExample:")
        print('  python find_task_session.py "i am looking for x-large, red color women faux fur lined winter warm jacket coat" 100000')
        sys.exit(1)
    
    instruction = sys.argv[1]
    num_products = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2].isdigit() else None
    
    print(f"Searching for instruction: {instruction}")
    print(f"Parameters: num_products={num_products}\n")
    
    idx, goal = find_task_session(instruction, num_products)
    
    if idx is not None and goal is not None:
        print(f"\n{'='*60}")
        print(f"Found task at session index: {idx}")
        print(f"{'='*60}")
        print(f"Full instruction_text: {goal.get('instruction_text', '')}")
        print(f"ASIN: {goal.get('asin', '')}")
        print(f"Category: {goal.get('category', '')}")
        print(f"Query: {goal.get('query', '')}")
        print(f"Attributes: {goal.get('attributes', [])}")
        print(f"Goal options: {goal.get('goal_options', [])}")
        print(f"Price upper limit: ${goal.get('price_upper', 0):.2f}")
        print(f"\n{'='*60}")
        print(f"To access this task, use:")
        print(f"  env.reset(session={idx})")
        print(f"  or")
        print(f"  env.reset(session_int={idx})")
        print(f"{'='*60}")
    else:
        print("\nTask not found!")
        sys.exit(1)
