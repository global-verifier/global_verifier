#!/usr/bin/env python3
"""
从 log.md 中提取状态并总结成 s_t 格式
去除个性化数据和没用信息
"""

import re
import json
import ast
from state_processor import StateProcessor, process_state_for_experience


def extract_state_from_log_line(line: str) -> dict:
    """从log的一行中提取state字典"""
    # 尝试解析Python字典格式
    # 查找 {...} 格式的字典
    match = re.search(r'\{.*\}', line, re.DOTALL)
    if match:
        try:
            # 使用ast.literal_eval更安全
            state_dict = ast.literal_eval(match.group(0))
            return state_dict
        except:
            pass
    return None


def parse_log_file(log_path: str):
    """解析log.md文件，提取两个状态"""
    with open(log_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # 提取第1-4行的状态（第2行是state）
    state1 = None
    state2 = None
    
    # 第2行是第一个state
    if len(lines) >= 2:
        state1_dict = extract_state_from_log_line(lines[1])
        if state1_dict:
            state1 = state1_dict
    
    # 第26行后（第27行开始）是第二个state
    if len(lines) >= 27:
        state2_dict = extract_state_from_log_line(lines[26])
        if state2_dict:
            state2 = state2_dict
    
    return state1, state2


def state_to_summary_string(processed_state: dict, raw_state: dict = None) -> str:
    """
    将处理后的状态转换为简洁的s_t字符串表示
    
    Args:
        processed_state: 处理后的状态
        raw_state: 原始状态（用于提取instruction_text）
    """
    parts = []
    
    # 页面类型
    parts.append(f"PageType: {processed_state['page_type']}")
    
    # 关键元素
    elements = processed_state['key_elements']
    if elements.get('has_search_bar'):
        parts.append("HasSearchBar")
    
    clickables = elements.get('clickable_items', [])
    if clickables:
        # 只取关键的可点击项（前5个）
        click_str = ', '.join(clickables[:5])
        parts.append(f"Clickables: [{click_str}]")
    
    # 内容摘要
    summary = processed_state['content_summary']
    if summary.get('product_count', 0) > 0:
        parts.append(f"ProductCount: {summary['product_count']}")
    
    if summary.get('price_range'):
        pr = summary['price_range']
        parts.append(f"PriceRange: ${pr['min']:.2f}-${pr['max']:.2f}")
    
    if summary.get('key_text'):
        # 只取前2个关键文本，限制长度
        key_texts = summary['key_text'][:2]
        key_text_str = ' | '.join([t[:50] for t in key_texts])
        parts.append(f"KeyProducts: [{key_text_str}]")
    
    # URL模式（已去除个性化数据）
    if processed_state.get('url_pattern'):
        parts.append(f"URLPattern: {processed_state['url_pattern']}")
    
    return " | ".join(parts)


def main():
    log_path = 'log.md'
    
    print("=" * 80)
    print("从 log.md 提取状态")
    print("=" * 80)
    
    # 解析log文件
    state1_raw, state2_raw = parse_log_file(log_path)
    
    processor = StateProcessor()
    
    # 处理第一个状态（第5行前，即第1-4行的内容）
    print("\n【状态1 - 第5行前的内容】")
    print("-" * 80)
    if state1_raw:
        processed1 = processor.process_state(state1_raw)
        s_t1 = state_to_summary_string(processed1, state1_raw)
        
        print("原始状态（已去除个性化数据）:")
        print(json.dumps(processed1, indent=2, ensure_ascii=False))
        print("\n总结后的 s_t:")
        print(s_t1)
    else:
        print("未能提取状态1")
    
    # 处理第二个状态（第26行后的内容）
    print("\n【状态2 - 第26行后的内容】")
    print("-" * 80)
    if state2_raw:
        processed2 = processor.process_state(state2_raw)
        s_t2 = state_to_summary_string(processed2, state2_raw)
        
        print("原始状态（已去除个性化数据）:")
        print(json.dumps(processed2, indent=2, ensure_ascii=False))
        print("\n总结后的 s_t:")
        print(s_t2)
    else:
        print("未能提取状态2")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()

