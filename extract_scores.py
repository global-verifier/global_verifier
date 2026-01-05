#!/usr/bin/env python3
"""
一次性提取所有 explorer_summary.csv 的最后一列，每 20 个为一行输出到文件。
"""

import csv
from pathlib import Path


def extract_all_scores(base_dir: str = ".", items_per_row: int = 20, output_file: str = "scores_output.txt") -> None:
    """遍历所有子目录，提取 explorer_summary.csv 的最后一列，输出到文件。"""
    base_path = Path(base_dir)
    
    # 查找所有 explorer_summary.csv 文件
    csv_files = sorted(base_path.glob("**/explorer_summary.csv"))
    
    if not csv_files:
        print("未找到任何 explorer_summary.csv 文件")
        return
    
    output_path = base_path / output_file
    
    with open(output_path, "w", encoding="utf-8") as out:
        out.write(f"找到 {len(csv_files)} 个 CSV 文件\n\n")
        out.write("=" * 80 + "\n")
        
        for csv_path in csv_files:
            values = []
            with open(csv_path, "r", encoding="utf-8") as f:
                reader = csv.reader(f)
                next(reader, None)  # 跳过标题行
                for row in reader:
                    if row:
                        values.append(row[-1])
            
            out.write(f"\n【{csv_path}】 共 {len(values)} 条\n")
            out.write("-" * 60 + "\n")
            
            # 按 items_per_row 分组输出
            for i in range(0, len(values), items_per_row):
                chunk = values[i:i + items_per_row]
                out.write(", ".join(chunk) + "\n")
        
        out.write("\n" + "=" * 80 + "\n")
        out.write("提取完成!\n")
    
    print(f"结果已输出到: {output_path}")


if __name__ == "__main__":
    # 在 global_verifier 目录下运行
    extract_all_scores("/home/xingkun/global_verifier", items_per_row=20, output_file="scores_output.txt")
