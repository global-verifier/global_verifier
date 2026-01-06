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
        
        # 收集路径行和对应的平均值行
        summary_items = []  # [(path_line, [avg_lines]), ...]
        
        for csv_path in csv_files:
            values = []
            with open(csv_path, "r", encoding="utf-8") as f:
                reader = csv.reader(f)
                next(reader, None)  # 跳过标题行
                for row in reader:
                    if row:
                        values.append(row[-1])
            
            path_line = f"【{csv_path}】 共 {len(values)} 条"
            out.write(f"\n{path_line}\n")
            out.write("-" * 60 + "\n")
            
            current_avg_lines = []
            # 按 items_per_row 分组输出
            for i in range(0, len(values), items_per_row):
                chunk = values[i:i + items_per_row]
                # 计算个数和平均值
                numeric_values = [float(v) for v in chunk]
                count = len(numeric_values)
                avg = sum(numeric_values) / count if count > 0 else 0
                avg_line = f"[{count}个, 平均: {avg:.4f}]"
                current_avg_lines.append(avg_line)
                out.write(f"{avg_line}\n")
                out.write(", ".join(chunk) + "\n")
            
            summary_items.append((path_line, current_avg_lines))
        
        # 最后汇总 - 保持原顺序
        out.write("\n" + "=" * 80 + "\n")
        out.write("【汇总】\n")
        out.write("-" * 60 + "\n")
        for path_line, avg_lines in summary_items:
            out.write(f"{path_line}\n")
            for avg_line in avg_lines:
                out.write(f"{avg_line}\n")
        
        out.write("\n" + "=" * 80 + "\n")
        out.write("提取完成!\n")
    
    print(f"结果已输出到: {output_path}")


if __name__ == "__main__":
    # 在 global_verifier 目录下运行
    extract_all_scores("./", items_per_row=20, output_file="scores_output.txt")
