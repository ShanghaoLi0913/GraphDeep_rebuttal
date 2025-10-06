#!/usr/bin/env python3
"""
通用JSONL文件导出工具
"""

import json
import pandas as pd
import sys
import os

def load_jsonl(file_path):
    """加载JSONL文件"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                try:
                    data_item = json.loads(line)
                    # 跳过配置行和统计行
                    if 'config' not in data_item and 'final_stats' not in data_item:
                        data.append(data_item)
                except json.JSONDecodeError:
                    continue
    return data

def export_single_file(input_file, output_file=None, sheet_name=None):
    """导出单个JSONL文件到Excel"""
    if not os.path.exists(input_file):
        print(f"❌ 文件不存在: {input_file}")
        return
    
    # 默认输出文件名
    if output_file is None:
        base_name = os.path.splitext(input_file)[0]
        output_file = f"{base_name}.xlsx"
    
    # 默认工作表名
    if sheet_name is None:
        sheet_name = os.path.splitext(os.path.basename(input_file))[0]
    
    # 加载数据
    data = load_jsonl(input_file)
    if not data:
        print(f"❌ 文件中没有有效数据: {input_file}")
        return
    
    # 转换为DataFrame
    df = pd.json_normalize(data)
    
    # 导出到Excel
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name=sheet_name, index=False)
    
    print(f"✅ 成功导出到 {output_file}")
    print(f"数据量: {len(df)} 个样本")
    print(f"列数: {len(df.columns)} 列")

def export_multiple_files():
    """导出多个推理结果文件"""
    # 文件路径
    files = [
        ('experiment_records/inference_results_20250624_173120_subgraph1.jsonl', '版本1_简单匹配'),
        ('experiment_records/inference_results_20250624_140443_subgraph2.jsonl', '版本2_精确匹配'),
        ('llm_reeval_v1.jsonl', 'LLM重评估_版本1'),
        ('llm_reeval_v2.jsonl', 'LLM重评估_版本2')
    ]
    
    output_file = 'all_results_comparison.xlsx'
    
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        for file_path, sheet_name in files:
            if os.path.exists(file_path):
                data = load_jsonl(file_path)
                if data:
                    df = pd.json_normalize(data)
                    df.to_excel(writer, sheet_name=sheet_name, index=False)
                    print(f"✅ 已添加 {sheet_name}: {len(df)} 个样本")
                else:
                    print(f"⚠️ 跳过空文件: {file_path}")
            else:
                print(f"⚠️ 文件不存在: {file_path}")
    
    print(f"✅ 成功导出所有结果到 {output_file}")

if __name__ == "__main__":
    if len(sys.argv) == 2:
        # 单文件模式
        input_file = sys.argv[1]
        export_single_file(input_file)
    elif len(sys.argv) == 3:
        # 单文件指定输出名
        input_file = sys.argv[1]
        output_file = sys.argv[2]
        export_single_file(input_file, output_file)
    else:
        # 默认模式：导出所有文件
        export_multiple_files() 