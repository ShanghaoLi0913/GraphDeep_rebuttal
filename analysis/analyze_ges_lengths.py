#!/usr/bin/env python3
"""
分析 GES 文件中 gold triples 和 GES 的长度分布
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from pathlib import Path

def analyze_ges_lengths(file_path):
    """
    分析 GES 文件中的长度分布
    
    Args:
        file_path: GES 文件路径
    """
    print(f"正在分析文件: {file_path}")
    
    gold_lengths = []
    ges_lengths = []
    expansion_ratios = []
    semantic_improvements = []
    
    # 读取文件并分析
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
                
            try:
                data = json.loads(line)
                
                # 跳过 batch_stats 和 final_stats
                if 'batch_stats' in data or 'final_stats' in data:
                    continue
                
                # 获取 gold triples 长度
                gold_triples = data.get('gold_triples', [])
                gold_len = len(gold_triples)
                gold_lengths.append(gold_len)
                
                # 获取 GES 相关信息
                gold_expansion_set = data.get('gold_expansion_set', [])
                if gold_expansion_set:
                    ges_len = len(gold_expansion_set)
                    ges_lengths.append(ges_len)
                    
                    # 计算扩展比例
                    if gold_len > 0:
                        expansion_ratio = ges_len / gold_len
                        expansion_ratios.append(expansion_ratio)
                    
                # 获取质量指标（如果存在）
                ges_quality = data.get('ges_quality_metrics', {})
                if ges_quality:
                    semantic_imp = ges_quality.get('semantic_improvement', 0)
                    semantic_improvements.append(semantic_imp)
                
            except json.JSONDecodeError:
                print(f"Warning: 跳过无效JSON行 {line_num}")
                continue
            except Exception as e:
                print(f"Warning: 处理行 {line_num} 时出错: {e}")
                continue
    
    # 统计信息
    total_samples = len(gold_lengths)
    print(f"\n=== 统计结果 ===")
    print(f"总样本数: {total_samples:,}")
    
    if not gold_lengths:
        print("未找到有效数据")
        return
    
    # Gold Triples 长度分析
    print(f"\n=== Gold Triples 长度分析 ===")
    print(f"平均长度: {np.mean(gold_lengths):.2f}")
    print(f"中位数长度: {np.median(gold_lengths):.0f}")
    print(f"最小长度: {min(gold_lengths)}")
    print(f"最大长度: {max(gold_lengths)}")
    print(f"标准差: {np.std(gold_lengths):.2f}")
    
    # GES 长度分析
    if ges_lengths:
        print(f"\n=== GES 长度分析 ===")
        print(f"平均长度: {np.mean(ges_lengths):.2f}")
        print(f"中位数长度: {np.median(ges_lengths):.0f}")
        print(f"最小长度: {min(ges_lengths)}")
        print(f"最大长度: {max(ges_lengths)}")
        print(f"标准差: {np.std(ges_lengths):.2f}")
        
        # 扩展比例分析
        if expansion_ratios:
            print(f"\n=== 扩展比例分析 ===")
            print(f"平均扩展比例: {np.mean(expansion_ratios):.2f}x")
            print(f"中位数扩展比例: {np.median(expansion_ratios):.2f}x")
            print(f"最小扩展比例: {min(expansion_ratios):.2f}x")
            print(f"最大扩展比例: {max(expansion_ratios):.2f}x")
        
        # 语义改进分析
        if semantic_improvements:
            print(f"\n=== 语义改进分析 ===")
            print(f"平均语义改进: {np.mean(semantic_improvements):.4f}")
            print(f"中位数语义改进: {np.median(semantic_improvements):.4f}")
            print(f"最小语义改进: {min(semantic_improvements):.4f}")
            print(f"最大语义改进: {max(semantic_improvements):.4f}")
    
    # 长度分布统计
    print(f"\n=== Gold Triples 长度分布 ===")
    gold_counter = Counter(gold_lengths)
    for length in sorted(gold_counter.keys())[:15]:  # 显示前15个最常见的长度
        count = gold_counter[length]
        percentage = count / total_samples * 100
        print(f"长度 {length:2d}: {count:,} 个 ({percentage:5.1f}%)")
    
    if ges_lengths:
        print(f"\n=== GES 长度分布 ===")
        ges_counter = Counter(ges_lengths)
        for length in sorted(ges_counter.keys())[:20]:  # 显示前20个最常见的长度
            count = ges_counter[length]
            percentage = count / len(ges_lengths) * 100
            print(f"长度 {length:2d}: {count:,} 个 ({percentage:5.1f}%)")
    
    # 范围分布
    print(f"\n=== Gold Triples 范围分布 ===")
    ranges = [(1, 2), (3, 5), (6, 10), (11, 15), (16, 20), (21, 30), (31, float('inf'))]
    for start, end in ranges:
        if end == float('inf'):
            count = sum(1 for x in gold_lengths if x >= start)
            print(f"{start:2d}+ 个: {count:5,} ({count/total_samples*100:5.1f}%)")
        else:
            count = sum(1 for x in gold_lengths if start <= x <= end)
            print(f"{start:2d}-{end:2d} 个: {count:5,} ({count/total_samples*100:5.1f}%)")
    
    if ges_lengths:
        print(f"\n=== GES 范围分布 ===")
        for start, end in ranges:
            if end == float('inf'):
                count = sum(1 for x in ges_lengths if x >= start)
                print(f"{start:2d}+ 个: {count:5,} ({count/len(ges_lengths)*100:5.1f}%)")
            else:
                count = sum(1 for x in ges_lengths if start <= x <= end)
                print(f"{start:2d}-{end:2d} 个: {count:5,} ({count/len(ges_lengths)*100:5.1f}%)")
    
    # 添加百分位数分析
    print(f"\n=== Gold Triples 百分位数分析 ===")
    percentiles = [10, 25, 50, 75, 90, 95, 99]
    for p in percentiles:
        value = np.percentile(gold_lengths, p)
        print(f"{p:2d}% 分位数: {value:.1f}")
    
    if ges_lengths:
        print(f"\n=== GES 百分位数分析 ===")
        for p in percentiles:
            value = np.percentile(ges_lengths, p)
            print(f"{p:2d}% 分位数: {value:.1f}")
    
    # 添加扩展比例详细分析
    if expansion_ratios:
        print(f"\n=== 扩展比例详细分析 ===")
        expansion_ranges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, float('inf'))]
        for start, end in expansion_ranges:
            if end == float('inf'):
                count = sum(1 for x in expansion_ratios if x >= start)
                print(f"{start:.1f}x+ 扩展: {count:5,} ({count/len(expansion_ratios)*100:5.1f}%)")
            else:
                count = sum(1 for x in expansion_ratios if start <= x < end)
                print(f"{start:.1f}-{end:.1f}x 扩展: {count:5,} ({count/len(expansion_ratios)*100:5.1f}%)")

if __name__ == "__main__":
    import sys
    
    # 支持命令行参数
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    else:
        # 默认文件路径
        file_path = "/mnt/d/experiments/GraphDeEP/experiment_records/trimming_results/metaqa-1hop/test_simple_trimming_results_with_ges.jsonl"
    
    if not Path(file_path).exists():
        print(f"错误: 文件 {file_path} 不存在")
        print(f"用法: python {sys.argv[0]} [文件路径]")
        exit(1)
    
    analyze_ges_lengths(file_path) 