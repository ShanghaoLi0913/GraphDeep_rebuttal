#!/usr/bin/env python3
"""
分析两个版本的裁剪结果差异
对比 trimming_results_20250624_133213_subgraph1.jsonl (版本1) 和 trimming_results_20250624_130417_subgraph2.jsonl (版本2)
特别关注：Gold Triples是否都在最终的裁剪子图中
"""

import json
import numpy as np
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Any

def check_gold_triples_in_trimmed(gold_triples: List[List[str]], trimmed_triples: List[List[str]]) -> Dict:
    """检查gold triples是否都在trimmed subgraph中"""
    # 转换为集合便于比较
    gold_set = set(tuple(triple) for triple in gold_triples)
    trimmed_set = set(tuple(triple) for triple in trimmed_triples)
    
    # 计算重叠
    intersection = gold_set & trimmed_set
    missing_gold = gold_set - trimmed_set
    
    return {
        'total_gold_count': len(gold_set),
        'preserved_gold_count': len(intersection),
        'missing_gold_count': len(missing_gold),
        'preservation_rate': len(intersection) / len(gold_set) if gold_set else 1.0,
        'all_gold_preserved': len(missing_gold) == 0,
        'missing_gold_triples': list(missing_gold)
    }

def analyze_gold_preservation(samples: List[Dict]) -> Dict:
    """分析所有样本的gold triples保持情况"""
    preservation_stats = []
    
    for sample in samples:
        gold_check = check_gold_triples_in_trimmed(
            sample['gold_triples'], 
            sample['trimmed_triples']
        )
        gold_check['sample_id'] = sample['sample_id']
        gold_check['question'] = sample['question']
        preservation_stats.append(gold_check)
    
    # 汇总统计
    total_samples = len(preservation_stats)
    fully_preserved = sum(1 for s in preservation_stats if s['all_gold_preserved'])
    preservation_rates = [s['preservation_rate'] for s in preservation_stats]
    
    summary = {
        'total_samples': total_samples,
        'fully_preserved_count': fully_preserved,
        'fully_preserved_rate': fully_preserved / total_samples if total_samples > 0 else 0,
        'average_preservation_rate': np.mean(preservation_rates),
        'median_preservation_rate': np.median(preservation_rates),
        'min_preservation_rate': min(preservation_rates) if preservation_rates else 0,
        'preservation_stats': preservation_stats
    }
    
    return summary

def load_trimming_results(file_path: str) -> Tuple[Dict, List[Dict]]:
    """加载裁剪结果文件"""
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # 第一行是配置信息
    try:
        config = json.loads(lines[0].strip())
    except json.JSONDecodeError as e:
        print(f"Error parsing config line: {e}")
        print(f"Config line content: {lines[0][:200]}...")
        raise
    
    # 其余行是样本数据（跳过batch_stats行）
    samples = []
    for i, line in enumerate(lines[1:], 1):
        line = line.strip()
        if not line:  # 跳过空行
            continue
            
        try:
            data = json.loads(line)
            if 'sample_id' in data:  # 只保留样本数据，跳过batch_stats
                samples.append(data)
        except json.JSONDecodeError as e:
            print(f"Error parsing line {i+1}: {e}")
            print(f"Line content: {line[:200]}...")
            # 尝试修复常见的JSON问题
            try:
                # 修复已知的格式错误
                fixed_line = line.replace('old_triples_length"g":', '"gold_triples_length":')
                fixed_line = fixed_line.replace("'", '"')
                data = json.loads(fixed_line)
                if 'sample_id' in data:
                    samples.append(data)
                    print(f"Fixed line {i+1} successfully")
            except Exception as fix_e:
                print(f"Could not fix line {i+1}: {fix_e}, skipping...")
                continue
    
    return config, samples

def analyze_sample_differences(sample1: Dict, sample2: Dict) -> Dict:
    """分析单个样本的差异"""
    diff = {
        'sample_id': sample1['sample_id'],
        'question': sample1['question'],
        'golden_texts': sample1['golden_texts'],
        
        # 基本统计差异
        'gold_triples_diff': sample2['gold_triples_length'] - sample1['gold_triples_length'],
        'v1_gold_count': sample1['gold_triples_length'],
        'v2_gold_count': sample2['gold_triples_length'],
        
        # 处理时间差异
        'processing_time_diff': sample2['processing_time'] - sample1['processing_time'],
        'v1_time': sample1['processing_time'],
        'v2_time': sample2['processing_time'],
        
        # 答案覆盖情况
        'v1_covered': sample1['answer_covered'],
        'v2_covered': sample2['answer_covered'],
        'coverage_changed': sample1['answer_covered'] != sample2['answer_covered'],
        
        # Gold triples内容差异
        'v1_gold_triples': sample1['gold_triples'],
        'v2_gold_triples': sample2['gold_triples'],
        'gold_triples_same': sample1['gold_triples'] == sample2['gold_triples'],
        
        # Trimmed triples内容差异
        'v1_trimmed_triples': sample1['trimmed_triples'],
        'v2_trimmed_triples': sample2['trimmed_triples'],
        'trimmed_triples_same': sample1['trimmed_triples'] == sample2['trimmed_triples']
    }
    
    return diff

def calculate_overlap_metrics(list1: List, list2: List) -> Dict:
    """计算两个列表的重叠度量"""
    set1 = set(map(tuple, list1))
    set2 = set(map(tuple, list2))
    
    intersection = set1 & set2
    union = set1 | set2
    
    return {
        'intersection_size': len(intersection),
        'union_size': len(union),
        'jaccard': len(intersection) / len(union) if union else 0,
        'recall_v1_to_v2': len(intersection) / len(set1) if set1 else 0,
        'precision_v2_to_v1': len(intersection) / len(set2) if set2 else 0,
        'only_in_v1': len(set1 - set2),
        'only_in_v2': len(set2 - set1)
    }

def analyze_datasets(file1: str, file2: str) -> Dict:
    """分析两个数据集的整体差异"""
    print(f"Loading {file1}...")
    config1, samples1 = load_trimming_results(file1)
    
    print(f"Loading {file2}...")
    config2, samples2 = load_trimming_results(file2)
    
    print(f"Version 1: {len(samples1)} samples")
    print(f"Version 2: {len(samples2)} samples")
    
    # 分析gold triples保持情况
    print("\n🔍 分析版本1的Gold Triples保持情况...")
    v1_preservation = analyze_gold_preservation(samples1)
    
    print("🔍 分析版本2的Gold Triples保持情况...")
    v2_preservation = analyze_gold_preservation(samples2)
    
    # 处理样本数量不匹配的情况
    if len(samples1) != len(samples2):
        print(f"⚠️  Warning: Sample count mismatch: {len(samples1)} vs {len(samples2)}")
        
        # 只分析共同的样本数量
        min_samples = min(len(samples1), len(samples2))
        print(f"📝 Will analyze the first {min_samples} samples from both datasets")
        
        # 创建样本ID到索引的映射
        samples1_dict = {s['sample_id']: s for s in samples1}
        samples2_dict = {s['sample_id']: s for s in samples2}
        
        # 找出共同的样本ID
        common_ids = set(samples1_dict.keys()) & set(samples2_dict.keys())
        print(f"📊 Found {len(common_ids)} common sample IDs")
        
        if len(common_ids) == 0:
            raise ValueError("No common sample IDs found between the two datasets!")
        
        # 重新构建样本列表，只包含共同的样本
        samples1 = [samples1_dict[sid] for sid in sorted(common_ids)]
        samples2 = [samples2_dict[sid] for sid in sorted(common_ids)]
        
        print(f"✅ Using {len(samples1)} common samples for analysis")
    
    # 逐样本分析
    sample_diffs = []
    gold_overlap_metrics = []
    trimmed_overlap_metrics = []
    
    for i, (s1, s2) in enumerate(zip(samples1, samples2)):
        if i % 1000 == 0:
            print(f"Processing sample {i}...")
            
        if s1['sample_id'] != s2['sample_id']:
            print(f"⚠️  Sample ID mismatch at {i}: {s1['sample_id']} vs {s2['sample_id']}")
            continue
        
        diff = analyze_sample_differences(s1, s2)
        sample_diffs.append(diff)
        
        # 计算gold triples重叠度
        gold_overlap = calculate_overlap_metrics(s1['gold_triples'], s2['gold_triples'])
        gold_overlap['sample_id'] = s1['sample_id']
        gold_overlap_metrics.append(gold_overlap)
        
        # 计算trimmed triples重叠度
        trimmed_overlap = calculate_overlap_metrics(s1['trimmed_triples'], s2['trimmed_triples'])
        trimmed_overlap['sample_id'] = s1['sample_id']
        trimmed_overlap_metrics.append(trimmed_overlap)
    
    # 整体统计
    analysis = {
        'config1': config1,
        'config2': config2,
        'sample_diffs': sample_diffs,
        'gold_overlap_metrics': gold_overlap_metrics,
        'trimmed_overlap_metrics': trimmed_overlap_metrics,
        'v1_gold_preservation': v1_preservation,
        'v2_gold_preservation': v2_preservation
    }
    
    return analysis

def generate_summary_stats(analysis: Dict) -> None:
    """生成汇总统计"""
    diffs = analysis['sample_diffs']
    gold_overlaps = analysis['gold_overlap_metrics']
    trimmed_overlaps = analysis['trimmed_overlap_metrics']
    
    print("\n" + "="*60)
    print("📊 汇总统计")
    print("="*60)
    
    # 基本统计
    print(f"总样本数: {len(diffs)}")
    
    # Gold Triples保持情况分析
    v1_preservation = analysis['v1_gold_preservation']
    v2_preservation = analysis['v2_gold_preservation']
    
    print(f"\n🎯 Gold Triples 保持情况:")
    print(f"  版本1:")
    print(f"    完全保持率: {v1_preservation['fully_preserved_rate']*100:.1f}% ({v1_preservation['fully_preserved_count']}/{v1_preservation['total_samples']})")
    print(f"    平均保持率: {v1_preservation['average_preservation_rate']*100:.1f}%")
    print(f"    最低保持率: {v1_preservation['min_preservation_rate']*100:.1f}%")
    
    print(f"  版本2:")
    print(f"    完全保持率: {v2_preservation['fully_preserved_rate']*100:.1f}% ({v2_preservation['fully_preserved_count']}/{v2_preservation['total_samples']})")
    print(f"    平均保持率: {v2_preservation['average_preservation_rate']*100:.1f}%")
    print(f"    最低保持率: {v2_preservation['min_preservation_rate']*100:.1f}%")
    
    # 找出Gold Triples丢失的案例
    v1_missing_cases = [s for s in v1_preservation['preservation_stats'] if not s['all_gold_preserved']]
    v2_missing_cases = [s for s in v2_preservation['preservation_stats'] if not s['all_gold_preserved']]
    
    if v1_missing_cases:
        print(f"\n⚠️  版本1中有{len(v1_missing_cases)}个样本Gold Triples未完全保持")
        print("   前5个案例:")
        for i, case in enumerate(v1_missing_cases[:5]):
            print(f"     {i+1}. Sample {case['sample_id']}: 保持率 {case['preservation_rate']*100:.1f}% ({case['preserved_gold_count']}/{case['total_gold_count']})")
    
    if v2_missing_cases:
        print(f"\n⚠️  版本2中有{len(v2_missing_cases)}个样本Gold Triples未完全保持")
        print("   前5个案例:")
        for i, case in enumerate(v2_missing_cases[:5]):
            print(f"     {i+1}. Sample {case['sample_id']}: 保持率 {case['preservation_rate']*100:.1f}% ({case['preserved_gold_count']}/{case['total_gold_count']})")
    
    # Gold triples差异统计
    gold_diff_values = [d['gold_triples_diff'] for d in diffs]
    print(f"\n🎯 Gold Triples 数量差异:")
    print(f"  平均差异: {np.mean(gold_diff_values):.2f}")
    print(f"  中位数差异: {np.median(gold_diff_values):.2f}")
    print(f"  标准差: {np.std(gold_diff_values):.2f}")
    print(f"  最大减少: {min(gold_diff_values)}")
    print(f"  最大增加: {max(gold_diff_values)}")
    
    # 版本1 vs 版本2 gold triples数量对比
    v1_gold_counts = [d['v1_gold_count'] for d in diffs]
    v2_gold_counts = [d['v2_gold_count'] for d in diffs]
    print(f"\n📈 Gold Triples 数量对比:")
    print(f"  版本1平均: {np.mean(v1_gold_counts):.2f}")
    print(f"  版本2平均: {np.mean(v2_gold_counts):.2f}")
    print(f"  版本1中位数: {np.median(v1_gold_counts):.2f}")
    print(f"  版本2中位数: {np.median(v2_gold_counts):.2f}")
    
    # 完全相同的样本比例
    same_gold_count = sum(1 for d in diffs if d['gold_triples_same'])
    same_trimmed_count = sum(1 for d in diffs if d['trimmed_triples_same'])
    print(f"\n🔄 相同性分析:")
    print(f"  Gold triples完全相同: {same_gold_count}/{len(diffs)} ({same_gold_count/len(diffs)*100:.1f}%)")
    print(f"  Trimmed triples完全相同: {same_trimmed_count}/{len(diffs)} ({same_trimmed_count/len(diffs)*100:.1f}%)")
    
    # 答案覆盖变化
    coverage_changed_count = sum(1 for d in diffs if d['coverage_changed'])
    print(f"  答案覆盖发生变化: {coverage_changed_count}/{len(diffs)} ({coverage_changed_count/len(diffs)*100:.1f}%)")
    
    # Gold triples重叠度统计
    jaccard_scores = [m['jaccard'] for m in gold_overlaps]
    print(f"\n🎯 Gold Triples 重叠度:")
    print(f"  平均Jaccard相似度: {np.mean(jaccard_scores):.3f}")
    print(f"  中位数Jaccard相似度: {np.median(jaccard_scores):.3f}")
    
    # 处理时间对比
    time_diffs = [d['processing_time_diff'] for d in diffs]
    print(f"\n⏱️ 处理时间对比:")
    print(f"  平均时间差异: {np.mean(time_diffs):.4f}秒")
    print(f"  版本1平均时间: {np.mean([d['v1_time'] for d in diffs]):.4f}秒")
    print(f"  版本2平均时间: {np.mean([d['v2_time'] for d in diffs]):.4f}秒")

def find_extreme_cases(analysis: Dict, top_k: int = 5) -> None:
    """找出极端案例"""
    diffs = analysis['sample_diffs']
    
    print("\n" + "="*60)
    print("🔍 极端案例分析")
    print("="*60)
    
    # Gold triples数量减少最多的案例
    sorted_by_reduction = sorted(diffs, key=lambda x: x['gold_triples_diff'])
    print(f"\n📉 Gold Triples减少最多的{top_k}个案例:")
    for i, diff in enumerate(sorted_by_reduction[:top_k]):
        print(f"  {i+1}. Sample {diff['sample_id']}: {diff['v1_gold_count']} → {diff['v2_gold_count']} (减少{-diff['gold_triples_diff']})")
        print(f"     问题: {diff['question']}")
        print(f"     答案: {diff['golden_texts']}")
        print()
    
    # Gold triples数量增加最多的案例
    sorted_by_increase = sorted(diffs, key=lambda x: x['gold_triples_diff'], reverse=True)
    print(f"\n📈 Gold Triples增加最多的{top_k}个案例:")
    for i, diff in enumerate(sorted_by_increase[:top_k]):
        if diff['gold_triples_diff'] > 0:
            print(f"  {i+1}. Sample {diff['sample_id']}: {diff['v1_gold_count']} → {diff['v2_gold_count']} (增加{diff['gold_triples_diff']})")
            print(f"     问题: {diff['question']}")
            print(f"     答案: {diff['golden_texts']}")
            print()

def print_detailed_summary(analysis: Dict) -> None:
    """打印详细分析汇总"""
    diffs = analysis['sample_diffs']
    gold_overlaps = analysis['gold_overlap_metrics']
    trimmed_overlaps = analysis['trimmed_overlap_metrics']
    
    print(f"\n📋 详细分析汇总:")
    print(f"  - 样本差异分析: {len(diffs)} 个样本")
    print(f"  - Gold Triples重叠分析: {len(gold_overlaps)} 个样本")
    print(f"  - Trimmed Triples重叠分析: {len(trimmed_overlaps)} 个样本")

def main():
    # 文件路径
    file1 = "experiment_records/trimming_results_20250624_133213_subgraph1.jsonl"  # 版本1
    file2 = "experiment_records/trimming_results_20250624_130417_subgraph2.jsonl"  # 版本2
    
    print("🔍 开始分析两个版本的裁剪结果差异...")
    print(f"版本1: {file1}")
    print(f"版本2: {file2}")
    
    # 执行分析
    analysis = analyze_datasets(file1, file2)
    
    # 生成统计报告
    generate_summary_stats(analysis)
    
    # 找出极端案例
    find_extreme_cases(analysis)
    
    # 打印详细汇总
    print_detailed_summary(analysis)
    
    print("\n✅ 分析完成！所有结果已在终端显示。")

if __name__ == "__main__":
    main() 