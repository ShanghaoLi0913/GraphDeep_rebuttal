import json
import os
import argparse
from collections import Counter

def find_latest_trimming_file():
    """找到最新的trimming结果文件"""
    experiment_dir = "experiment_records"
    trimming_files = []
    
    for file in os.listdir(experiment_dir):
        if file.startswith("trimming_results_") and file.endswith(".jsonl"):
            trimming_files.append(file)
    
    if not trimming_files:
        print("未找到trimming结果文件")
        return None
    
    # 按时间戳排序，返回最新的
    trimming_files.sort(reverse=True)
    latest_file = os.path.join(experiment_dir, trimming_files[0])
    print(f"分析文件: {latest_file}")
    return latest_file

def analyze_trimming_results(file_path):
    """分析裁剪结果"""
    
    if not file_path:
        return
    
    print(f"\n=== 分析裁剪结果: {os.path.basename(file_path)} ===")
    
    # 存储数据
    samples = []
    config = None
    final_stats = None
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            try:
                data = json.loads(line)
                
                if 'config' in data:
                    config = data['config']
                elif 'final_stats' in data:
                    final_stats = data['final_stats']
                elif 'batch_stats' in data:
                    continue  # 跳过批次统计
                else:
                    # 这是样本数据
                    samples.append(data)
            except:
                continue
    
    print(f"\n📊 基本统计:")
    print(f"总样本数: {len(samples)}")
    if config:
        print(f"配置信息: topk={config.get('topk')}, 时间戳={config.get('timestamp')}")
    
    if not samples:
        print("没有找到有效的样本数据")
        return
    
    # 检测是否为双重策略数据
    has_golden_expansion = any('golden_expansion_set' in s for s in samples)
    has_expansion_length = any('golden_expansion_length' in s for s in samples)
    
    if has_golden_expansion or has_expansion_length:
        print(f"\n🎯 检测到双重策略数据 - 分析TUS和FGAS两套Golden Triples")
        strategy_type = "dual_strategy"
    else:
        print(f"\n🎯 检测到传统策略数据 - 分析单一Golden Triples")
        strategy_type = "traditional"
    
    # 1. Gold Triples长度分析 (TUS策略)
    print(f"\n🎯 TUS Golden Triples长度分析 (精确shortest path):")
    gold_lengths = [s.get('gold_triples_length', len(s.get('gold_triples', []))) for s in samples]
    
    print(f"最小长度: {min(gold_lengths)}")
    print(f"最大长度: {max(gold_lengths)}")
    print(f"平均长度: {sum(gold_lengths)/len(gold_lengths):.2f}")
    print(f"中位数: {sorted(gold_lengths)[len(gold_lengths)//2]}")
    
    # 双重策略额外分析
    if strategy_type == "dual_strategy":
        # 2. Golden Expansion Set长度分析 (FGAS策略)
        print(f"\n🌟 FGAS Golden Expansion Set长度分析 (扩展语义集合):")
        
        # 获取expansion长度
        if has_expansion_length:
            expansion_lengths = [s.get('golden_expansion_length', 0) for s in samples]
        else:
            expansion_lengths = [len(s.get('golden_expansion_set', [])) for s in samples]
        
        print(f"最小长度: {min(expansion_lengths)}")
        print(f"最大长度: {max(expansion_lengths)}")
        print(f"平均长度: {sum(expansion_lengths)/len(expansion_lengths):.2f}")
        print(f"中位数: {sorted(expansion_lengths)[len(expansion_lengths)//2]}")
        
        # 扩展倍数分析
        expansion_ratios = []
        for gold_len, exp_len in zip(gold_lengths, expansion_lengths):
            if gold_len > 0:
                expansion_ratios.append(exp_len / gold_len)
        
        if expansion_ratios:
            print(f"\n📈 扩展倍数分析 (GES/TUS):")
            print(f"最小扩展倍数: {min(expansion_ratios):.2f}")
            print(f"最大扩展倍数: {max(expansion_ratios):.2f}")
            print(f"平均扩展倍数: {sum(expansion_ratios)/len(expansion_ratios):.2f}")
            print(f"中位扩展倍数: {sorted(expansion_ratios)[len(expansion_ratios)//2]:.2f}")
        
        # 对比分析
        print(f"\n📊 TUS vs FGAS 对比:")
        print(f"{'指标':<20} {'TUS (精确)':<15} {'FGAS (扩展)':<15} {'扩展倍数':<10}")
        print(f"{'-'*60}")
        print(f"{'平均长度':<20} {sum(gold_lengths)/len(gold_lengths):<15.2f} {sum(expansion_lengths)/len(expansion_lengths):<15.2f} {sum(expansion_lengths)/sum(gold_lengths):<10.2f}")
        print(f"{'中位长度':<20} {sorted(gold_lengths)[len(gold_lengths)//2]:<15} {sorted(expansion_lengths)[len(expansion_lengths)//2]:<15} {'-':<10}")
        print(f"{'最大长度':<20} {max(gold_lengths):<15} {max(expansion_lengths):<15} {'-':<10}")
        
        # 双重策略优势分析
        print(f"\n🚀 双重策略优势评估:")
        tus_reasonable = sum(1 for l in gold_lengths if 1 <= l <= 5)
        fgas_rich = sum(1 for l in expansion_lengths if l >= 10)
        
        print(f"TUS精确性: {tus_reasonable}/{len(samples)} ({tus_reasonable/len(samples)*100:.1f}%) 样本的TUS长度在1-5范围")
        print(f"FGAS丰富性: {fgas_rich}/{len(samples)} ({fgas_rich/len(samples)*100:.1f}%) 样本的FGAS长度≥10")
        
        if tus_reasonable > len(samples) * 0.8 and fgas_rich > len(samples) * 0.8:
            print("✅ 双重策略效果优秀！TUS精确且FGAS丰富")
        elif tus_reasonable > len(samples) * 0.6 and fgas_rich > len(samples) * 0.6:
            print("⚠️  双重策略效果良好，仍有优化空间")
        else:
            print("❌ 双重策略需要进一步优化")
    
    # 精简分布统计
    print(f"\n📊 分布统计:")
    
    # TUS分布
    tus_1_3 = sum(1 for x in gold_lengths if 1 <= x <= 3)
    tus_4_10 = sum(1 for x in gold_lengths if 4 <= x <= 10)
    tus_over_10 = sum(1 for x in gold_lengths if x > 10)
    
    print(f"TUS分布: 1-3个({tus_1_3}, {tus_1_3/len(samples)*100:.1f}%) | "
          f"4-10个({tus_4_10}, {tus_4_10/len(samples)*100:.1f}%) | "
          f">10个({tus_over_10}, {tus_over_10/len(samples)*100:.1f}%)")
    
    # FGAS分布（如果是双重策略）
    if strategy_type == "dual_strategy":
        if has_expansion_length:
            expansion_lengths = [s.get('golden_expansion_length', 0) for s in samples]
        else:
            expansion_lengths = [len(s.get('golden_expansion_set', [])) for s in samples]
            
        ges_under_5 = sum(1 for x in expansion_lengths if x < 5)
        ges_5_15 = sum(1 for x in expansion_lengths if 5 <= x <= 15)
        ges_16_30 = sum(1 for x in expansion_lengths if 16 <= x <= 30)
        ges_over_30 = sum(1 for x in expansion_lengths if x > 30)
        
        print(f"GES分布: <5个({ges_under_5}, {ges_under_5/len(samples)*100:.1f}%) | "
              f"5-15个({ges_5_15}, {ges_5_15/len(samples)*100:.1f}%) | "
              f"16-30个({ges_16_30}, {ges_16_30/len(samples)*100:.1f}%) | "
              f">30个({ges_over_30}, {ges_over_30/len(samples)*100:.1f}%)")
        
        # 验证修复效果
        if ges_over_30 == 0:
            print(f"✅ GES限制修复成功！所有GES都在topk范围内")
        else:
            print(f"⚠️  注意：仍有{ges_over_30}个样本的GES超过topk限制")
    
    # 2. 子图覆盖率分析
    print(f"\n📈 子图统计:")
    original_lengths = [s.get('original_subgraph_length', 0) for s in samples]
    trimmed_lengths = [s.get('trimmed_subgraph_length', 20) for s in samples]
    
    print(f"原始子图长度 - 最大: {max(original_lengths)}, 最小: {min(original_lengths)}, 平均: {sum(original_lengths)/len(original_lengths):.1f}")
    print(f"裁剪子图长度 - 最大: {max(trimmed_lengths)}, 最小: {min(trimmed_lengths)}, 平均: {sum(trimmed_lengths)/len(trimmed_lengths):.1f}")
    
    # 覆盖率
    coverage_rates = [t/o*100 if o > 0 else 0 for t, o in zip(trimmed_lengths, original_lengths)]
    print(f"覆盖率 - 最大: {max(coverage_rates):.1f}%, 最小: {min(coverage_rates):.1f}%, 平均: {sum(coverage_rates)/len(coverage_rates):.1f}%")
    
    # 3. 答案覆盖情况
    print(f"\n✅ 答案覆盖情况:")
    answer_covered = [s.get('answer_covered', False) for s in samples]
    covered_count = sum(answer_covered)
    coverage_percentage = covered_count / len(samples) * 100
    print(f"答案覆盖样本: {covered_count}/{len(samples)} ({coverage_percentage:.1f}%)")
    
    # 简化示例分析
    print(f"\n📝 简要示例:")
    moderate_sample = next((s for s in samples if 1 <= s.get('gold_triples_length', 0) <= 5), None)
    if moderate_sample:
        gold_len = moderate_sample.get('gold_triples_length', 0)
        exp_len = moderate_sample.get('golden_expansion_length', 0) if strategy_type == "dual_strategy" else 0
        print(f"典型样本: TUS={gold_len}" + (f", GES={exp_len}" if strategy_type == "dual_strategy" else "") + f" - {moderate_sample.get('question', '')[:50]}...")
    
    # 检查是否有异常长的样本
    long_samples = [s for s in samples if s.get('gold_triples_length', 0) > 20]
    if long_samples:
        print(f"⚠️  发现{len(long_samples)}个TUS长度>20的样本，最长为{max(s.get('gold_triples_length', 0) for s in long_samples)}")
    
    # 5. 最终统计
    if final_stats:
        print(f"\n🎉 最终统计:")
        print(f"处理总数: {final_stats.get('total_samples')}")
        print(f"答案覆盖数: {final_stats.get('answer_covered_count')}")
        print(f"答案召回率: {final_stats.get('answer_recall', 0):.2f}%")
        print(f"处理耗时: {final_stats.get('total_time', 0):.2f}秒")
    
    # 6. 改善效果评估
    print(f"\n🚀 策略效果评估:")
    
    if strategy_type == "dual_strategy":
        print(f"双重策略效果分析:")
        
        # TUS效果
        tus_very_long = sum(1 for l in gold_lengths if l > 100)
        tus_long = sum(1 for l in gold_lengths if l > 20)
        tus_reasonable = sum(1 for l in gold_lengths if 1 <= l <= 5)
        
        print(f"\nTUS (注意力精度) 效果:")
        print(f"  超长样本 (>100): {tus_very_long} 个 ({tus_very_long/len(samples)*100:.1f}%)")
        print(f"  偏长样本 (>20):  {tus_long} 个 ({tus_long/len(samples)*100:.1f}%)")
        print(f"  合理样本 (1-5):  {tus_reasonable} 个 ({tus_reasonable/len(samples)*100:.1f}%)")
        
        # FGAS效果
        if has_expansion_length:
            fgas_lengths = [s.get('golden_expansion_length', 0) for s in samples]
        else:
            fgas_lengths = [len(s.get('golden_expansion_set', [])) for s in samples]
            
        fgas_too_small = sum(1 for l in fgas_lengths if l < 5)
        fgas_reasonable = sum(1 for l in fgas_lengths if 5 <= l <= 20)
        fgas_rich = sum(1 for l in fgas_lengths if l > 20)
        
        print(f"\nFGAS (语义丰富度) 效果:")
        print(f"  过小样本 (<5):   {fgas_too_small} 个 ({fgas_too_small/len(samples)*100:.1f}%)")
        print(f"  合理样本 (5-20): {fgas_reasonable} 个 ({fgas_reasonable/len(samples)*100:.1f}%)")
        print(f"  丰富样本 (>20):  {fgas_rich} 个 ({fgas_rich/len(samples)*100:.1f}%)")
        
        # 综合评价
        print(f"\n双重策略综合评价:")
        if tus_reasonable > len(samples) * 0.8 and (fgas_reasonable + fgas_rich) > len(samples) * 0.8:
            print("🎉 双重策略效果卓越！TUS精确且FGAS丰富，预期两个指标都显著")
        elif tus_reasonable > len(samples) * 0.6 and (fgas_reasonable + fgas_rich) > len(samples) * 0.6:
            print("✅ 双重策略效果良好，TUS和FGAS都有较好表现")
        else:
            print("⚠️  双重策略仍有改进空间")
            
    else:
        # 传统策略评估
        very_long_count = sum(1 for l in gold_lengths if l > 100)
        long_count = sum(1 for l in gold_lengths if l > 20)
        reasonable_count = sum(1 for l in gold_lengths if 1 <= l <= 5)
        
        print(f"传统策略效果分析:")
        print(f"超长样本 (>100): {very_long_count} 个 ({very_long_count/len(samples)*100:.1f}%)")
        print(f"偏长样本 (>20):  {long_count} 个 ({long_count/len(samples)*100:.1f}%)")
        print(f"合理样本 (1-5):  {reasonable_count} 个 ({reasonable_count/len(samples)*100:.1f}%)")
        
        if very_long_count == 0 and long_count < len(samples) * 0.1:
            print("✅ 精确匹配效果良好！大部分gold_triples长度合理")
        elif very_long_count == 0:
            print("⚠️  已消除超长情况，但仍有部分偏长样本需要进一步优化")
        else:
            print("❌ 仍存在超长样本，可能需要进一步优化匹配算法")
    
    # 7. 问题类型分析
    print(f"\n🔍 问题类型分析:")
    
    # 分析问题模式
    question_patterns = {}
    for sample in samples:
        question = sample.get('question', '').lower()
        gold_len = sample.get('gold_triples_length', 0)
        
        # 提取问题类型
        if 'act in' in question or 'acted in' in question:
            pattern = 'act in'
        elif 'directed' in question:
            pattern = 'directed'
        elif 'written' in question:
            pattern = 'written'
        elif 'genre' in question:
            pattern = 'genre'
        elif 'year' in question:
            pattern = 'year'
        elif 'language' in question:
            pattern = 'language'
        elif 'tags' in question:
            pattern = 'tags'
        else:
            pattern = 'other'
        
        if pattern not in question_patterns:
            question_patterns[pattern] = {'count': 0, 'gold_lengths': []}
        
        question_patterns[pattern]['count'] += 1
        question_patterns[pattern]['gold_lengths'].append(gold_len)
    
    print(f"问题类型分布:")
    for pattern, data in sorted(question_patterns.items(), key=lambda x: x[1]['count'], reverse=True):
        count = data['count']
        avg_gold_len = sum(data['gold_lengths']) / len(data['gold_lengths'])
        percentage = count / len(samples) * 100
        print(f"  {pattern:>10}: {count:4d} 个样本 ({percentage:5.1f}%), 平均gold长度: {avg_gold_len:.1f}")
    
    # 分析单答案vs多答案问题
    single_answer = sum(1 for s in samples if len(s.get('golden_texts', [])) == 1)
    multi_answer = len(samples) - single_answer
    
    print(f"\n答案数量分析:")
    print(f"单答案问题: {single_answer} 个 ({single_answer/len(samples)*100:.1f}%)")
    print(f"多答案问题: {multi_answer} 个 ({multi_answer/len(samples)*100:.1f}%)")
    
    # 分析单答案问题的gold triple长度
    single_answer_gold_lengths = [s.get('gold_triples_length', 0) for s in samples if len(s.get('golden_texts', [])) == 1]
    multi_answer_gold_lengths = [s.get('gold_triples_length', 0) for s in samples if len(s.get('golden_texts', [])) > 1]
    
    if single_answer_gold_lengths:
        avg_single = sum(single_answer_gold_lengths) / len(single_answer_gold_lengths)
        print(f"单答案问题平均gold长度: {avg_single:.1f}")
    
    if multi_answer_gold_lengths:
        avg_multi = sum(multi_answer_gold_lengths) / len(multi_answer_gold_lengths)
        print(f"多答案问题平均gold长度: {avg_multi:.1f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='分析裁剪结果')
    parser.add_argument('--file', '-f', help='指定要分析的trimming结果文件路径')
    parser.add_argument('--latest', '-l', action='store_true', help='分析最新的trimming结果文件')
    
    args = parser.parse_args()
    
    if args.file:
        # 使用指定的文件
        file_path = args.file
        if not os.path.exists(file_path):
            print(f"文件不存在: {file_path}")
            exit(1)
    elif args.latest:
        # 使用最新的文件
        file_path = find_latest_trimming_file()
        if not file_path:
            exit(1)
    else:
        # 默认使用指定的文件
        file_path = "experiment_records/trimming_results_tus_consistent_20250623_203829.jsonl"
        if not os.path.exists(file_path):
            print(f"文件不存在: {file_path}")
            exit(1)
    
    # 分析结果
    analyze_trimming_results(file_path) 