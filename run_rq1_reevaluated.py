"""
RQ1分析脚本 - 重新评估版本

用于分析经过ChatGPT重新评估的推理结果，比较TUS、NTUS、FGAS指标
在truthful vs hallucinated responses之间的差异。
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import argparse
import os
from datetime import datetime
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def load_reevaluated_data(file_path: str) -> Tuple[List[Dict], Dict]:
    """加载GPT评估后的数据"""
    
    results = []
    config = None
    is_reevaluated = False
    
    print(f"Loading data from: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f):
            data = json.loads(line)
            
            if line_num == 0 and 'config' in data:
                config = data
                is_reevaluated = config.get('reevaluated', False)
                continue
                
            # 检查是否是GPT重新评估的数据
            if 'gpt_judgment' in data:
                results.append(data)
                is_reevaluated = True
            elif not is_reevaluated:
                # 如果不是重新评估的数据，但包含必要的指标，也加载
                if ('tus_score' in data and 'ntus_score' in data and 
                    'fgas_score' in data and 'metrics' in data):
                    results.append(data)
    
    if is_reevaluated:
        print(f"Loaded {len(results)} GPT-evaluated samples")
        if config and config.get('reevaluated'):
            print(f"Original file: {config.get('original_file', 'Unknown')}")
    else:
        print(f"Loaded {len(results)} original samples (not GPT-evaluated)")
        print("Note: This appears to be original data, not GPT-evaluated data.")
    
    return results, config

def extract_metrics_data(results: List[Dict]) -> Dict[str, List]:
    """提取指标数据"""
    
    data = {
        'tus_scores': [],
        'ntus_scores': [],
        'fgas_scores': [],
        'hit_at_1': [],
        'original_hit_at_1': []  # 保存原始评估结果用于对比
    }
    
    valid_count = 0
    is_reevaluated_data = False
    
    for result in results:
        try:
            # 检查必要的指标
            if ('tus_score' in result and 'ntus_score' in result and 
                'fgas_score' in result and 'metrics' in result):
                
                data['tus_scores'].append(result['tus_score'])
                data['ntus_scores'].append(result['ntus_score'])
                data['fgas_scores'].append(result['fgas_score'])
                data['hit_at_1'].append(result['metrics']['hit@1'])
                
                # 检查是否有重新评估数据
                if 'original_hit@1' in result:
                    data['original_hit_at_1'].append(result['original_hit@1'])
                    is_reevaluated_data = True
                else:
                    data['original_hit_at_1'].append(result['metrics']['hit@1'])
                
                valid_count += 1
                
        except Exception as e:
            print(f"Warning: Skipping invalid sample: {e}")
            continue
    
    if valid_count == 0:
        raise ValueError("No valid samples found with required metrics")
    
    print(f"Extracted metrics from {valid_count} valid samples")
    
    # 显示重新评估的影响
    if is_reevaluated_data:
        original_accuracy = np.mean(data['original_hit_at_1'])
        new_accuracy = np.mean(data['hit_at_1'])
        changed_count = sum(1 for orig, new in zip(data['original_hit_at_1'], data['hit_at_1']) if orig != new)
        
        print(f"Original accuracy: {original_accuracy:.3f}")
        print(f"New accuracy: {new_accuracy:.3f}")
        print(f"Changed evaluations: {changed_count}/{valid_count} ({changed_count/valid_count*100:.1f}%)")
        
        # 统计GPT判断分布  
        truthful_count = sum(data['hit_at_1'])
        hallucinating_count = valid_count - truthful_count
        print(f"GPT judgments: {truthful_count} truthful, {hallucinating_count} hallucinated")
    else:
        accuracy = np.mean(data['hit_at_1'])
        truthful_count = sum(data['hit_at_1'])
        hallucinating_count = valid_count - truthful_count
        print(f"Accuracy: {accuracy:.3f}")
        print(f"Distribution: {truthful_count} truthful, {hallucinating_count} hallucinated")
        print("Note: Using original evaluation results (no GPT evaluation detected)")
    
    return data

def perform_statistical_analysis(data: Dict[str, List]) -> Dict:
    """执行统计分析"""
    
    # 转换为numpy数组
    tus_scores = np.array(data['tus_scores'])
    ntus_scores = np.array(data['ntus_scores'])
    fgas_scores = np.array(data['fgas_scores'])
    hit_at_1 = np.array(data['hit_at_1'])
    
    # 分组：真实 vs 幻觉响应
    truthful_mask = hit_at_1 == True
    hallucinated_mask = hit_at_1 == False
    
    truthful_count = np.sum(truthful_mask)
    hallucinated_count = np.sum(hallucinated_mask)
    
    print(f"\nGroup sizes:")
    print(f"Truthful responses: {truthful_count}")
    print(f"Hallucinated responses: {hallucinated_count}")
    
    if truthful_count < 2 or hallucinated_count < 2:
        raise ValueError("Not enough samples in one or both groups for statistical analysis")
    
    # 计算每组的统计量
    analysis_results = {}
    
    metrics = ['TUS', 'NTUS', 'FGAS']
    scores = [tus_scores, ntus_scores, fgas_scores]
    
    for metric, score_array in zip(metrics, scores):
        truthful_scores = score_array[truthful_mask]
        hallucinated_scores = score_array[hallucinated_mask]
        
        # 基本统计
        truthful_stats = {
            'mean': np.mean(truthful_scores),
            'std': np.std(truthful_scores, ddof=1),
            'median': np.median(truthful_scores),
            'count': len(truthful_scores)
        }
        
        hallucinated_stats = {
            'mean': np.mean(hallucinated_scores),
            'std': np.std(hallucinated_scores, ddof=1),
            'median': np.median(hallucinated_scores),
            'count': len(hallucinated_scores)
        }
        
        # t检验
        t_stat, p_value = stats.ttest_ind(truthful_scores, hallucinated_scores)
        
        # Cohen's d (效应大小)
        pooled_std = np.sqrt(((len(truthful_scores) - 1) * truthful_stats['std']**2 + 
                             (len(hallucinated_scores) - 1) * hallucinated_stats['std']**2) / 
                            (len(truthful_scores) + len(hallucinated_scores) - 2))
        cohens_d = (truthful_stats['mean'] - hallucinated_stats['mean']) / pooled_std
        
        analysis_results[metric] = {
            'truthful': truthful_stats,
            'hallucinated': hallucinated_stats,
            't_statistic': t_stat,
            'p_value': p_value,
            'cohens_d': cohens_d,
            'significant': p_value < 0.05
        }
    
    return analysis_results

def create_visualizations(data: Dict[str, List], analysis_results: Dict, output_dir: str):
    """创建可视化图表"""
    
    tus_scores = np.array(data['tus_scores'])
    ntus_scores = np.array(data['ntus_scores'])
    fgas_scores = np.array(data['fgas_scores'])
    hit_at_1 = np.array(data['hit_at_1'])
    
    # 1. KDE密度分布图
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    metrics = ['TUS', 'NTUS', 'FGAS']
    scores = [tus_scores, ntus_scores, fgas_scores]
    colors = ['blue', 'green', 'red']
    
    for i, (metric, score_array, color) in enumerate(zip(metrics, scores, colors)):
        truthful_scores = score_array[hit_at_1 == True]
        hallucinated_scores = score_array[hit_at_1 == False]
        
        # 绘制KDE
        sns.kdeplot(data=truthful_scores, ax=axes[i], label='Truthful', color=color, alpha=0.7)
        sns.kdeplot(data=hallucinated_scores, ax=axes[i], label='Hallucinated', color='orange', alpha=0.7)
        
        axes[i].set_title(f'{metric} Score Distribution (GPT Evaluation)')
        axes[i].set_xlabel(f'{metric} Score')
        axes[i].set_ylabel('Density')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/rq1_kde_plots_reevaluated.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. 箱线图
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    for i, (metric, score_array, color) in enumerate(zip(metrics, scores, colors)):
        truthful_scores = score_array[hit_at_1 == True]
        hallucinated_scores = score_array[hit_at_1 == False]
        
        data_for_box = [truthful_scores, hallucinated_scores]
        bp = axes[i].boxplot(data_for_box, labels=['Truthful', 'Hallucinated'], 
                           patch_artist=True, notch=True)
        
        bp['boxes'][0].set_facecolor(color)
        bp['boxes'][1].set_facecolor('orange')
        
        axes[i].set_title(f'{metric} Score Comparison (GPT Evaluation)')
        axes[i].set_ylabel(f'{metric} Score')
        axes[i].grid(True, alpha=0.3)
        
        # 添加显著性标注
        if analysis_results[metric]['significant']:
            axes[i].text(0.5, 0.95, f"p={analysis_results[metric]['p_value']:.4f}*", 
                        transform=axes[i].transAxes, ha='center', va='top',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
        else:
            axes[i].text(0.5, 0.95, f"p={analysis_results[metric]['p_value']:.4f}", 
                        transform=axes[i].transAxes, ha='center', va='top',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.7))
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/rq1_boxplots_reevaluated.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. 均值对比柱状图
    fig, ax = plt.subplots(figsize=(12, 8))
    
    metrics_means_truthful = [analysis_results[metric]['truthful']['mean'] for metric in metrics]
    metrics_means_hallucinated = [analysis_results[metric]['hallucinated']['mean'] for metric in metrics]
    metrics_stds_truthful = [analysis_results[metric]['truthful']['std'] for metric in metrics]
    metrics_stds_hallucinated = [analysis_results[metric]['hallucinated']['std'] for metric in metrics]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, metrics_means_truthful, width, 
                   yerr=metrics_stds_truthful, label='Truthful', 
                   color=['blue', 'green', 'red'], alpha=0.7, capsize=5)
    
    bars2 = ax.bar(x + width/2, metrics_means_hallucinated, width,
                   yerr=metrics_stds_hallucinated, label='Hallucinated',
                   color='orange', alpha=0.7, capsize=5)
    
    ax.set_xlabel('Metrics')
    ax.set_ylabel('Mean Score')
    ax.set_title('Mean Metric Scores: Truthful vs Hallucinated (GPT Evaluation)')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 添加显著性标星
    for i, metric in enumerate(metrics):
        if analysis_results[metric]['significant']:
            max_height = max(metrics_means_truthful[i] + metrics_stds_truthful[i],
                           metrics_means_hallucinated[i] + metrics_stds_hallucinated[i])
            ax.text(i, max_height + 0.01, '*', ha='center', va='bottom', fontsize=20, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/rq1_bar_chart_reevaluated.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. TUS vs FGAS散点图
    fig, ax = plt.subplots(figsize=(10, 8))
    
    truthful_mask = hit_at_1 == True
    hallucinated_mask = hit_at_1 == False
    
    ax.scatter(tus_scores[truthful_mask], fgas_scores[truthful_mask], 
              c='blue', alpha=0.6, label='Truthful', s=50)
    ax.scatter(tus_scores[hallucinated_mask], fgas_scores[hallucinated_mask], 
              c='orange', alpha=0.6, label='Hallucinated', s=50)
    
    # 计算相关性
    correlation_all = np.corrcoef(tus_scores, fgas_scores)[0, 1]
    correlation_truthful = np.corrcoef(tus_scores[truthful_mask], fgas_scores[truthful_mask])[0, 1]
    correlation_hallucinated = np.corrcoef(tus_scores[hallucinated_mask], fgas_scores[hallucinated_mask])[0, 1]
    
    ax.set_xlabel('TUS Score')
    ax.set_ylabel('FGAS Score')
    ax.set_title(f'TUS vs FGAS Correlation (GPT Evaluation)\nAll: r={correlation_all:.3f}, Truthful: r={correlation_truthful:.3f}, Hallucinated: r={correlation_hallucinated:.3f}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/rq1_scatter_plot_reevaluated.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Visualizations saved to {output_dir}/")

def save_analysis_results(analysis_results: Dict, data: Dict, output_dir: str):
    """保存分析结果"""
    
    # 检查是否是重新评估的数据
    is_reevaluated = any(orig != new for orig, new in zip(data['original_hit_at_1'], data['hit_at_1']))
    
    # JSON格式结果
    json_results = {
        'timestamp': datetime.now().isoformat(),
        'evaluation_method': 'ChatGPT Re-evaluation' if is_reevaluated else 'Original Evaluation',
        'total_samples': len(data['tus_scores']),
        'truthful_samples': sum(data['hit_at_1']),
        'hallucinated_samples': len(data['hit_at_1']) - sum(data['hit_at_1']),
        'accuracy': np.mean(data['hit_at_1']),
        'metrics_analysis': {}
    }
    
    if is_reevaluated:
        json_results['original_accuracy'] = np.mean(data['original_hit_at_1'])
        json_results['changed_evaluations'] = sum(1 for orig, new in zip(data['original_hit_at_1'], data['hit_at_1']) if orig != new)
    
    for metric in ['TUS', 'NTUS', 'FGAS']:
        json_results['metrics_analysis'][metric] = {
            'truthful_mean': float(analysis_results[metric]['truthful']['mean']),
            'truthful_std': float(analysis_results[metric]['truthful']['std']),
            'hallucinated_mean': float(analysis_results[metric]['hallucinated']['mean']),
            'hallucinated_std': float(analysis_results[metric]['hallucinated']['std']),
            'p_value': float(analysis_results[metric]['p_value']),
            'cohens_d': float(analysis_results[metric]['cohens_d']),
            'significant': bool(analysis_results[metric]['significant'])
        }
    
    with open(f'{output_dir}/rq1_analysis_reevaluated.json', 'w', encoding='utf-8') as f:
        json.dump(json_results, f, indent=2, ensure_ascii=False)
    
    # 详细文本报告
    with open(f'{output_dir}/rq1_detailed_report_reevaluated.txt', 'w', encoding='utf-8') as f:
        report_title = "RQ1 统计分析报告 (ChatGPT评估版本)" if is_reevaluated else "RQ1 统计分析报告 (原始评估版本)"
        f.write(f"{report_title}\n")
        f.write("=" * 50 + "\n\n")
        
        f.write(f"分析时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"评估方法: {'ChatGPT评估' if is_reevaluated else '原始评估'}\n")
        f.write(f"总样本数: {len(data['tus_scores'])}\n")
        f.write(f"真实响应: {sum(data['hit_at_1'])}\n")
        f.write(f"幻觉响应: {len(data['hit_at_1']) - sum(data['hit_at_1'])}\n")
        f.write(f"准确率: {np.mean(data['hit_at_1']):.4f}\n")
        
        if is_reevaluated:
            f.write(f"原准确率: {np.mean(data['original_hit_at_1']):.4f}\n")
            f.write(f"变更评估: {sum(1 for orig, new in zip(data['original_hit_at_1'], data['hit_at_1']) if orig != new)}\n")
        
        f.write("\n")
        
        for metric in ['TUS', 'NTUS', 'FGAS']:
            result = analysis_results[metric]
            f.write(f"{metric} 分析:\n")
            f.write(f"  真实响应: {result['truthful']['mean']:.4f} ± {result['truthful']['std']:.4f}\n")
            f.write(f"  幻觉响应: {result['hallucinated']['mean']:.4f} ± {result['hallucinated']['std']:.4f}\n")
            f.write(f"  t统计量: {result['t_statistic']:.4f}\n")
            f.write(f"  p值: {result['p_value']:.6f}\n")
            f.write(f"  Cohen's d: {result['cohens_d']:.4f}\n")
            f.write(f"  显著性: {'是' if result['significant'] else '否'}\n\n")
        
        f.write("解释:\n")
        f.write("- p < 0.05 表示统计显著\n")
        f.write("- Cohen's d: 0.2(小), 0.5(中), 0.8(大)效应\n")
        f.write("- 正值表示真实响应分数更高\n")
    
    print(f"Analysis results saved to {output_dir}/")

def main():
    parser = argparse.ArgumentParser(description='RQ1 Analysis with Re-evaluated Data')
    parser.add_argument('--input', '-i', required=True, 
                       help='Re-evaluated inference results file')
    parser.add_argument('--output_dir', '-o', 
                       help='Output directory (default: rq1_reevaluated_TIMESTAMP)')
    
    args = parser.parse_args()
    
    # 设置输出目录
    if not args.output_dir:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = f"experiment_records/rq1_reevaluated_{timestamp}"
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    try:
        # 加载数据
        results, config = load_reevaluated_data(args.input)
        
        # 提取指标数据
        data = extract_metrics_data(results)
        
        # 执行统计分析
        analysis_results = perform_statistical_analysis(data)
        
        # 创建可视化
        create_visualizations(data, analysis_results, args.output_dir)
        
        # 保存结果
        save_analysis_results(analysis_results, data, args.output_dir)
        
        # 打印摘要
        print(f"\n=== RQ1 Analysis Summary (Re-evaluated) ===")
        for metric in ['TUS', 'NTUS', 'FGAS']:
            result = analysis_results[metric]
            significance = "***" if result['significant'] else ""
            print(f"{metric}: p={result['p_value']:.4f}{significance}, d={result['cohens_d']:.3f}")
        
        print(f"\nResults saved to: {args.output_dir}")
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 