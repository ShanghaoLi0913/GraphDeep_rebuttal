"""
RQ1实验：分析LLM对gold-relevant triples的注意力利用与幻觉的关系
增强版：同时分析TUS、NTUS和FGAS指标
"""

import json
import os
import logging
import numpy as np
from datetime import datetime
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

def plot_rq1_figures(truthful_tus, hallucinated_tus, truthful_ntus, hallucinated_ntus, truthful_fgas, hallucinated_fgas, output_dir):
    """
    绘制RQ1分析图表：包含TUS、NTUS和FGAS的对比分析
    """
    # 设置图表样式
    plt.style.use('seaborn-v0_8')
    
    # 1. KDE密度图：TUS、NTUS和FGAS分布对比
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # TUS KDE图
    ax1 = axes[0]
    if len(truthful_tus) > 0 and len(hallucinated_tus) > 0:
        ax1.hist(truthful_tus, alpha=0.7, density=True, bins=20, label='Truthful', color='green')
        ax1.hist(hallucinated_tus, alpha=0.7, density=True, bins=20, label='Hallucinated', color='red')
        ax1.set_xlabel('TUS Score')
        ax1.set_ylabel('Density')
        ax1.set_title('TUS Score Distribution: Truthful vs Hallucinated')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
    
    # NTUS KDE图
    ax2 = axes[1]
    if len(truthful_ntus) > 0 and len(hallucinated_ntus) > 0:
        ax2.hist(truthful_ntus, alpha=0.7, density=True, bins=20, label='Truthful', color='green')
        ax2.hist(hallucinated_ntus, alpha=0.7, density=True, bins=20, label='Hallucinated', color='red')
        ax2.set_xlabel('NTUS Score')
        ax2.set_ylabel('Density')
        ax2.set_title('NTUS Score Distribution: Truthful vs Hallucinated')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    # FGAS KDE图
    ax3 = axes[2]
    if len(truthful_fgas) > 0 and len(hallucinated_fgas) > 0:
        ax3.hist(truthful_fgas, alpha=0.7, density=True, bins=20, label='Truthful', color='green')
        ax3.hist(hallucinated_fgas, alpha=0.7, density=True, bins=20, label='Hallucinated', color='red')
        ax3.set_xlabel('FGAS Score')
        ax3.set_ylabel('Density')
        ax3.set_title('FGAS Score Distribution: Truthful vs Hallucinated')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    kde_path = os.path.join(output_dir, 'rq1_kde_distributions.png')
    plt.savefig(kde_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. 箱线图：TUS、NTUS和FGAS对比
    fig, axes = plt.subplots(1, 3, figsize=(15, 6))
    
    # TUS箱线图
    ax1 = axes[0]
    tus_data = [truthful_tus, hallucinated_tus]
    labels = ['Truthful', 'Hallucinated']
    bp1 = ax1.boxplot(tus_data, labels=labels, patch_artist=True)
    bp1['boxes'][0].set_facecolor('lightgreen')
    bp1['boxes'][1].set_facecolor('lightcoral')
    ax1.set_ylabel('TUS Score')
    ax1.set_title('TUS Score: Truthful vs Hallucinated')
    ax1.grid(True, alpha=0.3)
    
    # NTUS箱线图
    ax2 = axes[1]
    ntus_data = [truthful_ntus, hallucinated_ntus]
    bp2 = ax2.boxplot(ntus_data, labels=labels, patch_artist=True)
    bp2['boxes'][0].set_facecolor('lightgreen')
    bp2['boxes'][1].set_facecolor('lightcoral')
    ax2.set_ylabel('NTUS Score')
    ax2.set_title('NTUS Score: Truthful vs Hallucinated')
    ax2.grid(True, alpha=0.3)
    
    # FGAS箱线图
    ax3 = axes[2]
    fgas_data = [truthful_fgas, hallucinated_fgas]
    bp3 = ax3.boxplot(fgas_data, labels=labels, patch_artist=True)
    bp3['boxes'][0].set_facecolor('lightgreen')
    bp3['boxes'][1].set_facecolor('lightcoral')
    ax3.set_ylabel('FGAS Score')
    ax3.set_title('FGAS Score: Truthful vs Hallucinated')
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    box_path = os.path.join(output_dir, 'rq1_boxplots.png')
    plt.savefig(box_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. 均值对比柱状图
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    
    x = np.arange(3)  # TUS, NTUS, FGAS
    width = 0.35
    
    truthful_means = [np.mean(truthful_tus), np.mean(truthful_ntus), np.mean(truthful_fgas)]
    hallucinated_means = [np.mean(hallucinated_tus), np.mean(hallucinated_ntus), np.mean(hallucinated_fgas)]
    truthful_stds = [np.std(truthful_tus), np.std(truthful_ntus), np.std(truthful_fgas)]
    hallucinated_stds = [np.std(hallucinated_tus), np.std(hallucinated_ntus), np.std(hallucinated_fgas)]
    
    bars1 = ax.bar(x - width/2, truthful_means, width, yerr=truthful_stds, 
                   label='Truthful', color='green', alpha=0.7, capsize=5)
    bars2 = ax.bar(x + width/2, hallucinated_means, width, yerr=hallucinated_stds,
                   label='Hallucinated', color='red', alpha=0.7, capsize=5)
    
    ax.set_ylabel('Score')
    ax.set_title('Mean Scores: Truthful vs Hallucinated Responses')
    ax.set_xticks(x)
    ax.set_xticklabels(['TUS', 'NTUS', 'FGAS'])
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 添加数值标签
    for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
        height1 = bar1.get_height()
        height2 = bar2.get_height()
        ax.text(bar1.get_x() + bar1.get_width()/2., height1 + truthful_stds[i],
                f'{height1:.3f}', ha='center', va='bottom', fontsize=10)
        ax.text(bar2.get_x() + bar2.get_width()/2., height2 + hallucinated_stds[i],
                f'{height2:.3f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    bar_path = os.path.join(output_dir, 'rq1_mean_comparison.png')
    plt.savefig(bar_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. 散点图：TUS vs FGAS
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    ax.scatter(truthful_tus, truthful_fgas, alpha=0.6, label='Truthful', 
               color='green', s=50)
    ax.scatter(hallucinated_tus, hallucinated_fgas, alpha=0.6, label='Hallucinated', 
               color='red', s=50)
    
    ax.set_xlabel('TUS Score')
    ax.set_ylabel('FGAS Score')
    ax.set_title('TUS vs FGAS: Truthful vs Hallucinated Responses')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 添加相关性分析
    if len(truthful_tus) > 1 and len(truthful_fgas) > 1:
        truthful_corr = np.corrcoef(truthful_tus, truthful_fgas)[0, 1]
        ax.text(0.05, 0.95, f'Truthful Correlation: {truthful_corr:.3f}', 
                transform=ax.transAxes, bbox=dict(boxstyle="round", facecolor='lightgreen'))
    
    if len(hallucinated_tus) > 1 and len(hallucinated_fgas) > 1:
        hallucinated_corr = np.corrcoef(hallucinated_tus, hallucinated_fgas)[0, 1]
        ax.text(0.05, 0.85, f'Hallucinated Correlation: {hallucinated_corr:.3f}', 
                transform=ax.transAxes, bbox=dict(boxstyle="round", facecolor='lightcoral'))
    
    plt.tight_layout()
    scatter_path = os.path.join(output_dir, 'rq1_scatter_tus_fgas.png')
    plt.savefig(scatter_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return kde_path, box_path, bar_path, scatter_path

def run_rq1_experiment(args):
    """
    运行RQ1实验：分析LLM对gold-relevant triples的注意力与幻觉的关系
    增强版：同时分析TUS、NTUS和FGAS指标
    """
    logger = logging.getLogger(__name__)
    logger.info("Starting enhanced RQ1 experiment (TUS + NTUS + FGAS)...")
    
    # 创建输出目录
    output_dir = os.path.join('experiment_records', f'rq1_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # 使用提供的结果文件
        results_file = args.results_file
        logger.info(f"Reading file: {results_file}")
        logger.info(f"File absolute path: {os.path.abspath(results_file)}")
        logger.info(f"File exists: {os.path.exists(results_file)}")
        logger.info(f"File size: {os.path.getsize(results_file)} bytes")
        
        # 检查文件的前两行内容
        with open(results_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            logger.info(f"First line preview: {lines[0].strip()[:100]}...")
            if len(lines) > 1:
                logger.info(f"Second line preview: {lines[1].strip()[:100]}...")
                # 检查第二行是否包含ntus_score
                has_ntus = 'ntus_score' in lines[1]
                logger.info(f"Second line contains 'ntus_score': {has_ntus}")
        
        # 分析结果
        truthful_tus, hallucinated_tus = [], []
        truthful_ntus, hallucinated_ntus = [], []
        truthful_fgas, hallucinated_fgas = [], []
        
        processed_count = 0
        with open(results_file, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                data = json.loads(line)
                if 'config' in data or 'stats' in data:
                    continue
                    
                processed_count += 1
                
                # 获取TUS、nTUS和FGAS分数以及Hit@1结果
                tus_score = data.get('tus_score', 0.0)
                ntus_score = data.get('ntus_score', 0.0)
                fgas_score = data.get('fgas_score', 0.0)
                is_correct = data.get('metrics', {}).get('hit@1', False)
                
                # 调试前5个样本
                if processed_count <= 5:
                    logger.info(f"Sample {processed_count}: TUS={tus_score:.4f}, NTUS={ntus_score:.4f}, FGAS={fgas_score:.4f}, Hit@1={is_correct}")
                    logger.info(f"  Question: {data.get('question', 'MISSING')[:50]}...")
                    logger.info(f"  Raw ntus_score: {repr(data.get('ntus_score'))}")
                    logger.info(f"  All keys: {list(data.keys())}")
                
                # 根据Hit@1结果分组
                if is_correct:
                    truthful_tus.append(tus_score)
                    truthful_ntus.append(ntus_score)
                    truthful_fgas.append(fgas_score)
                else:
                    hallucinated_tus.append(tus_score)
                    hallucinated_ntus.append(ntus_score)
                    hallucinated_fgas.append(fgas_score)
        
        logger.info(f"Processed {processed_count} samples")
        logger.info(f"Truthful samples: {len(truthful_tus)}, Hallucinated samples: {len(hallucinated_tus)}")
        logger.info(f"Sample NTUS values - Truthful: {truthful_ntus[:3]}, Hallucinated: {hallucinated_ntus[:3]}")
        
        # 计算统计指标
        stats_dict = {
            'truthful': {
                'count': len(truthful_tus),
                'tus': {
                    'mean': np.mean(truthful_tus) if len(truthful_tus) > 0 else 0.0,
                    'std': np.std(truthful_tus) if len(truthful_tus) > 0 else 0.0,
                    'median': np.median(truthful_tus) if len(truthful_tus) > 0 else 0.0
                },
                'ntus': {
                    'mean': np.mean(truthful_ntus) if len(truthful_ntus) > 0 else 0.0,
                    'std': np.std(truthful_ntus) if len(truthful_ntus) > 0 else 0.0,
                    'median': np.median(truthful_ntus) if len(truthful_ntus) > 0 else 0.0
                },
                'fgas': {
                    'mean': np.mean(truthful_fgas) if len(truthful_fgas) > 0 else 0.0,
                    'std': np.std(truthful_fgas) if len(truthful_fgas) > 0 else 0.0,
                    'median': np.median(truthful_fgas) if len(truthful_fgas) > 0 else 0.0
                }
            },
            'hallucinated': {
                'count': len(hallucinated_tus),
                'tus': {
                    'mean': np.mean(hallucinated_tus) if len(hallucinated_tus) > 0 else 0.0,
                    'std': np.std(hallucinated_tus) if len(hallucinated_tus) > 0 else 0.0,
                    'median': np.median(hallucinated_tus) if len(hallucinated_tus) > 0 else 0.0
                },
                'ntus': {
                    'mean': np.mean(hallucinated_ntus) if len(hallucinated_ntus) > 0 else 0.0,
                    'std': np.std(hallucinated_ntus) if len(hallucinated_ntus) > 0 else 0.0,
                    'median': np.median(hallucinated_ntus) if len(hallucinated_ntus) > 0 else 0.0
                },
                'fgas': {
                    'mean': np.mean(hallucinated_fgas) if len(hallucinated_fgas) > 0 else 0.0,
                    'std': np.std(hallucinated_fgas) if len(hallucinated_fgas) > 0 else 0.0,
                    'median': np.median(hallucinated_fgas) if len(hallucinated_fgas) > 0 else 0.0
                }
            }
        }
        
        # 进行t检验
        tus_t_stat, tus_p_value = stats.ttest_ind(truthful_tus, hallucinated_tus)
        ntus_t_stat, ntus_p_value = stats.ttest_ind(truthful_ntus, hallucinated_ntus)
        fgas_t_stat, fgas_p_value = stats.ttest_ind(truthful_fgas, hallucinated_fgas)
        
        stats_dict['statistical_tests'] = {
            'tus_t_test': {
                't_statistic': tus_t_stat,
                'p_value': tus_p_value
            },
            'ntus_t_test': {
                't_statistic': ntus_t_stat,
                'p_value': ntus_p_value
            },
            'fgas_t_test': {
                't_statistic': fgas_t_stat,
                'p_value': fgas_p_value
            }
        }
        
        # 计算效应大小 (Cohen's d)
        def cohens_d(group1, group2):
            n1, n2 = len(group1), len(group2)
            pooled_std = np.sqrt(((n1-1)*np.var(group1) + (n2-1)*np.var(group2)) / (n1+n2-2))
            return (np.mean(group1) - np.mean(group2)) / pooled_std
        
        if len(truthful_tus) > 1 and len(hallucinated_tus) > 1:
            tus_cohens_d = cohens_d(truthful_tus, hallucinated_tus)
            ntus_cohens_d = cohens_d(truthful_ntus, hallucinated_ntus)
            fgas_cohens_d = cohens_d(truthful_fgas, hallucinated_fgas)
        else:
            tus_cohens_d = 0.0
            ntus_cohens_d = 0.0
            fgas_cohens_d = 0.0
        
        stats_dict['effect_sizes'] = {
            'tus_cohens_d': tus_cohens_d,
            'ntus_cohens_d': ntus_cohens_d,
            'fgas_cohens_d': fgas_cohens_d
        }
        
        # 绘制所有图表
        kde_path, box_path, bar_path, scatter_path = plot_rq1_figures(
            truthful_tus, hallucinated_tus, truthful_ntus, hallucinated_ntus, 
            truthful_fgas, hallucinated_fgas, output_dir)
        
        # 保存分析结果
        analysis_file = os.path.join(output_dir, 'rq1_analysis.json')
        with open(analysis_file, 'w') as f:
            json.dump({
                'statistics': stats_dict,
                'plots': {
                    'kde_plot': kde_path,
                    'box_plot': box_path,
                    'bar_plot': bar_path,
                    'scatter_plot': scatter_path
                },
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }, f, indent=2)
        
        # 生成详细报告
        report_file = os.path.join(output_dir, 'rq1_detailed_report.txt')
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("RQ1 详细分析报告：TUS & FGAS vs 幻觉关系\n")
            f.write("=" * 60 + "\n\n")
            
            f.write(f"数据概况：\n")
            f.write(f"- 真实响应样本数: {stats_dict['truthful']['count']}\n")
            f.write(f"- 幻觉响应样本数: {stats_dict['hallucinated']['count']}\n")
            f.write(f"- 总样本数: {stats_dict['truthful']['count'] + stats_dict['hallucinated']['count']}\n")
            f.write(f"- 准确率: {stats_dict['truthful']['count'] / (stats_dict['truthful']['count'] + stats_dict['hallucinated']['count']) * 100:.2f}%\n\n")
            
            f.write("TUS (Triple Utilization Score) 分析：\n")
            f.write("-" * 40 + "\n")
            f.write(f"真实响应 TUS: {stats_dict['truthful']['tus']['mean']:.4f} ± {stats_dict['truthful']['tus']['std']:.4f}\n")
            f.write(f"幻觉响应 TUS: {stats_dict['hallucinated']['tus']['mean']:.4f} ± {stats_dict['hallucinated']['tus']['std']:.4f}\n")
            f.write(f"差异: {stats_dict['truthful']['tus']['mean'] - stats_dict['hallucinated']['tus']['mean']:.4f}\n")
            f.write(f"效应大小 (Cohen's d): {tus_cohens_d:.4f}\n")
            f.write(f"t检验 p值: {tus_p_value:.6f}\n")
            significance = "显著" if tus_p_value < 0.05 else "不显著"
            f.write(f"统计显著性: {significance}\n\n")
            
            f.write("NTUS (Normalized Triple Utilization Score) 分析：\n")
            f.write("-" * 40 + "\n")
            f.write(f"真实响应 NTUS: {stats_dict['truthful']['ntus']['mean']:.4f} ± {stats_dict['truthful']['ntus']['std']:.4f}\n")
            f.write(f"幻觉响应 NTUS: {stats_dict['hallucinated']['ntus']['mean']:.4f} ± {stats_dict['hallucinated']['ntus']['std']:.4f}\n")
            f.write(f"差异: {stats_dict['truthful']['ntus']['mean'] - stats_dict['hallucinated']['ntus']['mean']:.4f}\n")
            f.write(f"效应大小 (Cohen's d): {ntus_cohens_d:.4f}\n")
            f.write(f"t检验 p值: {ntus_p_value:.6f}\n")
            significance = "显著" if ntus_p_value < 0.05 else "不显著"
            f.write(f"统计显著性: {significance}\n\n")
            
            f.write("FGAS (FFN-Gold Alignment Score) 分析：\n")
            f.write("-" * 40 + "\n")
            f.write(f"真实响应 FGAS: {stats_dict['truthful']['fgas']['mean']:.4f} ± {stats_dict['truthful']['fgas']['std']:.4f}\n")
            f.write(f"幻觉响应 FGAS: {stats_dict['hallucinated']['fgas']['mean']:.4f} ± {stats_dict['hallucinated']['fgas']['std']:.4f}\n")
            f.write(f"差异: {stats_dict['truthful']['fgas']['mean'] - stats_dict['hallucinated']['fgas']['mean']:.4f}\n")
            f.write(f"效应大小 (Cohen's d): {fgas_cohens_d:.4f}\n")
            f.write(f"t检验 p值: {fgas_p_value:.6f}\n")
            significance = "显著" if fgas_p_value < 0.05 else "不显著"
            f.write(f"统计显著性: {significance}\n\n")
            
            f.write("RQ1 结论：\n")
            f.write("-" * 40 + "\n")
            f.write("基于TUS、NTUS和FGAS指标的分析，我们发现：\n\n")
            
            if tus_p_value < 0.05 and stats_dict['truthful']['tus']['mean'] > stats_dict['hallucinated']['tus']['mean']:
                f.write("1. TUS结果：真实响应显著具有更高的三元组利用分数，说明正确的回答更多地依赖于外部检索到的知识图谱信息。\n")
            else:
                f.write("1. TUS结果：真实响应与幻觉响应在三元组利用分数上无显著差异。\n")
                
            if ntus_p_value < 0.05 and stats_dict['truthful']['ntus']['mean'] > stats_dict['hallucinated']['ntus']['mean']:
                f.write("2. NTUS结果：真实响应显著具有更高的归一化三元组利用分数，说明正确的回答在考虑生成长度后仍更多利用外部知识。\n")
            else:
                f.write("2. NTUS结果：真实响应与幻觉响应在归一化三元组利用分数上无显著差异。\n")
            
            if fgas_p_value < 0.05 and stats_dict['truthful']['fgas']['mean'] > stats_dict['hallucinated']['fgas']['mean']:
                f.write("3. FGAS结果：真实响应显著具有更高的FFN-Gold对齐分数，说明正确的回答中FFN层表示与关键知识更好对齐。\n")
            else:
                f.write("3. FGAS结果：真实响应与幻觉响应在FFN-Gold对齐分数上无显著差异。\n")
            
            f.write("\n这些发现支持/质疑了以下假设：\n")
            f.write("- 幻觉发生时模型较少利用外部检索内容（基于TUS和NTUS）\n")
            f.write("- 幻觉发生时模型内部表示与关键知识对齐度较低（基于FGAS）\n")
        
        # 打印主要发现
        logger.info("\n=== Enhanced RQ1 Analysis Results ===")
        logger.info(f"Number of samples - Truthful: {stats_dict['truthful']['count']}, "
                   f"Hallucinated: {stats_dict['hallucinated']['count']}")
        logger.info(f"Mean TUS - Truthful: {stats_dict['truthful']['tus']['mean']:.4f}±{stats_dict['truthful']['tus']['std']:.4f}, "
                   f"Hallucinated: {stats_dict['hallucinated']['tus']['mean']:.4f}±{stats_dict['hallucinated']['tus']['std']:.4f}")
        logger.info(f"Mean NTUS - Truthful: {stats_dict['truthful']['ntus']['mean']:.4f}±{stats_dict['truthful']['ntus']['std']:.4f}, "
                   f"Hallucinated: {stats_dict['hallucinated']['ntus']['mean']:.4f}±{stats_dict['hallucinated']['ntus']['std']:.4f}")
        logger.info(f"Mean FGAS - Truthful: {stats_dict['truthful']['fgas']['mean']:.4f}±{stats_dict['truthful']['fgas']['std']:.4f}, "
                   f"Hallucinated: {stats_dict['hallucinated']['fgas']['mean']:.4f}±{stats_dict['hallucinated']['fgas']['std']:.4f}")
        logger.info(f"TUS t-test p-value: {tus_p_value:.6f}")
        logger.info(f"NTUS t-test p-value: {ntus_p_value:.6f}")
        logger.info(f"FGAS t-test p-value: {fgas_p_value:.6f}")
        logger.info(f"Analysis results saved to {analysis_file}")
        logger.info(f"Detailed report saved to {report_file}")
        logger.info(f"Plots saved to {output_dir}")
        
        return analysis_file
    except Exception as e:
        logger.error(f"运行RQ1实验时出错: {str(e)}")
        return None

# 如果作为独立脚本运行
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='运行RQ1实验')
    parser.add_argument('--results_file', type=str, 
                        default='experiment_records/inference_results_20250623_204903.jsonl',
                        help='推理结果文件路径')
    
    # 设置日志
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # 解析命令行参数
    args = parser.parse_args()
    result = run_rq1_experiment(args) 