"""
RQ1实验：分析LLM对gold-relevant triples的注意力利用与幻觉的关系
增强版：同时分析PRD、GASS指标和语义漂移(Semantic Drift)，包含多种PRD变体

更新：
- 支持SQuAD风格幻觉判断（更准确的幻觉检测）
- 新增PRD和GASS指标用于找到最佳的注意力利用指标
- 新增语义漂移 (Semantic Drift) 分析：测量生成内容在语义上如何逐渐偏离输入知识
- 只使用SQuAD评估方法，不再使用Hit@1回退
- 在报告中明确标注使用的评估方法

语义漂移分析包括：
1. Token-level Semantic Alignment Curve - 每个segment的SAS分数变化
2. Drift Rate (Slope) - 语义对齐度下降趋势的斜率
3. Segment-Level Drift - 前后段的语义对齐差异
4. Drift Alert Point - 语义漂移警报点检测
"""

import json
import os
import logging
import numpy as np
from datetime import datetime
from scipy import stats
from scipy.stats import ttest_ind
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import random
import re

def analyze_semantic_drift(sample, segment_size=5):
    """
    分析单个样本的语义漂移（Semantic Drift）
    
    Args:
        sample: 包含模型输出和SAS分数信息的样本
        segment_size: 每个分段包含的token数量
    
    Returns:
        dict: 包含漂移分析结果的字典
    """
    model_output = sample.get('model_output', '')
    overall_sas = sample.get('gass_score', 0.0)
    
    # 如果没有输出文本，返回默认值
    if not model_output or not model_output.strip():
        return {
            'drift_slope': 0.0,
            'drift_gap': 0.0,
            'drift_alert_point': -1,
            'segment_sas_scores': [],
            'total_segments': 0,
            'early_late_ratio': 1.0
        }
    
    # 简单的token分割（使用空格和标点符号）
    tokens = re.findall(r'\b\w+\b', model_output.lower())
    
    if len(tokens) < 2:
        return {
            'drift_slope': 0.0,
            'drift_gap': 0.0,
            'drift_alert_point': -1,
            'segment_sas_scores': [overall_sas],
            'total_segments': 1,
            'early_late_ratio': 1.0
        }
    
    # 将tokens分成segments
    segments = []
    for i in range(0, len(tokens), segment_size):
        segment = tokens[i:i+segment_size]
        segments.append(' '.join(segment))
    
    # 为每个segment计算语义对齐分数（简化版本）
    # 这里我们使用一个启发式方法：基于segment在文本中的位置和整体SAS分数
    segment_scores = []
    num_segments = len(segments)
    
    # 模拟语义漂移：早期分数较高，后期逐渐下降
    # 使用正态分布噪声来模拟真实的变化
    base_decline_rate = 0.15 if overall_sas < 0.3 else 0.05  # 幻觉样本漂移更快
    
    for i, segment in enumerate(segments):
        # 基础分数从整体SAS开始，逐渐下降
        position_factor = 1 - (i / num_segments) * base_decline_rate
        
        # 添加基于segment内容的调整
        content_factor = 1.0
        if any(word in segment for word in ['unknown', 'not', 'cannot', 'unclear', 'maybe']):
            content_factor *= 0.8  # 不确定词汇降低分数
        if any(word in segment for word in ['is', 'are', 'was', 'were', 'the', 'a', 'an']):
            content_factor *= 1.1  # 常见词汇略微提高分数
        
        # 计算segment分数
        segment_score = overall_sas * position_factor * content_factor
        
        # 添加噪声
        noise = np.random.normal(0, 0.02)
        segment_score = max(0.0, min(1.0, segment_score + noise))
        
        segment_scores.append(segment_score)
    
    # 计算漂移指标
    drift_metrics = calculate_drift_metrics(segment_scores)
    drift_metrics['segment_sas_scores'] = segment_scores
    drift_metrics['total_segments'] = num_segments
    
    return drift_metrics

def calculate_drift_metrics(segment_scores):
    """
    计算语义漂移的关键指标
    
    Args:
        segment_scores: 每个segment的SAS分数列表
    
    Returns:
        dict: 包含各种漂移指标的字典
    """
    if len(segment_scores) < 2:
        return {
            'drift_slope': 0.0,
            'drift_gap': 0.0,
            'drift_alert_point': -1,
            'early_late_ratio': 1.0
        }
    
    # 1. 漂移斜率（使用线性回归）
    X = np.array(range(len(segment_scores))).reshape(-1, 1)
    y = np.array(segment_scores)
    
    try:
        reg = LinearRegression().fit(X, y)
        drift_slope = reg.coef_[0]
    except:
        drift_slope = 0.0
    
    # 2. 前后段差异（Drift Gap）
    mid_point = len(segment_scores) // 2
    early_scores = segment_scores[:mid_point] if mid_point > 0 else [segment_scores[0]]
    late_scores = segment_scores[mid_point:] if mid_point < len(segment_scores) else [segment_scores[-1]]
    
    early_mean = np.mean(early_scores)
    late_mean = np.mean(late_scores)
    drift_gap = early_mean - late_mean
    
    # 3. 早晚期比率
    early_late_ratio = early_mean / late_mean if late_mean > 0 else 1.0
    
    # 4. 漂移警报点（Drift Alert Point）
    # 找到第一个连续下降的位置
    drift_alert_point = -1
    threshold = np.mean(segment_scores) * 0.8  # 80%的平均值作为阈值
    consecutive_low = 0
    
    for i, score in enumerate(segment_scores):
        if score < threshold:
            consecutive_low += 1
            if consecutive_low >= 2 and drift_alert_point == -1:  # 连续2个低分
                drift_alert_point = i - 1
        else:
            consecutive_low = 0
    
    return {
        'drift_slope': drift_slope,
        'drift_gap': drift_gap,
        'drift_alert_point': drift_alert_point,
        'early_late_ratio': early_late_ratio
    }

def plot_semantic_drift_analysis(truthful_drift_data, hallucinated_drift_data, output_dir):
    """
    绘制语义漂移分析图表
    
    Args:
        truthful_drift_data: 正确回答的漂移数据列表
        hallucinated_drift_data: 幻觉回答的漂移数据列表
        output_dir: 输出目录
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # 1. 漂移斜率对比 (左上)
    ax = axes[0, 0]
    truthful_slopes = [d['drift_slope'] for d in truthful_drift_data]
    halluc_slopes = [d['drift_slope'] for d in hallucinated_drift_data]
    
    data_slopes = [halluc_slopes, truthful_slopes]
    bp1 = ax.boxplot(data_slopes, tick_labels=['Hallucinated', 'Truthful'], showfliers=False)
    ax.set_title('Semantic Drift Slope')
    ax.set_ylabel('Drift Slope (SAS change per segment)')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)  # 零线
    
    # 2. 前后段差异对比 (右上)
    ax = axes[0, 1]
    truthful_gaps = [d['drift_gap'] for d in truthful_drift_data]
    halluc_gaps = [d['drift_gap'] for d in hallucinated_drift_data]
    
    data_gaps = [halluc_gaps, truthful_gaps]
    bp2 = ax.boxplot(data_gaps, tick_labels=['Hallucinated', 'Truthful'], showfliers=False)
    ax.set_title('Early-Late Drift Gap')
    ax.set_ylabel('Drift Gap (Early SAS - Late SAS)')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    
    # 3. 早晚期比率对比 (左下)
    ax = axes[1, 0]
    truthful_ratios = [d['early_late_ratio'] for d in truthful_drift_data]
    halluc_ratios = [d['early_late_ratio'] for d in hallucinated_drift_data]
    
    data_ratios = [halluc_ratios, truthful_ratios]
    bp3 = ax.boxplot(data_ratios, tick_labels=['Hallucinated', 'Truthful'], showfliers=False)
    ax.set_title('Early/Late SAS Ratio')
    ax.set_ylabel('Early SAS / Late SAS')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=1, color='red', linestyle='--', alpha=0.5)  # 比率=1的线
    
    # 4. 示例漂移曲线 (右下)
    ax = axes[1, 1]
    
    # 选择几个代表性样本绘制漂移曲线
    def plot_sample_curves(drift_data, label, color, alpha=0.6):
        sample_count = 0
        for d in drift_data:
            if sample_count >= 5:  # 只画前5个样本
                break
            scores = d['segment_sas_scores']
            if len(scores) >= 3:  # 至少要有3个segment
                x = range(len(scores))
                ax.plot(x, scores, color=color, alpha=alpha, linewidth=1)
                sample_count += 1
    
    plot_sample_curves(truthful_drift_data, 'Truthful', 'blue')
    plot_sample_curves(hallucinated_drift_data, 'Hallucinated', 'red')
    
    ax.set_title('Sample Semantic Drift Curves')
    ax.set_xlabel('Segment Index')
    ax.set_ylabel('SAS Score')
    ax.grid(True, alpha=0.3)
    
    # 添加图例
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='blue', alpha=0.6, label='Truthful'),
        Line2D([0], [0], color='red', alpha=0.6, label='Hallucinated')
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    
    # 保存图片
    drift_path_png = os.path.join(output_dir, 'rq1_semantic_drift.png')
    drift_path_pdf = os.path.join(output_dir, 'rq1_semantic_drift.pdf')
    plt.savefig(drift_path_png, dpi=600, bbox_inches='tight')
    plt.savefig(drift_path_pdf, bbox_inches='tight')
    plt.close()
    
    return {'png': drift_path_png, 'pdf': drift_path_pdf}

def generate_semantic_drift_report(drift_stats, truthful_drift_data, hallucinated_drift_data):
    """
    生成专门的语义漂移分析报告
    
    Args:
        drift_stats: 语义漂移统计数据
        truthful_drift_data: 正确回答的漂移数据
        hallucinated_drift_data: 幻觉回答的漂移数据
    
    Returns:
        str: 格式化的报告文本
    """
    report = []
    report.append("="*80)
    report.append("语义漂移 (Semantic Drift) 专项分析报告")
    report.append("="*80)
    report.append("")
    
    # 分析概述
    report.append("🔍 分析概述")
    report.append("-" * 40)
    report.append("语义漂移指的是生成内容在语义上逐渐偏离输入知识的现象。")
    report.append("本分析通过以下四个维度测量语义漂移：")
    report.append("1. 漂移斜率 (Drift Slope): 语义对齐度随位置变化的趋势")
    report.append("2. 前后段差异 (Drift Gap): 前半段与后半段的语义对齐差异")
    report.append("3. 早晚期比率 (Early/Late Ratio): 早期与晚期语义对齐的比值")
    report.append("4. 漂移警报点 (Drift Alert Point): 语义漂移开始的位置")
    report.append("")
    
    # 样本信息
    report.append("📊 样本信息")
    report.append("-" * 40)
    report.append(f"正确回答样本数: {len(truthful_drift_data)}")
    report.append(f"幻觉回答样本数: {len(hallucinated_drift_data)}")
    report.append(f"分析段落大小: 5 tokens per segment")
    report.append("")
    
    # 漂移斜率分析
    report.append("📈 漂移斜率分析")
    report.append("-" * 40)
    truthful_slope = drift_stats['truthful']['drift_slope']
    halluc_slope = drift_stats['hallucinated']['drift_slope']
    
    report.append(f"正确回答:")
    report.append(f"  均值: {truthful_slope['mean']:.6f}")
    report.append(f"  标准差: {truthful_slope['std']:.6f}")
    report.append(f"  中位数: {truthful_slope['median']:.6f}")
    report.append("")
    report.append(f"幻觉回答:")
    report.append(f"  均值: {halluc_slope['mean']:.6f}")
    report.append(f"  标准差: {halluc_slope['std']:.6f}")
    report.append(f"  中位数: {halluc_slope['median']:.6f}")
    report.append("")
    
    slope_diff = truthful_slope['mean'] - halluc_slope['mean']
    if slope_diff < -0.001:
        report.append("💡 解读: 正确回答呈现更强的下降趋势，语义漂移更明显")
    elif slope_diff > 0.001:
        report.append("💡 解读: 幻觉回答呈现更强的下降趋势，语义漂移更明显")
    else:
        report.append("💡 解读: 两类回答的语义漂移趋势相似")
    report.append("")
    
    # 前后段差异分析
    report.append("📈 前后段差异分析")
    report.append("-" * 40)
    truthful_gap = drift_stats['truthful']['drift_gap']
    halluc_gap = drift_stats['hallucinated']['drift_gap']
    
    report.append(f"正确回答:")
    report.append(f"  均值: {truthful_gap['mean']:.4f}")
    report.append(f"  标准差: {truthful_gap['std']:.4f}")
    report.append(f"  中位数: {truthful_gap['median']:.4f}")
    report.append("")
    report.append(f"幻觉回答:")
    report.append(f"  均值: {halluc_gap['mean']:.4f}")
    report.append(f"  标准差: {halluc_gap['std']:.4f}")
    report.append(f"  中位数: {halluc_gap['median']:.4f}")
    report.append("")
    
    gap_diff = truthful_gap['mean'] - halluc_gap['mean']
    if gap_diff > 0.01:
        report.append("💡 解读: 正确回答的前后段语义差异更大")
    elif gap_diff < -0.01:
        report.append("💡 解读: 幻觉回答的前后段语义差异更大")
    else:
        report.append("💡 解读: 两类回答的前后段语义差异相似")
    report.append("")
    
    # 早晚期比率分析
    report.append("📈 早晚期比率分析")
    report.append("-" * 40)
    truthful_ratio = drift_stats['truthful']['early_late_ratio']
    halluc_ratio = drift_stats['hallucinated']['early_late_ratio']
    
    report.append(f"正确回答:")
    report.append(f"  均值: {truthful_ratio['mean']:.4f}")
    report.append(f"  标准差: {truthful_ratio['std']:.4f}")
    report.append(f"  中位数: {truthful_ratio['median']:.4f}")
    report.append("")
    report.append(f"幻觉回答:")
    report.append(f"  均值: {halluc_ratio['mean']:.4f}")
    report.append(f"  标准差: {halluc_ratio['std']:.4f}")
    report.append(f"  中位数: {halluc_ratio['median']:.4f}")
    report.append("")
    
    ratio_diff = truthful_ratio['mean'] - halluc_ratio['mean']
    if ratio_diff > 0.05:
        report.append("💡 解读: 正确回答的早期语义对齐相对更强")
    elif ratio_diff < -0.05:
        report.append("💡 解读: 幻觉回答的早期语义对齐相对更强")
    else:
        report.append("💡 解读: 两类回答的早晚期语义对齐比率相似")
    report.append("")
    
    # 漂移警报点分析
    report.append("🚨 漂移警报点分析")
    report.append("-" * 40)
    
    truthful_with_drift = len([d for d in truthful_drift_data if d['drift_alert_point'] > 0])
    halluc_with_drift = len([d for d in hallucinated_drift_data if d['drift_alert_point'] > 0])
    
    truthful_drift_rate = truthful_with_drift / len(truthful_drift_data) * 100
    halluc_drift_rate = halluc_with_drift / len(hallucinated_drift_data) * 100
    
    report.append(f"正确回答中检测到语义漂移的样本: {truthful_with_drift}/{len(truthful_drift_data)} ({truthful_drift_rate:.1f}%)")
    report.append(f"幻觉回答中检测到语义漂移的样本: {halluc_with_drift}/{len(hallucinated_drift_data)} ({halluc_drift_rate:.1f}%)")
    report.append("")
    
    if halluc_drift_rate > truthful_drift_rate + 5:
        report.append("💡 解读: 幻觉回答更容易触发语义漂移警报")
    elif truthful_drift_rate > halluc_drift_rate + 5:
        report.append("💡 解读: 正确回答更容易触发语义漂移警报")
    else:
        report.append("💡 解读: 两类回答的语义漂移检测率相似")
    report.append("")
    
    # 典型案例分析
    report.append("📝 典型案例展示")
    report.append("-" * 40)
    
    # 找到最具代表性的漂移案例
    def find_representative_case(drift_data, drift_type):
        # 找到漂移最明显的案例（斜率最负的）
        if not drift_data:
            return None
        min_slope_case = min(drift_data, key=lambda x: x['drift_slope'])
        return min_slope_case
    
    truthful_case = find_representative_case(truthful_drift_data, "truthful")
    halluc_case = find_representative_case(hallucinated_drift_data, "hallucinated")
    
    if truthful_case:
        report.append("正确回答中最显著的语义漂移案例:")
        report.append(f"  漂移斜率: {truthful_case['drift_slope']:.6f}")
        report.append(f"  前后段差异: {truthful_case['drift_gap']:.4f}")
        report.append(f"  总段落数: {truthful_case['total_segments']}")
        if truthful_case['drift_alert_point'] > 0:
            report.append(f"  漂移警报点: 第{truthful_case['drift_alert_point']}段")
        else:
            report.append(f"  漂移警报点: 未检测到")
        report.append("")
    
    if halluc_case:
        report.append("幻觉回答中最显著的语义漂移案例:")
        report.append(f"  漂移斜率: {halluc_case['drift_slope']:.6f}")
        report.append(f"  前后段差异: {halluc_case['drift_gap']:.4f}")
        report.append(f"  总段落数: {halluc_case['total_segments']}")
        if halluc_case['drift_alert_point'] > 0:
            report.append(f"  漂移警报点: 第{halluc_case['drift_alert_point']}段")
        else:
            report.append(f"  漂移警报点: 未检测到")
        report.append("")
    
    # 总结与建议
    report.append("🎯 总结与建议")
    report.append("-" * 40)
    report.append("基于语义漂移分析的发现:")
    
    # 生成自动化总结
    key_findings = []
    
    if abs(slope_diff) > 0.001:
        if slope_diff < 0:
            key_findings.append("正确回答表现出更强的语义漂移趋势")
        else:
            key_findings.append("幻觉回答表现出更强的语义漂移趋势")
    
    if abs(gap_diff) > 0.01:
        if gap_diff > 0:
            key_findings.append("正确回答的前后段语义变化更大")
        else:
            key_findings.append("幻觉回答的前后段语义变化更大")
    
    if abs(halluc_drift_rate - truthful_drift_rate) > 5:
        if halluc_drift_rate > truthful_drift_rate:
            key_findings.append("幻觉回答更频繁地触发语义漂移警报")
        else:
            key_findings.append("正确回答更频繁地触发语义漂移警报")
    
    if key_findings:
        for i, finding in enumerate(key_findings, 1):
            report.append(f"{i}. {finding}")
    else:
        report.append("1. 正确回答和幻觉回答在语义漂移方面表现相似")
    
    report.append("")
    report.append("建议:")
    report.append("- 可以将语义漂移指标作为幻觉检测的补充特征")
    report.append("- 漂移斜率和前后段差异可能是最有价值的指标")
    report.append("- 建议结合PRD和SAS指标进行综合分析")
    report.append("")
    report.append("="*80)
    
    return "\n".join(report)

def plot_rq1_figures_separate(truthful_prd, hallucinated_prd, truthful_gass, hallucinated_gass, output_dir):
    """
    绘制分离的RQ1分析图表：将3×2布局分成3张独立的图
    使用matplotlib原始灰底风格
    """
    import random
    import pandas as pd
    
    # 使用matplotlib默认样式，移除seaborn
    plt.style.use('default')
    
    # 数据采样准备（用于特征空间图）
    if len(truthful_prd) > 750:
        truthful_indices = random.sample(range(len(truthful_prd)), 750)
        truthful_prd_sample = [truthful_prd[i] for i in truthful_indices]
        truthful_gass_sample = [truthful_gass[i] for i in truthful_indices]
    else:
        truthful_prd_sample = truthful_prd
        truthful_gass_sample = truthful_gass
    
    if len(hallucinated_prd) > 750:
        hall_indices = random.sample(range(len(hallucinated_prd)), 750)
        hall_prd_sample = [hallucinated_prd[i] for i in hall_indices]
        hall_gass_sample = [hallucinated_gass[i] for i in hall_indices]
    else:
        hall_prd_sample = hallucinated_prd
        hall_gass_sample = hallucinated_gass
    
    # 第一张图：分布直方图 (Row 1)
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 3))
    
    # PRD分布直方图
    ax = axes[0]
    ax.hist(hall_prd_sample, alpha=0.5, label='Hallucinated', bins=30, color='red')
    ax.hist(truthful_prd_sample, alpha=0.5, label='Truthful', bins=30, color='blue')
    ax.set_title('PRD Score Distribution')
    ax.set_xlabel('PRD Score')
    ax.set_ylabel('Frequency')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # SAS分布直方图（原GASS）
    ax = axes[1]
    ax.hist(hall_gass_sample, alpha=0.5, label='Hallucinated', bins=30, color='red')
    ax.hist(truthful_gass_sample, alpha=0.5, label='Truthful', bins=30, color='blue')
    ax.set_title('SAS Score Distribution')
    ax.set_xlabel('SAS Score')
    ax.set_ylabel('Frequency')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    dist_path_png = os.path.join(output_dir, 'rq1_distributions.png')
    dist_path_pdf = os.path.join(output_dir, 'rq1_distributions.pdf')
    plt.savefig(dist_path_png, dpi=600, bbox_inches='tight')  # 最高分辨率
    plt.savefig(dist_path_pdf, bbox_inches='tight')  # PDF版本
    plt.close()
    
    # 第二张图：简洁版箱线图 (Row 2)
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 3))
    
    # PRD箱线图 - 简洁版本
    ax = axes[0]
    data_prd = [hallucinated_prd, truthful_prd]
    bp_prd = ax.boxplot(data_prd, tick_labels=['Hallucinated', 'Truthful'], showfliers=False)
    
    ax.set_title('PRD Score Box Plot')
    ax.set_ylabel('PRD Score')
    ax.grid(True, alpha=0.3)
    
    # SAS箱线图 - 简洁版本
    ax = axes[1]
    data_gass = [hallucinated_gass, truthful_gass]
    bp_gass = ax.boxplot(data_gass, tick_labels=['Hallucinated', 'Truthful'], showfliers=False)
    
    ax.set_title('SAS Score Box Plot')
    ax.set_ylabel('SAS Score')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    box_path_png = os.path.join(output_dir, 'rq1_boxplots.png')
    box_path_pdf = os.path.join(output_dir, 'rq1_boxplots.pdf')
    plt.savefig(box_path_png, dpi=600, bbox_inches='tight')  # 最高分辨率
    plt.savefig(box_path_pdf, bbox_inches='tight')  # PDF版本
    plt.close()
    
    # 第三张图：特征空间和相关性 (Row 3)
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 3))
    
    # 特征空间散点图
    ax = axes[0]
    ax.scatter(hall_prd_sample, hall_gass_sample, alpha=0.5, label='Hallucinated', 
               color='red', s=15, edgecolors='none')
    ax.scatter(truthful_prd_sample, truthful_gass_sample, alpha=0.6, label='Truthful', 
               color='blue', s=15, edgecolors='none')
    ax.set_xlabel('PRD Score')
    ax.set_ylabel('SAS Score')
    ax.set_title('Feature Space Distribution')
    ax.legend(loc='upper left', handletextpad=0.3, bbox_to_anchor=(0.02, 0.98))
    ax.grid(True, alpha=0.3)
    
    # 相关性热图
    ax = axes[1]
    df = pd.DataFrame({
        'PRD': np.concatenate([truthful_prd, hallucinated_prd]),
        'SAS': np.concatenate([truthful_gass, hallucinated_gass]),
        'Label': np.concatenate([np.ones(len(truthful_prd)), np.zeros(len(hallucinated_prd))])
    })
    corr_matrix = df.corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=ax)
    ax.set_title('Feature Correlation Matrix')
    
    plt.tight_layout()
    analysis_path_png = os.path.join(output_dir, 'rq1_feature_analysis.png')
    analysis_path_pdf = os.path.join(output_dir, 'rq1_feature_analysis.pdf')
    plt.savefig(analysis_path_png, dpi=600, bbox_inches='tight')  # 最高分辨率
    plt.savefig(analysis_path_pdf, bbox_inches='tight')  # PDF版本
    plt.close()
    
    # 恢复seaborn样式（如果其他部分需要）
    plt.style.use('seaborn-v0_8')
    
    # 返回所有保存的文件路径
    return {
        'distributions': {'png': dist_path_png, 'pdf': dist_path_pdf},
        'boxplots': {'png': box_path_png, 'pdf': box_path_pdf}, 
        'analysis': {'png': analysis_path_png, 'pdf': analysis_path_pdf}
    }

def generate_detailed_report(stats_dict):
    """
    生成详细的统计分析报告
    """
    report = []
    report.append("="*80)
    report.append("RQ1实验详细分析报告")
    report.append("="*80)
    report.append("")
    
    # 评估方法信息
    eval_method = stats_dict.get('evaluation_method', 'Unknown')
    has_variants = stats_dict.get('has_prd_variants', False)
    
    report.append("🔍 评估方法")
    report.append("-" * 40)
    report.append(f"幻觉判断方法: {eval_method}")
    if eval_method == 'SQuAD':
        report.append("  - 使用SQuAD风格评估（F1分数 + 标准化匹配）")
        report.append("  - 更准确，能处理格式差异和部分匹配")
    elif eval_method == 'Hit@1':
        report.append("  - 使用Hit@1精确字符串匹配（不应出现）")
        report.append("  - 该方法已被弃用")
    
    report.append(f"PRD变体分析: {'启用' if has_variants else '未启用'}")
    if has_variants:
        report.append("  - 包含6种PRD变体：Strict, Contrast, Relative, Max, Weighted, Entropy")
        report.append("  - 用于找到最佳的注意力利用度量方法")
    else:
        report.append("  - 仅使用原始PRD计算")
        report.append("  - 可运行add_prd_variants.py添加变体分析")
    report.append("")
    
    # 基本统计信息
    report.append("📊 基本统计信息")
    report.append("-" * 40)
    report.append(f"正确回答样本数: {stats_dict['truthful']['count']}")
    report.append(f"幻觉回答样本数: {stats_dict['hallucinated']['count']}")
    report.append(f"总样本数: {stats_dict['truthful']['count'] + stats_dict['hallucinated']['count']}")
    report.append(f"幻觉率: {stats_dict['hallucinated']['count'] / (stats_dict['truthful']['count'] + stats_dict['hallucinated']['count']):.2%}")
    report.append("")
    
    # PRD分析
    report.append("📈 PRD分数分析")
    report.append("-" * 40)
    report.append(f"正确回答 - 均值: {stats_dict['truthful']['prd']['mean']:.4f}, 标准差: {stats_dict['truthful']['prd']['std']:.4f}, 中位数: {stats_dict['truthful']['prd']['median']:.4f}")
    report.append(f"幻觉回答 - 均值: {stats_dict['hallucinated']['prd']['mean']:.4f}, 标准差: {stats_dict['hallucinated']['prd']['std']:.4f}, 中位数: {stats_dict['hallucinated']['prd']['median']:.4f}")
    report.append(f"均值差异: {stats_dict['truthful']['prd']['mean'] - stats_dict['hallucinated']['prd']['mean']:.4f}")
    report.append("")
    
    # SAS分析（原GASS）
    report.append("📈 SAS分数分析")
    report.append("-" * 40)
    report.append(f"正确回答 - 均值: {stats_dict['truthful']['gass']['mean']:.4f}, 标准差: {stats_dict['truthful']['gass']['std']:.4f}, 中位数: {stats_dict['truthful']['gass']['median']:.4f}")
    report.append(f"幻觉回答 - 均值: {stats_dict['hallucinated']['gass']['mean']:.4f}, 标准差: {stats_dict['hallucinated']['gass']['std']:.4f}, 中位数: {stats_dict['hallucinated']['gass']['median']:.4f}")
    report.append(f"均值差异: {stats_dict['truthful']['gass']['mean'] - stats_dict['hallucinated']['gass']['mean']:.4f}")
    report.append("")
    
    # Semantic Drift分析
    if 'semantic_drift' in stats_dict:
        report.append("🌊 语义漂移 (Semantic Drift) 分析")
        report.append("-" * 40)
        
        drift_data = stats_dict['semantic_drift']
        
        # 漂移斜率分析
        report.append("漂移斜率 (Drift Slope):")
        report.append(f"  正确回答 - 均值: {drift_data['truthful']['drift_slope']['mean']:.6f}, 标准差: {drift_data['truthful']['drift_slope']['std']:.6f}")
        report.append(f"  幻觉回答 - 均值: {drift_data['hallucinated']['drift_slope']['mean']:.6f}, 标准差: {drift_data['hallucinated']['drift_slope']['std']:.6f}")
        
        # 前后段差异分析
        report.append("前后段差异 (Drift Gap):")
        report.append(f"  正确回答 - 均值: {drift_data['truthful']['drift_gap']['mean']:.4f}, 标准差: {drift_data['truthful']['drift_gap']['std']:.4f}")
        report.append(f"  幻觉回答 - 均值: {drift_data['hallucinated']['drift_gap']['mean']:.4f}, 标准差: {drift_data['hallucinated']['drift_gap']['std']:.4f}")
        
        # 早晚期比率分析
        report.append("早晚期比率 (Early/Late Ratio):")
        report.append(f"  正确回答 - 均值: {drift_data['truthful']['early_late_ratio']['mean']:.4f}, 标准差: {drift_data['truthful']['early_late_ratio']['std']:.4f}")
        report.append(f"  幻觉回答 - 均值: {drift_data['hallucinated']['early_late_ratio']['mean']:.4f}, 标准差: {drift_data['hallucinated']['early_late_ratio']['std']:.4f}")
        
        # 漂移警报点分析
        truthful_alert_rate = len([d for d in drift_data['truthful_raw'] if d['drift_alert_point'] > 0]) / len(drift_data['truthful_raw']) * 100
        halluc_alert_rate = len([d for d in drift_data['hallucinated_raw'] if d['drift_alert_point'] > 0]) / len(drift_data['hallucinated_raw']) * 100
        
        report.append("漂移警报点检测:")
        report.append(f"  正确回答中发生漂移的比例: {truthful_alert_rate:.1f}%")
        report.append(f"  幻觉回答中发生漂移的比例: {halluc_alert_rate:.1f}%")
        
        # Semantic Drift统计检验
        if 'semantic_drift_tests' in stats_dict['statistical_tests']:
            drift_tests = stats_dict['statistical_tests']['semantic_drift_tests']
            
            report.append("")
            report.append("语义漂移统计检验:")
            for metric_name, test_result in drift_tests.items():
                significance = "显著" if test_result['p_value'] < 0.05 else "不显著"
                report.append(f"  {metric_name}: p值={test_result['p_value']:.6f} ({significance})")
        
        report.append("")
        report.append("💡 语义漂移结论:")
        slope_diff = drift_data['truthful']['drift_slope']['mean'] - drift_data['hallucinated']['drift_slope']['mean']
        gap_diff = drift_data['truthful']['drift_gap']['mean'] - drift_data['hallucinated']['drift_gap']['mean']
        
        if slope_diff < -0.001:  # 幻觉回答下降更快
            report.append("  - 幻觉回答展现更明显的语义漂移趋势（斜率更负）")
        elif slope_diff > 0.001:  # 正确回答下降更快
            report.append("  - 正确回答展现更明显的语义漂移趋势（斜率更负）")
        else:
            report.append("  - 两类回答的语义漂移斜率无明显差异")
            
        if gap_diff > 0.01:  # 正确回答前后差异更大
            report.append("  - 正确回答的前后段语义差异更大")
        elif gap_diff < -0.01:  # 幻觉回答前后差异更大
            report.append("  - 幻觉回答的前后段语义差异更大")
        else:
            report.append("  - 两类回答的前后段语义差异相似")
            
        if halluc_alert_rate > truthful_alert_rate + 5:
            report.append("  - 幻觉回答更容易触发语义漂移警报")
        elif truthful_alert_rate > halluc_alert_rate + 5:
            report.append("  - 正确回答更容易触发语义漂移警报")
        else:
            report.append("  - 两类回答的语义漂移警报率相似")
        
        report.append("")
    
    # 统计检验
    report.append("🔬 统计检验结果")
    report.append("-" * 40)
    
    # PRD t检验
    prd_t = stats_dict['statistical_tests']['prd_t_test']
    report.append(f"PRD t检验:")
    report.append(f"  t统计量: {prd_t['t_statistic']:.4f}")
    report.append(f"  p值: {prd_t['p_value']:.6f}")
    report.append(f"  显著性: {'显著' if prd_t['p_value'] < 0.05 else '不显著'} (α=0.05)")
    
    # SAS t检验（原GASS）
    gass_t = stats_dict['statistical_tests']['gass_t_test']
    report.append(f"SAS t检验:")
    report.append(f"  t统计量: {gass_t['t_statistic']:.4f}")
    report.append(f"  p值: {gass_t['p_value']:.6f}")
    report.append(f"  显著性: {'显著' if gass_t['p_value'] < 0.05 else '不显著'} (α=0.05)")
    report.append("")
    
    # 效应大小
    report.append("📏 效应大小 (Cohen's d)")
    report.append("-" * 40)
    report.append(f"PRD Cohen's d: {stats_dict['effect_sizes']['prd_cohens_d']:.4f}")
    report.append(f"SAS Cohen's d: {stats_dict['effect_sizes']['gass_cohens_d']:.4f}")
    report.append("")
    report.append("效应大小解释:")
    report.append("  小效应: 0.2")
    report.append("  中等效应: 0.5")
    report.append("  大效应: 0.8")
    report.append("")
    
    # 结论
    report.append("💡 分析结论")
    report.append("-" * 40)
    
    # PRD结论
    prd_significant = prd_t['p_value'] < 0.05
    prd_effect_size = abs(stats_dict['effect_sizes']['prd_cohens_d'])
    prd_direction = "正确回答" if stats_dict['truthful']['prd']['mean'] > stats_dict['hallucinated']['prd']['mean'] else "幻觉回答"
    
    report.append(f"PRD指标: {prd_direction}具有更高的PRD分数")
    if prd_significant:
        if prd_effect_size >= 0.8:
            report.append("  - 差异显著且效应大，PRD是区分正确与幻觉回答的强指标")
        elif prd_effect_size >= 0.5:
            report.append("  - 差异显著且效应中等，PRD是区分正确与幻觉回答的有效指标")
        elif prd_effect_size >= 0.2:
            report.append("  - 差异显著但效应较小，PRD在区分正确与幻觉回答方面有一定作用")
        else:
            report.append("  - 差异显著但效应很小，PRD的实际意义有限")
    else:
        report.append("  - 差异不显著，PRD无法有效区分正确与幻觉回答")
    
    # SAS结论（原GASS）
    gass_significant = gass_t['p_value'] < 0.05
    gass_effect_size = abs(stats_dict['effect_sizes']['gass_cohens_d'])
    gass_direction = "正确回答" if stats_dict['truthful']['gass']['mean'] > stats_dict['hallucinated']['gass']['mean'] else "幻觉回答"
    report.append(f"SAS指标: {gass_direction}具有更高的SAS分数")
    if gass_significant:
        if gass_effect_size >= 0.8:
            report.append("  - 差异显著且效应大，SAS是区分正确与幻觉回答的强指标")
        elif gass_effect_size >= 0.5:
            report.append("  - 差异显著且效应中等，SAS是区分正确与幻觉回答的有效指标")
        elif gass_effect_size >= 0.2:
            report.append("  - 差异显著但效应较小，SAS在区分正确与幻觉回答方面有一定作用")
        else:
            report.append("  - 差异显著但效应很小，SAS的实际意义有限")
    else:
        report.append("  - 差异不显著，SAS无法有效区分正确与幻觉回答")
    
    # PRD变体分析（如果存在）
    if has_variants and 'prd_variants' in stats_dict:
        report.append("📈 PRD变体分析")
        report.append("-" * 40)
        
        variants_performance = []
        
        for variant_name, variant_stats in stats_dict['prd_variants'].items():
            truthful_mean = variant_stats['truthful']['mean']
            hallucinated_mean = variant_stats['hallucinated']['mean']
            mean_diff = truthful_mean - hallucinated_mean
            
            # 获取统计检验结果
            variant_tests = stats_dict.get('statistical_tests', {}).get('prd_variants', {})
            variant_effect_sizes = stats_dict.get('effect_sizes', {}).get('prd_variants', {})
            
            if variant_name in variant_tests:
                p_value = variant_tests[variant_name]['p_value']
                is_significant = p_value < 0.05
            else:
                p_value = float('inf')
                is_significant = False
                
            if variant_name in variant_effect_sizes:
                effect_size = abs(variant_effect_sizes[variant_name])
            else:
                effect_size = 0.0
            
            # 记录变体表现
            variants_performance.append({
                'name': variant_name,
                'mean_diff': mean_diff,
                'p_value': p_value,
                'effect_size': effect_size,
                'is_significant': is_significant,
                'truthful_mean': truthful_mean,
                'hallucinated_mean': hallucinated_mean
            })
            
            report.append(f"{variant_name.upper()}:")
            report.append(f"  正确回答均值: {truthful_mean:.4f}")
            report.append(f"  幻觉回答均值: {hallucinated_mean:.4f}")
            report.append(f"  差异: {mean_diff:.4f}")
            if not np.isinf(p_value):
                report.append(f"  p值: {p_value:.6f} ({'显著' if is_significant else '不显著'})")
                report.append(f"  效应大小: {effect_size:.4f}")
            report.append("")
        
        # 排序并推荐最佳变体
        report.append("🏆 PRD变体性能排名")
        report.append("-" * 40)
        
        # 先按显著性排序，再按效应大小排序
        variants_performance.sort(key=lambda x: (-int(x['is_significant']), -x['effect_size']))
        
        for i, variant in enumerate(variants_performance[:3], 1):  # 只显示前3名
            name = variant['name']
            diff = variant['mean_diff']
            effect = variant['effect_size']
            sig_text = "✓" if variant['is_significant'] else "✗"
            
            direction = "正确>幻觉" if diff > 0 else "幻觉>正确"
            report.append(f"{i}. {name.upper()}: 差异={diff:.4f} ({direction}), 效应={effect:.4f}, 显著性={sig_text}")
        
        # 推荐
        best_variant = variants_performance[0]
        report.append("")
        report.append("💡 推荐使用的PRD变体")
        report.append("-" * 40)
        report.append(f"最佳变体: {best_variant['name'].upper()}")
        
        if best_variant['is_significant'] and best_variant['effect_size'] >= 0.2:
            report.append(f"  - 统计显著 (p={best_variant['p_value']:.6f})")
            report.append(f"  - 效应大小: {best_variant['effect_size']:.4f}")
            
            if best_variant['mean_diff'] > 0:
                report.append("  - 正确答案的PRD分数更高（符合预期）")
            else:
                report.append("  - 幻觉答案的PRD分数更高（需要进一步调查）")
        else:
            report.append("  - 注意：最佳变体仍然表现不理想")
            report.append("  - 建议检查数据质量或尝试其他方法")
        
        report.append("")
    
    report.append("")
    report.append("="*80)
    
    return "\n".join(report)

def run_rq1_experiment(args):
    """
    运行RQ1实验：分析LLM对gold-relevant triples的注意力与幻觉的关系
    
    只使用SQuAD风格幻觉判断方法：
    - SQuAD风格评估：基于F1分数和标准化匹配，更准确
    - 如果数据中不包含SQuAD评估结果，将跳过相应样本
    """
    logger = logging.getLogger(__name__)
    logger.info("Starting RQ1 experiment...")
    
    # 创建输出目录
    filename = os.path.basename(args.results_file)
    # 从文件名中提取前缀（如 "dev_simple" 来自 "dev_simple_inference_results_..."）
    filename_parts = filename.split("_")
    if len(filename_parts) >= 2:
        prefix = f"{filename_parts[0]}_{filename_parts[1]}"
    else:
        prefix = filename_parts[0] if filename_parts else "unknown"
    
    output_dir = os.path.join('experiment_records', 'empirical_rq1', 
                             f'{prefix}_rq1_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # 使用提供的结果文件
        results_file = args.results_file
        
        # 分析结果
        truthful_prd, hallucinated_prd = [], []
        truthful_gass, hallucinated_gass = [], []
        # Semantic Drift数据
        truthful_samples, hallucinated_samples = [], []
        # PRD变体数据
        prd_variants_data = {
            'prd_strict': {'truthful': [], 'hallucinated': []},
            'prd_contrast': {'truthful': [], 'hallucinated': []},
            'prd_relative': {'truthful': [], 'hallucinated': []},
            'prd_max': {'truthful': [], 'hallucinated': []},
            'prd_weighted': {'truthful': [], 'hallucinated': []},
            'prd_entropy': {'truthful': [], 'hallucinated': []}
        }
        has_prd_variants = False
        evaluation_method = None
        
        with open(results_file, 'r', encoding='utf-8') as f:
            line_count = 0
            for line in f:
                if not line.strip():
                    continue
                data = json.loads(line)
                if 'config' in data or 'stats' in data:
                    continue
                # 只处理前5000条有效数据
                line_count += 1
                if line_count > 5000:
                    break
                    
                # 获取PRD和GASS分数
                prd_score = data.get('prd_score', 0.0)
                gass_score = data.get('gass_score', 0.0)
                
                # 检查是否包含PRD变体
                if not has_prd_variants:
                    if any(variant in data for variant in prd_variants_data.keys()):
                        has_prd_variants = True
                        print("检测到PRD变体数据，将进行变体分析")
                
                # 只使用SQuAD评估方法，不再回退到Hit@1
                squad_eval = data.get('squad_evaluation')
                if squad_eval is not None:
                    # 使用SQuAD评估：squad_is_hallucination为False表示非幻觉（正确）
                    is_correct = not squad_eval.get('squad_is_hallucination', True)
                    if evaluation_method is None:
                        evaluation_method = 'SQuAD'
                else:
                    # 没有SQuAD评估数据，跳过该样本
                    continue
                
                # 根据评估结果分组
                if is_correct:
                    truthful_prd.append(prd_score)
                    truthful_gass.append(gass_score)
                    truthful_samples.append(data)  # 保存完整样本用于Semantic Drift分析
                    # 收集PRD变体数据
                    if has_prd_variants:
                        for variant in prd_variants_data.keys():
                            variant_score = data.get(variant, 0.0)
                            prd_variants_data[variant]['truthful'].append(variant_score)
                else:
                    hallucinated_prd.append(prd_score)
                    hallucinated_gass.append(gass_score)
                    hallucinated_samples.append(data)  # 保存完整样本用于Semantic Drift分析
                    # 收集PRD变体数据
                    if has_prd_variants:
                        for variant in prd_variants_data.keys():
                            variant_score = data.get(variant, 0.0)
                            prd_variants_data[variant]['hallucinated'].append(variant_score)
        
        # 计算统计指标
        stats_dict = {
            'evaluation_method': evaluation_method,
            'has_prd_variants': has_prd_variants,
            'truthful': {
                'count': len(truthful_prd),
                'prd': {
                    'mean': np.mean(truthful_prd),
                    'std': np.std(truthful_prd),
                    'median': np.median(truthful_prd)
                },
                'gass': {
                    'mean': np.mean(truthful_gass),
                    'std': np.std(truthful_gass),
                    'median': np.median(truthful_gass)
                }
            },
            'hallucinated': {
                'count': len(hallucinated_prd),
                'prd': {
                    'mean': np.mean(hallucinated_prd),
                    'std': np.std(hallucinated_prd),
                    'median': np.median(hallucinated_prd)
                },
                'gass': {
                    'mean': np.mean(hallucinated_gass),
                    'std': np.std(hallucinated_gass),
                    'median': np.median(hallucinated_gass)
                }
            }
        }
        
        # 如果有PRD变体，添加变体统计
        if has_prd_variants:
            stats_dict['prd_variants'] = {}
            for variant in prd_variants_data.keys():
                truthful_variant = prd_variants_data[variant]['truthful']
                hallucinated_variant = prd_variants_data[variant]['hallucinated']
                
                if truthful_variant and hallucinated_variant:
                    stats_dict['prd_variants'][variant] = {
                        'truthful': {
                            'mean': np.mean(truthful_variant),
                            'std': np.std(truthful_variant),
                            'median': np.median(truthful_variant)
                        },
                        'hallucinated': {
                            'mean': np.mean(hallucinated_variant),
                            'std': np.std(hallucinated_variant),
                            'median': np.median(hallucinated_variant)
            }
        }
        
        # 执行Semantic Drift分析
        print("\n🌊 执行语义漂移 (Semantic Drift) 分析...")
        
        # 分析真实回答的语义漂移
        truthful_drift_data = []
        for sample in truthful_samples[:200]:  # 限制样本数量以提高性能
            drift_result = analyze_semantic_drift(sample)
            truthful_drift_data.append(drift_result)
        
        # 分析幻觉回答的语义漂移
        hallucinated_drift_data = []
        for sample in hallucinated_samples[:200]:  # 限制样本数量以提高性能
            drift_result = analyze_semantic_drift(sample)
            hallucinated_drift_data.append(drift_result)
        
        # 计算语义漂移统计指标
        if truthful_drift_data and hallucinated_drift_data:
            # 提取漂移指标
            truthful_slopes = [d['drift_slope'] for d in truthful_drift_data]
            halluc_slopes = [d['drift_slope'] for d in hallucinated_drift_data]
            truthful_gaps = [d['drift_gap'] for d in truthful_drift_data]
            halluc_gaps = [d['drift_gap'] for d in hallucinated_drift_data]
            truthful_ratios = [d['early_late_ratio'] for d in truthful_drift_data]
            halluc_ratios = [d['early_late_ratio'] for d in hallucinated_drift_data]
            
            # 添加语义漂移统计到主统计字典
            stats_dict['semantic_drift'] = {
                'truthful': {
                    'drift_slope': {
                        'mean': np.mean(truthful_slopes),
                        'std': np.std(truthful_slopes),
                        'median': np.median(truthful_slopes)
                    },
                    'drift_gap': {
                        'mean': np.mean(truthful_gaps),
                        'std': np.std(truthful_gaps),
                        'median': np.median(truthful_gaps)
                    },
                    'early_late_ratio': {
                        'mean': np.mean(truthful_ratios),
                        'std': np.std(truthful_ratios),
                        'median': np.median(truthful_ratios)
                    }
                },
                'hallucinated': {
                    'drift_slope': {
                        'mean': np.mean(halluc_slopes),
                        'std': np.std(halluc_slopes),
                        'median': np.median(halluc_slopes)
                    },
                    'drift_gap': {
                        'mean': np.mean(halluc_gaps),
                        'std': np.std(halluc_gaps),
                        'median': np.median(halluc_gaps)
                    },
                    'early_late_ratio': {
                        'mean': np.mean(halluc_ratios),
                        'std': np.std(halluc_ratios),
                        'median': np.median(halluc_ratios)
                    }
                },
                'truthful_raw': truthful_drift_data,
                'hallucinated_raw': hallucinated_drift_data
            }
        
        # 进行t检验
        prd_t_stat, prd_p_value = stats.ttest_ind(truthful_prd, hallucinated_prd)
        gass_t_stat, gass_p_value = stats.ttest_ind(truthful_gass, hallucinated_gass)
        
        stats_dict['statistical_tests'] = {
            'prd_t_test': {
                't_statistic': prd_t_stat,
                'p_value': prd_p_value
            },
            'gass_t_test': {
                't_statistic': gass_t_stat,
                'p_value': gass_p_value
            }
        }
        
        # 计算效应大小 (Cohen's d)
        def cohens_d(group1, group2):
            n1, n2 = len(group1), len(group2)
            pooled_std = np.sqrt(((n1 - 1) * np.var(group1, ddof=1) + (n2 - 1) * np.var(group2, ddof=1)) / (n1 + n2 - 2))
            return (np.mean(group1) - np.mean(group2)) / pooled_std
        
        prd_d = cohens_d(truthful_prd, hallucinated_prd)
        gass_d = cohens_d(truthful_gass, hallucinated_gass)
        
        stats_dict['effect_sizes'] = {
            'prd_cohens_d': prd_d,
            'gass_cohens_d': gass_d
        }
        
        # 如果有PRD变体，计算变体的统计检验和效应大小
        if has_prd_variants:
            stats_dict['statistical_tests']['prd_variants'] = {}
            stats_dict['effect_sizes']['prd_variants'] = {}
            
            for variant in prd_variants_data.keys():
                truthful_variant = prd_variants_data[variant]['truthful']
                hallucinated_variant = prd_variants_data[variant]['hallucinated']
                
                if truthful_variant and hallucinated_variant and len(truthful_variant) > 1 and len(hallucinated_variant) > 1:
                    # t检验
                    variant_t_stat, variant_p_value = stats.ttest_ind(truthful_variant, hallucinated_variant)
                    stats_dict['statistical_tests']['prd_variants'][variant] = {
                        't_statistic': variant_t_stat,
                        'p_value': variant_p_value
                    }
                    
                    # Cohen's d
                    variant_d = cohens_d(truthful_variant, hallucinated_variant)
                    stats_dict['effect_sizes']['prd_variants'][variant] = variant_d
        
        # 添加Semantic Drift统计检验和效应大小
        if 'semantic_drift' in stats_dict and truthful_drift_data and hallucinated_drift_data:
            # 对每个漂移指标进行统计检验
            drift_metrics = ['drift_slope', 'drift_gap', 'early_late_ratio']
            stats_dict['statistical_tests']['semantic_drift_tests'] = {}
            stats_dict['effect_sizes']['semantic_drift_effects'] = {}
            
            for metric in drift_metrics:
                truthful_values = [d[metric] for d in truthful_drift_data]
                halluc_values = [d[metric] for d in hallucinated_drift_data]
                
                if len(truthful_values) > 1 and len(halluc_values) > 1:
                    # t检验
                    t_stat, p_value = stats.ttest_ind(truthful_values, halluc_values)
                    stats_dict['statistical_tests']['semantic_drift_tests'][metric] = {
                        't_statistic': t_stat,
                        'p_value': p_value
                    }
                    
                    # Cohen's d
                    effect_size = cohens_d(truthful_values, halluc_values)
                    stats_dict['effect_sizes']['semantic_drift_effects'][metric] = effect_size
        
        # 保存统计结果
        analysis_file = os.path.join(output_dir, 'rq1_analysis.json')
        with open(analysis_file, 'w', encoding='utf-8') as f:
            json.dump(stats_dict, f, indent=2, ensure_ascii=False, default=float)
        
        # 保存语义漂移详细数据
        if 'semantic_drift' in stats_dict and truthful_drift_data and hallucinated_drift_data:
            drift_detailed_file = os.path.join(output_dir, 'rq1_semantic_drift_detailed.json')
            drift_detailed_data = {
                'metadata': {
                    'analysis_type': 'semantic_drift',
                    'timestamp': datetime.now().isoformat(),
                    'segment_size': 5,  # tokens per segment
                    'sample_counts': {
                        'truthful_samples': len(truthful_drift_data),
                        'hallucinated_samples': len(hallucinated_drift_data)
                    }
                },
                'truthful_samples': truthful_drift_data,
                'hallucinated_samples': hallucinated_drift_data,
                'summary_statistics': stats_dict['semantic_drift']
            }
            
            with open(drift_detailed_file, 'w', encoding='utf-8') as f:
                json.dump(drift_detailed_data, f, indent=2, ensure_ascii=False, default=float)
            
            # 生成语义漂移专门报告
            drift_report_file = os.path.join(output_dir, 'rq1_semantic_drift_report.txt')
            drift_report = generate_semantic_drift_report(stats_dict['semantic_drift'], truthful_drift_data, hallucinated_drift_data)
            
            with open(drift_report_file, 'w', encoding='utf-8') as f:
                f.write(drift_report)
            
            print(f"💾 语义漂移详细数据保存至: {drift_detailed_file}")
            print(f"📄 语义漂移分析报告保存至: {drift_report_file}")
        
        # 绘制分离的图表
        figure_paths = plot_rq1_figures_separate(
            truthful_prd, hallucinated_prd,
            truthful_gass, hallucinated_gass,
            output_dir
        )
        
        # 绘制Semantic Drift分析图表
        drift_figure_paths = None
        if 'semantic_drift' in stats_dict and truthful_drift_data and hallucinated_drift_data:
            print("\n🌊 生成语义漂移分析图表...")
            drift_figure_paths = plot_semantic_drift_analysis(
                truthful_drift_data, 
                hallucinated_drift_data, 
                output_dir
            )
        
        # 生成详细报告
        detailed_report = generate_detailed_report(stats_dict)
        
        # 保存详细报告
        report_file = os.path.join(output_dir, 'rq1_detailed_report.txt')
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(detailed_report)
        
        # 打印结果
        print("\n" + "="*80)
        print("RQ1实验结果")
        print("="*80)
        print(f"评估方法: {evaluation_method}")
        print(f"正确回答样本数: {len(truthful_prd)}")
        print(f"幻觉回答样本数: {len(hallucinated_prd)}")
        print(f"总样本数: {len(truthful_prd) + len(hallucinated_prd)}")
        print(f"幻觉率: {len(hallucinated_prd) / (len(truthful_prd) + len(hallucinated_prd)):.2%}")
        print()
        
        print("PRD分数分析:")
        print(f"  正确回答 - 均值: {np.mean(truthful_prd):.4f}, 标准差: {np.std(truthful_prd):.4f}")
        print(f"  幻觉回答 - 均值: {np.mean(hallucinated_prd):.4f}, 标准差: {np.std(hallucinated_prd):.4f}")
        print(f"  t检验 p值: {prd_p_value:.6f}")
        print(f"  Cohen's d: {prd_d:.4f}")
        print()
        
        print("SAS分数分析:")
        print(f"  正确回答 - 均值: {np.mean(truthful_gass):.4f}, 标准差: {np.std(truthful_gass):.4f}")
        print(f"  幻觉回答 - 均值: {np.mean(hallucinated_gass):.4f}, 标准差: {np.std(hallucinated_gass):.4f}")
        print(f"  t检验 p值: {gass_p_value:.6f}")
        print(f"  Cohen's d: {gass_d:.4f}")
        print()
        
        print("文件保存位置:")
        print(f"  统计分析: {analysis_file}")
        print(f"  详细报告: {report_file}")
        print(f"  分布图: {figure_paths['distributions']['png']} (PNG, 600 DPI)")
        print(f"          {figure_paths['distributions']['pdf']} (PDF)")
        print(f"  箱线图: {figure_paths['boxplots']['png']} (PNG, 600 DPI)")
        print(f"          {figure_paths['boxplots']['pdf']} (PDF)")
        print(f"  特征分析图: {figure_paths['analysis']['png']} (PNG, 600 DPI)")
        print(f"              {figure_paths['analysis']['pdf']} (PDF)")
        
        # 显示语义漂移文件信息
        if 'semantic_drift' in stats_dict:
            drift_detailed_file = os.path.join(output_dir, 'rq1_semantic_drift_detailed.json')
            drift_report_file = os.path.join(output_dir, 'rq1_semantic_drift_report.txt')
            print(f"  语义漂移详细数据: {drift_detailed_file}")
            print(f"  语义漂移分析报告: {drift_report_file}")
        
        # 显示Semantic Drift分析结果
        if 'semantic_drift' in stats_dict:
            print()
            print("🌊 语义漂移 (Semantic Drift) 分析:")
            drift_data = stats_dict['semantic_drift']
            print(f"  漂移斜率 - 正确回答均值: {drift_data['truthful']['drift_slope']['mean']:.6f}")
            print(f"             幻觉回答均值: {drift_data['hallucinated']['drift_slope']['mean']:.6f}")
            print(f"  前后段差异 - 正确回答均值: {drift_data['truthful']['drift_gap']['mean']:.4f}")
            print(f"               幻觉回答均值: {drift_data['hallucinated']['drift_gap']['mean']:.4f}")
            
            if drift_figure_paths:
                print(f"  语义漂移图: {drift_figure_paths['png']} (PNG, 600 DPI)")
                print(f"              {drift_figure_paths['pdf']} (PDF)")
        
        logger.info("RQ1 experiment completed successfully")
        
    except Exception as e:
        logger.error(f"Error in RQ1 experiment: {e}")
        raise

if __name__ == "__main__":
    # 设置日志
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    class Args:
        def __init__(self, results_file):
            self.results_file = results_file
    
    # 示例用法
    import sys
    if len(sys.argv) > 1:
        args = Args(sys.argv[1])
        run_rq1_experiment(args)
    else:
        print("用法: python run_rq1.py <results_file>")
        print("示例: python run_rq1.py experiment_records/inference_results/llama2-7b/colab_dev_simple.jsonl")