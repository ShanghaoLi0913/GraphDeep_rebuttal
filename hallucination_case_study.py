"""
幻觉类型Case Study分析脚本

基于PRD×SAS四象限分类深入分析不同类型的幻觉产生机制：

四象限分类：
- Q1 (High PRD, High SAS): 短路径过拟合 - 过度依赖最短路径但语义对齐较好
- Q2 (Low PRD, High SAS): 理想情况 - 低路径依赖但高语义对齐
- Q3 (Low PRD, Low SAS): 语义脱节 - 无重点且语义脱节的幻觉
- Q4 (High PRD, Low SAS): 路径误导 - 依赖路径但语义错误的幻觉

使用方法:
python hallucination_case_study.py --input_file results.jsonl --output_dir case_study_results
"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple
import pandas as pd
from datetime import datetime
import argparse
from collections import defaultdict

class HallucinationCaseStudy:
    def __init__(self, input_file: str, output_dir: str):
        self.input_file = input_file
        self.output_dir = output_dir
        self.samples = []
        self.quadrant_samples = {
            'Q1_high_prd_high_sas': [],
            'Q2_low_prd_high_sas': [],
            'Q3_low_prd_low_sas': [],
            'Q4_high_prd_low_sas': []
        }
        
        os.makedirs(output_dir, exist_ok=True)
        
    def load_data(self, max_samples=5000):
        """加载推理结果数据"""
        print(f"📖 Loading inference results (max {max_samples} samples)...")
        
        sample_count = 0
        with open(self.input_file, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                    
                try:
                    data = json.loads(line.strip())
                    
                    # 跳过配置行
                    if 'config' in data:
                        continue
                    
                    # 确保有必要的字段
                    if all(key in data for key in ['tus_score', 'prd_score', 'gass_score', 'metrics']):
                        self.samples.append(data)
                        sample_count += 1
                        
                        # 达到最大样本数时停止
                        if sample_count >= max_samples:
                            break
                        
                except json.JSONDecodeError:
                    continue
        
        print(f"✅ Loaded {len(self.samples)} valid samples")
        
    def calculate_thresholds(self) -> Tuple[float, float]:
        """计算PRD和SAS的分类阈值（使用中位数）"""
        prd_scores = [s.get('prd_score', 0) for s in self.samples]
        sas_scores = [s.get('gass_score', 0) for s in self.samples]  # 使用GASS作为SAS的代理指标
        
        prd_threshold = np.median(prd_scores)
        sas_threshold = np.median(sas_scores)
        
        print(f"📊 Classification thresholds:")
        print(f"   PRD threshold (median): {prd_threshold:.4f}")
        print(f"   SAS threshold (median): {sas_threshold:.4f}")
        
        return prd_threshold, sas_threshold
    
    def classify_samples(self):
        """根据PRD×SAS四象限对样本进行分类"""
        print("\n🔍 Classifying samples into quadrants...")
        
        prd_threshold, sas_threshold = self.calculate_thresholds()
        
        # 重置分类
        for key in self.quadrant_samples:
            self.quadrant_samples[key] = []
        
        for sample in self.samples:
            prd_score = sample.get('prd_score', 0)
            sas_score = sample.get('gass_score', 0)
            hit_at_1 = sample.get('metrics', {}).get('hit@1', False)
            
            # 添加分类信息到样本
            sample['prd_category'] = 'High' if prd_score >= prd_threshold else 'Low'
            sample['sas_category'] = 'High' if sas_score >= sas_threshold else 'Low'
            sample['quadrant'] = f"{sample['prd_category']} PRD, {sample['sas_category']} SAS"
            
            # 四象限分类
            if prd_score >= prd_threshold and sas_score >= sas_threshold:
                self.quadrant_samples['Q1_high_prd_high_sas'].append(sample)
            elif prd_score < prd_threshold and sas_score >= sas_threshold:
                self.quadrant_samples['Q2_low_prd_high_sas'].append(sample)
            elif prd_score < prd_threshold and sas_score < sas_threshold:
                self.quadrant_samples['Q3_low_prd_low_sas'].append(sample)
            else:  # prd_score >= prd_threshold and sas_score < sas_threshold
                self.quadrant_samples['Q4_high_prd_low_sas'].append(sample)
        
        # 打印分类统计
        print("\n📈 Quadrant distribution:")
        for quad_name, samples in self.quadrant_samples.items():
            correct_count = sum(1 for s in samples if s.get('metrics', {}).get('hit@1', False))
            halluc_count = len(samples) - correct_count
            print(f"   {quad_name}: {len(samples)} samples ({correct_count} correct, {halluc_count} hallucinated)")
    
    def analyze_quadrant_characteristics(self):
        """分析每个象限的特征"""
        print("\n🔬 Analyzing quadrant characteristics...")
        
        analysis = {}
        
        for quad_name, samples in self.quadrant_samples.items():
            if not samples:
                continue
                
            # 计算统计指标
            prd_scores = [s.get('prd_score', 0) for s in samples]
            sas_scores = [s.get('gass_score', 0) for s in samples]
            tus_scores = [s.get('tus_score', 0) for s in samples]
            hit_rates = [s.get('metrics', {}).get('hit@1', False) for s in samples]
            
            analysis[quad_name] = {
                'count': len(samples),
                'hit_rate': np.mean(hit_rates) * 100,
                'hallucination_rate': (1 - np.mean(hit_rates)) * 100,
                'avg_prd': np.mean(prd_scores),
                'avg_sas': np.mean(sas_scores),
                'avg_tus': np.mean(tus_scores),
                'std_prd': np.std(prd_scores),
                'std_sas': np.std(sas_scores),
                'std_tus': np.std(tus_scores)
            }
        
        return analysis
    
    def select_representative_cases(self, num_cases: int = 3) -> Dict:
        """为每个象限选择代表性案例"""
        print(f"\n🎯 Selecting {num_cases} representative cases per quadrant...")
        
        representative_cases = {}
        
        quadrant_descriptions = {
            'Q1_high_prd_high_sas': "短路径过拟合 (High PRD, High SAS)",
            'Q2_low_prd_high_sas': "理想情况 (Low PRD, High SAS)", 
            'Q3_low_prd_low_sas': "语义脱节 (Low PRD, Low SAS)",
            'Q4_high_prd_low_sas': "路径误导 (High PRD, Low SAS)"
        }
        
        for quad_name, samples in self.quadrant_samples.items():
            if not samples:
                representative_cases[quad_name] = []
                continue
            
            # 选择策略：优先选择幻觉样本，然后按分数排序
            hallucinated_samples = [s for s in samples if not s.get('metrics', {}).get('hit@1', False)]
            correct_samples = [s for s in samples if s.get('metrics', {}).get('hit@1', False)]
            
            selected_cases = []
            
            # 优先从幻觉样本中选择
            if hallucinated_samples:
                # 根据象限特点排序选择
                if 'high_prd' in quad_name:
                    # 高PRD象限：按PRD分数降序排列
                    hallucinated_samples.sort(key=lambda x: x.get('prd_score', 0), reverse=True)
                else:
                    # 低PRD象限：按SAS分数排序
                    hallucinated_samples.sort(key=lambda x: x.get('gass_score', 0), reverse=True)
                
                selected_cases.extend(hallucinated_samples[:min(num_cases, len(hallucinated_samples))])
            
            # 如果幻觉样本不够，从正确样本中补充
            remaining_slots = num_cases - len(selected_cases)
            if remaining_slots > 0 and correct_samples:
                if 'high_prd' in quad_name:
                    correct_samples.sort(key=lambda x: x.get('prd_score', 0), reverse=True)
                else:
                    correct_samples.sort(key=lambda x: x.get('gass_score', 0), reverse=True)
                
                selected_cases.extend(correct_samples[:remaining_slots])
            
            representative_cases[quad_name] = {
                'description': quadrant_descriptions[quad_name],
                'cases': selected_cases[:num_cases]
            }
            
            print(f"   {quad_name}: Selected {len(selected_cases)} cases")
        
        return representative_cases
    
    def create_visualizations(self, analysis: Dict):
        """创建可视化图表"""
        print("\n📊 Creating visualizations...")
        
        # 1. 四象限散点图 - 使用子样本以避免过度密集
        # 设置画布大小为与论文栏宽一致（单位是英寸）
        # 1栏约 3.3 in，2栏约 6.9 in。KDD 是 double column，用 3.3in（1栏）或 6.9in（双栏）
        plt.figure(figsize=(5, 4.5))  # 加宽并稍加高画布
        
        colors = ['red', 'green', 'blue', 'orange']
        quad_names = ['Q1_high_prd_high_sas', 'Q2_low_prd_high_sas', 'Q3_low_prd_low_sas', 'Q4_high_prd_low_sas']
        quad_labels = ['Q1: High PRD, High SAS', 'Q2: Low PRD, High SAS', 'Q3: Low PRD, Low SAS', 'Q4: High PRD, Low SAS']
        
        # 先收集所有数据，然后进行采样以保持幻觉率
        all_correct_data = []
        all_halluc_data = []
        
        import random
        random.seed(42)  # 确保结果可重现
        max_samples_per_quad = 150  # 每个象限最多150个样本
        
        for i, (quad_name, color, label) in enumerate(zip(quad_names, colors, quad_labels)):
            samples = self.quadrant_samples[quad_name]
            if samples:
                prd_scores = [s.get('prd_score', 0) for s in samples]
                sas_scores = [s.get('gass_score', 0) for s in samples]
                
                # 区分正确和错误样本
                correct_samples = [(prd, sas) for s, prd, sas in zip(samples, prd_scores, sas_scores) 
                                 if s.get('metrics', {}).get('hit@1', False)]
                halluc_samples = [(prd, sas) for s, prd, sas in zip(samples, prd_scores, sas_scores) 
                                if not s.get('metrics', {}).get('hit@1', False)]
                
                # 计算原始幻觉率
                total_samples = len(correct_samples) + len(halluc_samples)
                if total_samples > 0:
                    original_halluc_rate = len(halluc_samples) / total_samples
                    
                    # 如果样本总数超过限制，进行采样
                    if total_samples > max_samples_per_quad:
                        # 按比例分配采样数量
                        target_halluc_count = int(max_samples_per_quad * original_halluc_rate)
                        target_correct_count = max_samples_per_quad - target_halluc_count
                        
                        # 采样（保持比例）
                        sampled_correct = random.sample(correct_samples, min(target_correct_count, len(correct_samples)))
                        sampled_halluc = random.sample(halluc_samples, min(target_halluc_count, len(halluc_samples)))
                        
                        correct_samples = sampled_correct
                        halluc_samples = sampled_halluc
                
                # 收集采样后的数据
                if correct_samples:
                    correct_prd, correct_sas = zip(*correct_samples)
                    all_correct_data.append((list(correct_prd), list(correct_sas), color, label))
                if halluc_samples:
                    halluc_prd, halluc_sas = zip(*halluc_samples)
                    all_halluc_data.append((list(halluc_prd), list(halluc_sas), color, label))
                
                print(f"   {quad_name}: Sampled {len(correct_samples)} correct + {len(halluc_samples)} hallucinated (orig: {len(samples)} total)")
        
        # 先绘制所有正确样本（底层）
        for correct_prd, correct_sas, color, label in all_correct_data:
            plt.scatter(correct_prd, correct_sas, c=color, alpha=0.6, s=50, marker='o', 
                       label=f'{label} (Truthful)', zorder=1)
        
        # 再绘制所有幻觉样本（顶层）- 先绘制白色描边，再绘制彩色×
        for halluc_prd, halluc_sas, color, label in all_halluc_data:
            # 先绘制白色描边（更大的×）
            plt.scatter(halluc_prd, halluc_sas, c='white', s=150, marker='x', 
                       linewidths=3.5, zorder=2, alpha=1.0)
            # 再绘制彩色×（稍小一点）
            plt.scatter(halluc_prd, halluc_sas, c=color, alpha=0.9, s=120, marker='x', 
                       linewidths=1.5, label=f'{label} (Hallucinated)', zorder=3)
        
        # 添加阈值线
        prd_threshold, sas_threshold = self.calculate_thresholds()
        plt.axvline(x=prd_threshold, color='gray', linestyle='--', alpha=0.7, label=f'PRD threshold ({prd_threshold:.3f})')
        plt.axhline(y=sas_threshold, color='gray', linestyle='--', alpha=0.7, label=f'SAS threshold ({sas_threshold:.3f})')
        
        plt.xlabel('PRD Score (Path Reliance Degree)', fontsize=14)
        plt.ylabel('SAS Score (Semantic Alignment Score)', fontsize=14)
        plt.title('Hallucination Analysis:\nPRD × SAS Quadrant Classification', fontsize=14, ha='center')
        plt.tick_params(axis='both', which='major', labelsize=14)
        plt.grid(True, alpha=0.3)
        
        # ✅ legend 放在图外，并设置多列 & 字体大小
        plt.legend(
            loc='upper center',
            bbox_to_anchor=(0.5, -0.15),  # 缩小间距
            ncol=2,                       # 控制列数，如需要可用 3
            fontsize=13,                  # 统一字体大小为13
            frameon=False
        )
        plt.savefig(os.path.join(self.output_dir, 'quadrant_scatter_plot.png'), dpi=600, bbox_inches='tight')
        plt.savefig(os.path.join(self.output_dir, 'quadrant_scatter_plot.pdf'), dpi=600, bbox_inches='tight')
        plt.close()
        
        # 2. 象限统计柱状图
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        quad_names_clean = ['Q1\n(High PRD,\nHigh SAS)', 'Q2\n(Low PRD,\nHigh SAS)', 
                           'Q3\n(Low PRD,\nLow SAS)', 'Q4\n(High PRD,\nLow SAS)']
        
        # 样本数量
        counts = [analysis.get(quad, {}).get('count', 0) for quad in quad_names]
        ax1.bar(quad_names_clean, counts, color=colors)
        ax1.set_title('Sample Count by Quadrant', fontsize=12)
        ax1.set_ylabel('Number of Samples', fontsize=12)
        ax1.tick_params(axis='both', which='major', labelsize=12)
        
        # 幻觉率
        halluc_rates = [analysis.get(quad, {}).get('hallucination_rate', 0) for quad in quad_names]
        ax2.bar(quad_names_clean, halluc_rates, color=colors)
        ax2.set_title('Hallucination Rate by Quadrant', fontsize=12)
        ax2.set_ylabel('Hallucination Rate (%)', fontsize=12)
        ax2.tick_params(axis='both', which='major', labelsize=12)
        
        # 平均PRD分数
        avg_prds = [analysis.get(quad, {}).get('avg_prd', 0) for quad in quad_names]
        ax3.bar(quad_names_clean, avg_prds, color=colors)
        ax3.set_title('Average PRD Score by Quadrant', fontsize=12)
        ax3.set_ylabel('Average PRD Score', fontsize=12)
        ax3.tick_params(axis='both', which='major', labelsize=12)
        
        # 平均SAS分数
        avg_sass = [analysis.get(quad, {}).get('avg_sas', 0) for quad in quad_names]
        ax4.bar(quad_names_clean, avg_sass, color=colors)
        ax4.set_title('Average SAS Score by Quadrant', fontsize=12)
        ax4.set_ylabel('Average SAS Score', fontsize=12)
        ax4.tick_params(axis='both', which='major', labelsize=12)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'quadrant_statistics.png'), dpi=600, bbox_inches='tight')
        plt.savefig(os.path.join(self.output_dir, 'quadrant_statistics.pdf'), dpi=600, bbox_inches='tight')
        plt.close()
        
        print(f"   📈 Saved visualizations to {self.output_dir}")
    
    def generate_detailed_report(self, analysis: Dict, representative_cases: Dict):
        """生成详细的案例研究报告"""
        print("\n📝 Generating detailed case study report...")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = os.path.join(self.output_dir, f'hallucination_case_study_report_{timestamp}.md')
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("# Hallucination Case Study Analysis Report\n\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Input file: {self.input_file}\n")
            f.write(f"Total samples analyzed: {len(self.samples)}\n\n")
            
            # Executive Summary
            f.write("## Executive Summary\n\n")
            f.write("This report analyzes different types of hallucinations based on the PRD×SAS quadrant classification:\n\n")
            f.write("- **Q1 (High PRD, High SAS)**: 短路径过拟合 - Models over-rely on shortest paths but maintain semantic alignment\n")
            f.write("- **Q2 (Low PRD, High SAS)**: 理想情况 - Low path dependence with high semantic alignment (ideal behavior)\n")
            f.write("- **Q3 (Low PRD, Low SAS)**: 语义脱节 - Semantic misalignment with unfocused reasoning\n")
            f.write("- **Q4 (High PRD, Low SAS)**: 路径误导 - High path dependence but semantic errors\n\n")
            
            # Quadrant Analysis
            f.write("## Quadrant Analysis\n\n")
            f.write("| Quadrant | Count | Hit Rate (%) | Hallucination Rate (%) | Avg PRD | Avg SAS | Avg TUS |\n")
            f.write("|----------|-------|--------------|------------------------|---------|---------|----------|\n")
            
            for quad_name in ['Q1_high_prd_high_sas', 'Q2_low_prd_high_sas', 'Q3_low_prd_low_sas', 'Q4_high_prd_low_sas']:
                if quad_name in analysis:
                    data = analysis[quad_name]
                    f.write(f"| {quad_name.replace('_', ' ').title()} | {data['count']} | {data['hit_rate']:.1f} | {data['hallucination_rate']:.1f} | {data['avg_prd']:.4f} | {data['avg_sas']:.4f} | {data['avg_tus']:.4f} |\n")
            
            f.write("\n")
            
            # Representative Cases
            f.write("## Representative Cases\n\n")
            
            for quad_name, quad_data in representative_cases.items():
                f.write(f"### {quad_data['description']}\n\n")
                
                for i, case in enumerate(quad_data['cases'], 1):
                    f.write(f"#### Case {i}\n")
                    f.write(f"- **Question**: {case.get('question', 'N/A')}\n")
                    f.write(f"- **Model Answer**: {case.get('answer', case.get('predicted_answer', 'N/A'))}\n")
                    f.write(f"- **Gold Answer**: {', '.join(case.get('golden_answers', case.get('golden_texts', ['N/A'])))}\n")
                    f.write(f"- **Correct**: {'✅' if case.get('metrics', {}).get('hit@1', False) else '❌'}\n")
                    f.write(f"- **PRD Score**: {case.get('prd_score', 0):.4f}\n")
                    f.write(f"- **SAS Score**: {case.get('gass_score', 0):.4f}\n")
                    f.write(f"- **TUS Score**: {case.get('tus_score', 0):.4f}\n")
                    
                    # 如果有语义漂移数据，也包含进来
                    if 'semantic_drift' in case or 'semantic_drift_analysis' in case:
                        drift_data = case.get('semantic_drift', case.get('semantic_drift_analysis', {}))
                        if drift_data:
                            f.write(f"- **Drift Slope**: {drift_data.get('drift_slope', 0):.6f}\n")
                            f.write(f"- **Drift Gap**: {drift_data.get('drift_gap', 0):.4f}\n")
                    
                    f.write("\n")
                
                f.write("\n")
            
            # Insights and Conclusions
            f.write("## Key Insights\n\n")
            f.write("### Hallucination Mechanisms\n\n")
            
            for quad_name in ['Q1_high_prd_high_sas', 'Q2_low_prd_high_sas', 'Q3_low_prd_low_sas', 'Q4_high_prd_low_sas']:
                if quad_name in analysis:
                    data = analysis[quad_name]
                    quad_type = quad_name.replace('_', ' ').replace('high', 'High').replace('low', 'Low').replace('prd', 'PRD').replace('sas', 'SAS')
                    
                    f.write(f"#### {quad_type}\n")
                    f.write(f"- Sample count: {data['count']}\n")
                    f.write(f"- Hallucination rate: {data['hallucination_rate']:.1f}%\n")
                    
                    if 'high_prd_high_sas' in quad_name:
                        f.write("- **Mechanism**: Over-reliance on shortest paths with good semantic alignment\n")
                        f.write("- **Interpretation**: Models follow logical reasoning paths but may miss broader context\n")
                    elif 'low_prd_high_sas' in quad_name:
                        f.write("- **Mechanism**: Balanced reasoning with good semantic alignment\n")
                        f.write("- **Interpretation**: Ideal behavior - models integrate multiple information sources\n")
                    elif 'low_prd_low_sas' in quad_name:
                        f.write("- **Mechanism**: Unfocused reasoning with poor semantic alignment\n")
                        f.write("- **Interpretation**: Models generate plausible but semantically ungrounded responses\n")
                    elif 'high_prd_low_sas' in quad_name:
                        f.write("- **Mechanism**: Path-dependent but semantically incorrect reasoning\n")
                        f.write("- **Interpretation**: Models follow paths but misinterpret semantic content\n")
                    
                    f.write("\n")
        
        print(f"   📋 Detailed report saved to: {report_file}")
        return report_file
    
    def save_classified_data(self):
        """保存分类后的数据以供进一步分析"""
        print("\n💾 Saving classified data...")
        
        # 保存每个象限的数据
        for quad_name, samples in self.quadrant_samples.items():
            if samples:
                output_file = os.path.join(self.output_dir, f'{quad_name}_samples.jsonl')
                with open(output_file, 'w', encoding='utf-8') as f:
                    for sample in samples:
                        f.write(json.dumps(sample, ensure_ascii=False) + '\n')
                print(f"   💾 Saved {len(samples)} samples to {output_file}")
        
        # 保存汇总统计
        summary_file = os.path.join(self.output_dir, 'classification_summary.json')
        summary = {
            'total_samples': len(self.samples),
            'quadrant_counts': {quad: len(samples) for quad, samples in self.quadrant_samples.items()},
            'timestamp': datetime.now().isoformat()
        }
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        
        print(f"   📊 Summary saved to {summary_file}")
    
    def run_analysis(self, num_cases: int = 3, max_samples: int = 5000):
        """运行完整的案例研究分析"""
        print("🚀 Starting Hallucination Case Study Analysis")
        print("=" * 60)
        
        # 1. 加载数据
        self.load_data(max_samples)
        
        if not self.samples:
            print("❌ No valid samples found!")
            return
        
        # 2. 分类样本
        self.classify_samples()
        
        # 3. 分析象限特征
        analysis = self.analyze_quadrant_characteristics()
        
        # 4. 选择代表性案例
        representative_cases = self.select_representative_cases(num_cases)
        
        # 5. 创建可视化
        self.create_visualizations(analysis)
        
        # 6. 生成详细报告
        report_file = self.generate_detailed_report(analysis, representative_cases)
        
        # 7. 保存分类数据
        self.save_classified_data()
        
        print("\n" + "=" * 60)
        print("✅ Case Study Analysis Complete!")
        print(f"📁 Results saved to: {self.output_dir}")
        print(f"📋 Main report: {report_file}")
        print("=" * 60)
        
        return analysis, representative_cases

def main():
    parser = argparse.ArgumentParser(description='Hallucination Case Study Analysis')
    parser.add_argument('--input_file', type=str, required=True, 
                       help='Input JSONL file with inference results')
    parser.add_argument('--output_dir', type=str, default='case_study_results',
                       help='Output directory for analysis results')
    parser.add_argument('--num_cases', type=int, default=3,
                       help='Number of representative cases per quadrant')
    parser.add_argument('--max_samples', type=int, default=5000,
                       help='Maximum number of samples to analyze')
    
    args = parser.parse_args()
    
    # 创建分析器并运行
    analyzer = HallucinationCaseStudy(args.input_file, args.output_dir)
    analysis, cases = analyzer.run_analysis(args.num_cases, args.max_samples)
    
    return analysis, cases

if __name__ == "__main__":
    main()
    