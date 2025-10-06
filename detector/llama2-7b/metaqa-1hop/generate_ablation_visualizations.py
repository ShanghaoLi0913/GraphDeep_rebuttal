#!/usr/bin/env python3
"""
生成消融实验可视化图表
A: Feature Importance可视化 (GGA-Full)
B: ROC curves对比 (GGA-Core vs GGA-Full)
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import joblib
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 设置matplotlib字体以支持中文（如果需要）
plt.rcParams['font.family'] = ['DejaVu Sans']
plt.rcParams['figure.dpi'] = 300

class AblationVisualizer:
    """消融实验可视化类"""
    
    def __init__(self):
        self.feature_configs = {
            'GGA_Core': {
                'features': ['gass_score', 'prd_score'],
                'description': 'Mechanistic features only',
                'color': '#4ECDC4'
            },
            'GGA_Full': {
                'features': ['gass_score', 'prd_score', 'output_length', 'repetition_score', 
                           'avg_word_length', 'unique_word_ratio', 'has_ans_prefix', 'comma_count', 'question_mark_count'],
                'description': 'Mechanistic + Surface features',
                'color': '#45B7D1'
            }
        }
        
        self.feature_names_readable = {
            'gass_score': 'SAS Score',
            'prd_score': 'PRD Score', 
            'output_length': 'Answer Length',
            'repetition_score': 'Repetition Ratio',
            'avg_word_length': 'Avg Word Length',
            'unique_word_ratio': 'Unique Word Ratio',
            'has_ans_prefix': 'Has "Ans:" Prefix',
            'comma_count': 'Comma Count',
            'question_mark_count': 'Question Mark Count'
        }
    
    def extract_features(self, data, feature_set):
        """提取特征"""
        features = []
        labels = []
        
        for item in data:
            feature_dict = {
                'gass_score': item.get('gass_score', 0),
                'prd_score': item.get('prd_score', 0),
            }
            
            # 表面特征
            if any(f in feature_set for f in ['output_length', 'repetition_score', 'avg_word_length', 
                                            'unique_word_ratio', 'has_ans_prefix', 'comma_count', 'question_mark_count']):
                if 'model_output' in item:
                    output = item['model_output']
                    words = output.split()
                    feature_dict.update({
                        'output_length': len(words),
                        'repetition_score': self.calculate_repetition(output),
                        'avg_word_length': np.mean([len(w) for w in words]) if words else 0,
                        'unique_word_ratio': len(set(words)) / len(words) if words else 0,
                        'has_ans_prefix': 1 if 'ans:' in output.lower() else 0,
                        'comma_count': output.count(','),
                        'question_mark_count': output.count('?'),
                    })
                else:
                    feature_dict.update({
                        'output_length': 0,
                        'repetition_score': 0,
                        'avg_word_length': 0,
                        'unique_word_ratio': 0,
                        'has_ans_prefix': 0,
                        'comma_count': 0,
                        'question_mark_count': 0,
                    })
            
            # 选择指定特征
            selected_features = [feature_dict[f] for f in feature_set]
            features.append(selected_features)
            
            # 提取标签
            if 'squad_evaluation' in item:
                is_hallucination = item['squad_evaluation'].get('squad_is_hallucination', False)
            else:
                is_hallucination = not item.get('metrics', {}).get('hit@1', False)
            labels.append(int(is_hallucination))
        
        X = np.array(features)
        y = np.array(labels)
        
        # 处理NaN和无穷大值
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        
        return X, y
    
    def calculate_repetition(self, text):
        """计算重复率"""
        words = text.lower().split()
        if len(words) <= 1:
            return 0
        unique_words = len(set(words))
        return 1 - (unique_words / len(words))
    
    def plot_feature_importance(self, train_file, test_file):
        """绘制GGA-Full的特征重要性"""
        print("📊 生成特征重要性可视化...")
        
        # 加载数据
        with open(train_file, 'r', encoding='utf-8') as f:
            next(f)
            train_data = [json.loads(line) for line in f if line.strip()]
        
        # 提取GGA-Full特征
        feature_set = self.feature_configs['GGA_Full']['features']
        X_train, y_train = self.extract_features(train_data, feature_set)
        
        # 标准化
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        
        # 训练验证分割
        X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
            X_train_scaled, y_train, test_size=0.2, random_state=42, stratify=y_train
        )
        
        # SMOTE处理类别不平衡
        try:
            smote = SMOTE(random_state=42)
            X_train_balanced, y_train_balanced = smote.fit_resample(X_train_split, y_train_split)
        except:
            X_train_balanced, y_train_balanced = X_train_split, y_train_split
        
        # 训练XGBoost模型
        model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.8,
            random_state=42,
            eval_metric='logloss'
        )
        
        model.fit(X_train_balanced, y_train_balanced)
        
        # 获取特征重要性
        importance_scores = model.feature_importances_
        
        # 创建可读的特征名称
        readable_names = [self.feature_names_readable[f] for f in feature_set]
        
        # 按重要性排序
        indices = np.argsort(importance_scores)[::-1]
        sorted_features = [readable_names[i] for i in indices]
        sorted_scores = importance_scores[indices]
        
        # 绘制特征重要性
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # 区分mechanistic和surface特征的颜色
        colors = []
        for feature in sorted_features:
            if feature in ['SAS Score', 'PRD Score']:
                colors.append('#2E86AB')  # 深蓝色 - mechanistic
            else:
                colors.append('#A23B72')  # 紫色 - surface
        
        bars = ax.barh(range(len(sorted_features)), sorted_scores, color=colors, alpha=0.8)
        
        # 添加数值标签
        for i, (bar, score) in enumerate(zip(bars, sorted_scores)):
            ax.text(score + max(sorted_scores) * 0.01, i, f'{score:.3f}', 
                   va='center', ha='left', fontsize=10, fontweight='bold')
        
        ax.set_yticks(range(len(sorted_features)))
        ax.set_yticklabels(sorted_features, fontsize=11)
        ax.set_xlabel('Feature Importance', fontsize=12, fontweight='bold')
        ax.set_title('Feature Importance Analysis (GGA-Full on LLaMA2-7B)', 
                    fontsize=14, fontweight='bold', pad=20)
        ax.grid(axis='x', alpha=0.3)
        
        # 添加图例
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#2E86AB', label='Mechanistic Features'),
            Patch(facecolor='#A23B72', label='Surface Features')
        ]
        ax.legend(handles=legend_elements, loc='lower right', fontsize=10)
        
        plt.tight_layout()
        
        # 保存图片
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = "results/ablation_visualizations"
        os.makedirs(output_dir, exist_ok=True)
        
        plt.savefig(f"{output_dir}/feature_importance_{timestamp}.png", 
                   dpi=300, bbox_inches='tight')
        plt.savefig(f"{output_dir}/feature_importance_{timestamp}.pdf", 
                   bbox_inches='tight')
        
        print(f"💾 特征重要性图保存至: {output_dir}/feature_importance_{timestamp}.png")
        plt.show()
        
        return model, scaler
    
    def plot_roc_comparison(self, train_file, test_file):
        """绘制GGA-Core vs GGA-Full的ROC对比"""
        print("📊 生成ROC曲线对比...")
        
        # 加载数据
        with open(train_file, 'r', encoding='utf-8') as f:
            next(f)
            train_data = [json.loads(line) for line in f if line.strip()]
        
        with open(test_file, 'r', encoding='utf-8') as f:
            next(f)
            test_data = [json.loads(line) for line in f if line.strip()]
        
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # 对每个配置计算ROC曲线
        for config_name, config in self.feature_configs.items():
            feature_set = config['features']
            color = config['color']
            
            # 提取特征
            X_train, y_train = self.extract_features(train_data, feature_set)
            X_test, y_test = self.extract_features(test_data, feature_set)
            
            # 标准化
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # 训练验证分割
            X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
                X_train_scaled, y_train, test_size=0.2, random_state=42, stratify=y_train
            )
            
            # SMOTE处理类别不平衡
            try:
                smote = SMOTE(random_state=42)
                X_train_balanced, y_train_balanced = smote.fit_resample(X_train_split, y_train_split)
            except:
                X_train_balanced, y_train_balanced = X_train_split, y_train_split
            
            # 训练模型
            model = xgb.XGBClassifier(
                n_estimators=200,
                max_depth=5,
                learning_rate=0.1,
                subsample=0.8,
                random_state=42,
                eval_metric='logloss'
            )
            
            model.fit(X_train_balanced, y_train_balanced)
            
            # 在测试集上预测
            y_proba = model.predict_proba(X_test_scaled)[:, 1]
            
            # 计算ROC曲线
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            roc_auc = auc(fpr, tpr)
            
            # 绘制ROC曲线
            ax.plot(fpr, tpr, color=color, lw=3, 
                   label=f'{config_name} (AUC = {roc_auc:.3f})', alpha=0.8)
        
        # 绘制随机分类器线
        ax.plot([0, 1], [0, 1], 'k--', lw=2, alpha=0.6, label='Random Classifier')
        
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
        ax.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
        ax.set_title('ROC Curve Comparison: GGA-Core vs GGA-Full (LLaMA2-7B)', 
                    fontsize=14, fontweight='bold', pad=20)
        ax.legend(loc="lower right", fontsize=11)
        ax.grid(alpha=0.3)
        
        # 设置相等的纵横比
        ax.set_aspect('equal', adjustable='box')
        
        plt.tight_layout()
        
        # 保存图片
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = "results/ablation_visualizations"
        os.makedirs(output_dir, exist_ok=True)
        
        plt.savefig(f"{output_dir}/roc_comparison_{timestamp}.png", 
                   dpi=300, bbox_inches='tight')
        plt.savefig(f"{output_dir}/roc_comparison_{timestamp}.pdf", 
                   bbox_inches='tight')
        
        print(f"💾 ROC对比图保存至: {output_dir}/roc_comparison_{timestamp}.png")
        plt.show()
    
    def generate_both_plots(self, train_file, test_file):
        """生成两个图表"""
        print("🚀 生成消融实验可视化图表")
        print("="*50)
        
        # 生成特征重要性图
        model, scaler = self.plot_feature_importance(train_file, test_file)
        
        print("\n" + "-"*50)
        
        # 生成ROC对比图
        self.plot_roc_comparison(train_file, test_file)
        
        print("\n✅ 所有可视化图表生成完成!")
        print("📁 图表保存在: results/ablation_visualizations/")
        print("\n💡 使用建议:")
        print("   Figure A: Feature Importance - 展示mechanistic vs surface特征贡献")
        print("   Figure B: ROC Comparison - 视觉展示GGA-Core到GGA-Full的改进")

def main():
    # 切换到正确目录
    os.chdir('/mnt/d/experiments/GraphDeEP/detector/llama2-7b/metaqa-1hop')
    
    # 创建可视化器
    visualizer = AblationVisualizer()
    
    # 数据文件路径
    train_file = '/mnt/d/experiments/GraphDeEP/experiment_records/inference_results/llama2-7b/colab_train_simple_part1&2.jsonl'
    test_file = '/mnt/d/experiments/GraphDeEP/experiment_records/inference_results/llama2-7b/colab_test_simple.jsonl'
    
    # 生成图表
    visualizer.generate_both_plots(train_file, test_file)

if __name__ == "__main__":
    main()