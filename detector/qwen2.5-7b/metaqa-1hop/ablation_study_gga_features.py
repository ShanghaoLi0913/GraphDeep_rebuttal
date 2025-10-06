#!/usr/bin/env python3
"""
GGA特征消融实验 (Ablation Study) - Qwen2.5-7B版本
测试不同特征组合的检测效果：
1. SAS only - 单独评估语义对齐的检测能力
2. PRD only - 单独评估注意力路径过度依赖的检测能力  
3. PRD + SAS (GGA-Core) - 机制特征组合，具备解释性和一定检测能力
4. PRD + SAS + Surface (GGA-Full) - 主方法，检测性能最佳，兼具解释性与实用性
"""

import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, accuracy_score
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
import joblib
import warnings
warnings.filterwarnings('ignore')

class GGAAblationStudy:
    """GGA特征消融实验类"""
    
    def __init__(self):
        self.results = {}
        self.models = {}
        self.scalers = {}
        
        # 定义特征组合
        self.feature_configs = {
            'SAS_only': {
                'features': ['gass_score'],
                'description': '单独评估语义对齐的检测能力'
            },
            'PRD_only': {
                'features': ['prd_score'], 
                'description': '单独评估注意力路径过度依赖的检测能力'
            },
            'GGA_Core': {
                'features': ['gass_score', 'prd_score'],
                'description': '机制特征组合，具备解释性和一定检测能力'
            },
            'GGA_Full': {
                'features': ['gass_score', 'prd_score', 'output_length', 'repetition_score', 
                           'avg_word_length', 'unique_word_ratio', 'has_ans_prefix', 'comma_count', 'question_mark_count'],
                'description': '主方法，检测性能最佳，兼具解释性与实用性'
            }
        }
    
    def extract_features(self, data, feature_set):
        """根据特征集提取相应特征"""
        features = []
        labels = []
        
        for item in data:
            feature_dict = {
                'gass_score': item.get('gass_score', 0),
                'prd_score': item.get('prd_score', 0),
            }
            
            # 表面特征（只在GGA_Full时需要）
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
                    # 默认值
                    feature_dict.update({
                        'output_length': 0,
                        'repetition_score': 0,
                        'avg_word_length': 0,
                        'unique_word_ratio': 0,
                        'has_ans_prefix': 0,
                        'comma_count': 0,
                        'question_mark_count': 0,
                    })
            
            # 只选择指定的特征
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
    
    def optimize_threshold(self, y_true, y_proba):
        """优化分类阈值"""
        thresholds = np.arange(0.1, 0.9, 0.01)
        best_threshold = 0.5
        best_f1 = 0
        
        for threshold in thresholds:
            y_pred = (y_proba >= threshold).astype(int)
            f1 = f1_score(y_true, y_pred)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
        
        return best_threshold, best_f1
    
    def train_single_config(self, config_name, X_train, y_train, X_val, y_val, X_test, y_test):
        """训练单个特征配置的模型"""
        print(f"\n🔍 训练 {config_name}...")
        print(f"   特征: {self.feature_configs[config_name]['features']}")
        print(f"   说明: {self.feature_configs[config_name]['description']}")
        
        # 使用XGBoost作为主要模型（表现通常最好）
        model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.8,
            random_state=42,
            eval_metric='logloss'
        )
        
        # 训练模型
        model.fit(X_train, y_train)
        
        # 在验证集上优化阈值
        y_val_proba = model.predict_proba(X_val)[:, 1]
        threshold, _ = self.optimize_threshold(y_val, y_val_proba)
        
        # 在测试集上评估
        y_test_proba = model.predict_proba(X_test)[:, 1]
        y_test_pred = (y_test_proba >= threshold).astype(int)
        
        # 计算指标
        metrics = {
            'auc': roc_auc_score(y_test, y_test_proba),
            'f1_score': f1_score(y_test, y_test_pred),
            'precision': precision_score(y_test, y_test_pred),
            'recall': recall_score(y_test, y_test_pred),
            'accuracy': accuracy_score(y_test, y_test_pred),
            'threshold': threshold,
            'feature_count': len(self.feature_configs[config_name]['features'])
        }
        
        print(f"   AUC: {metrics['auc']:.4f}")
        print(f"   F1-Score: {metrics['f1_score']:.4f}")
        print(f"   Precision: {metrics['precision']:.4f}")
        print(f"   Recall: {metrics['recall']:.4f}")
        
        self.results[config_name] = metrics
        self.models[config_name] = model
        
        return metrics
    
    def run_ablation_study(self, train_file, test_file):
        """运行完整的消融实验"""
        print("🚀 GGA特征消融实验 (Qwen2.5-7B)")
        print("="*60)
        
        # 加载数据
        print("📥 加载数据...")
        with open(train_file, 'r', encoding='utf-8') as f:
            next(f)  # 跳过配置行
            train_data = [json.loads(line) for line in f if line.strip()]
        
        with open(test_file, 'r', encoding='utf-8') as f:
            next(f)  # 跳过配置行
            test_data = [json.loads(line) for line in f if line.strip()]
        
        print(f"训练样本: {len(train_data)}")
        print(f"测试样本: {len(test_data)}")
        
        # 对每个特征配置进行实验
        for config_name, config in self.feature_configs.items():
            feature_set = config['features']
            
            # 提取特征
            X_train, y_train = self.extract_features(train_data, feature_set)
            X_test, y_test = self.extract_features(test_data, feature_set)
            
            # 特征标准化
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            self.scalers[config_name] = scaler
            
            # 训练验证分割
            X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
                X_train_scaled, y_train, test_size=0.2, random_state=42, stratify=y_train
            )
            
            # 处理类别不平衡（对于单特征可能会有问题，所以加try-except）
            try:
                smote = SMOTE(random_state=42)
                X_train_balanced, y_train_balanced = smote.fit_resample(X_train_split, y_train_split)
            except ValueError as e:
                print(f"   ⚠️ SMOTE失败，使用原始数据: {e}")
                X_train_balanced, y_train_balanced = X_train_split, y_train_split
            
            # 训练和评估
            self.train_single_config(config_name, X_train_balanced, y_train_balanced, 
                                   X_val_split, y_val_split, X_test_scaled, y_test)
    
    def plot_ablation_results(self):
        """绘制消融实验结果"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        configs = list(self.results.keys())
        metrics = ['auc', 'f1_score', 'precision', 'recall']
        metric_names = ['AUC', 'F1-Score', 'Precision', 'Recall']
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
        
        for i, (metric, metric_name) in enumerate(zip(metrics, metric_names)):
            ax = axes[i//2, i%2]
            
            values = [self.results[config][metric] for config in configs]
            bars = ax.bar(range(len(configs)), values, color=colors, alpha=0.8)
            
            # 添加数值标签
            for j, (bar, value) in enumerate(zip(bars, values)):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
            
            ax.set_ylabel(metric_name, fontsize=12)
            ax.set_title(f'{metric_name} - Feature Ablation Study', fontsize=14, fontweight='bold')
            ax.set_xticks(range(len(configs)))
            ax.set_xticklabels(configs, rotation=45, ha='right')
            ax.set_ylim(0, 1.1)
            ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        # 保存图片
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = "results/ablation_study"
        os.makedirs(output_dir, exist_ok=True)
        
        output_path = f"{output_dir}/gga_ablation_study_{timestamp}.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\n💾 消融实验图表保存至: {output_path}")
        
        plt.show()
        
        return timestamp
    
    def generate_ablation_table(self):
        """生成消融实验结果表格"""
        print("\n📊 GGA特征消融实验结果表格 (Qwen2.5-7B)")
        print("="*80)
        
        # 创建表格数据
        table_data = []
        for config_name, metrics in self.results.items():
            config = self.feature_configs[config_name]
            table_data.append({
                'Feature Set': config_name,
                'AUC': f"{metrics['auc']:.3f}",
                'F1': f"{metrics['f1_score']:.3f}",
                'Precision': f"{metrics['precision']:.3f}",
                'Recall': f"{metrics['recall']:.3f}",
                'Features': len(config['features']),
                '用途定位': config['description']
            })
        
        # 打印表格
        df = pd.DataFrame(table_data)
        print(df.to_string(index=False))
        
        return df
    
    def save_results(self, timestamp):
        """保存实验结果"""
        output_dir = "results/ablation_study"
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存详细结果
        results_data = {
            'timestamp': timestamp,
            'experiment_type': 'gga_feature_ablation',
            'feature_configs': self.feature_configs,
            'results': self.results,
            'summary': {
                'best_auc': max(self.results.keys(), key=lambda x: self.results[x]['auc']),
                'best_f1': max(self.results.keys(), key=lambda x: self.results[x]['f1_score']),
                'best_precision': max(self.results.keys(), key=lambda x: self.results[x]['precision']),
                'best_recall': max(self.results.keys(), key=lambda x: self.results[x]['recall'])
            }
        }
        
        with open(f"{output_dir}/ablation_results_{timestamp}.json", 'w', encoding='utf-8') as f:
            json.dump(results_data, f, indent=2, ensure_ascii=False)
        
        # 保存模型
        models_dir = f"{output_dir}/models_{timestamp}"
        os.makedirs(models_dir, exist_ok=True)
        
        for config_name, model in self.models.items():
            joblib.dump(model, f"{models_dir}/{config_name}_model.joblib")
            joblib.dump(self.scalers[config_name], f"{models_dir}/{config_name}_scaler.joblib")
        
        print(f"\n💾 实验结果保存至: {output_dir}")
        print(f"🤖 模型保存至: {models_dir}")

def main():
    # 切换到检测器目录
    os.chdir('/mnt/d/experiments/GraphDeEP/detector/qwen2.5-7b/metaqa-1hop')
    
    # 创建消融实验对象
    ablation = GGAAblationStudy()
    
    # 数据文件路径 - 更新为qwen2.5-7b路径
    train_file = '/mnt/d/experiments/GraphDeEP/experiment_records/inference_results/qwen2.5-7b/colab_train_simple_part1&2.jsonl'
    test_file = '/mnt/d/experiments/GraphDeEP/experiment_records/inference_results/qwen2.5-7b/colab_test_simple.jsonl'
    
    # 运行消融实验
    ablation.run_ablation_study(train_file, test_file)
    
    # 生成结果表格
    df = ablation.generate_ablation_table()
    
    # 绘制结果图表
    timestamp = ablation.plot_ablation_results()
    
    # 保存结果
    ablation.save_results(timestamp)
    
    print(f"\n✅ GGA特征消融实验完成!")
    
    # 显示最佳配置
    best_config = max(ablation.results.keys(), key=lambda x: ablation.results[x]['f1_score'])
    best_metrics = ablation.results[best_config]
    
    print(f"\n🏆 最佳配置: {best_config}")
    print(f"   AUC: {best_metrics['auc']:.4f}")
    print(f"   F1-Score: {best_metrics['f1_score']:.4f}")
    print(f"   Precision: {best_metrics['precision']:.4f}")
    print(f"   Recall: {best_metrics['recall']:.4f}")

if __name__ == "__main__":
    main()