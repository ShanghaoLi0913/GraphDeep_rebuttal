#!/usr/bin/env python3
"""
简化版检测器：仅使用SAS (gass_score) 和 PRD (prd_score) 特征
用于验证核心图特征的有效性
"""

import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, precision_recall_curve
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
import joblib
import warnings
warnings.filterwarnings('ignore')

class SasPrdOnlyDetector:
    """仅使用SAS和PRD特征的检测器"""
    
    def __init__(self):
        self.models = {}
        self.best_models = {}
        self.scaler = StandardScaler()
        self.results = {}
        self.feature_names = ['gass_score', 'prd_score']
        self.best_threshold = {}
        
    def extract_features(self, data):
        """提取SAS和PRD特征"""
        features = []
        labels = []
        
        for item in data:
            # 只提取核心图特征
            feature_dict = {
                'gass_score': item.get('gass_score', 0),
                'prd_score': item.get('prd_score', 0),
            }
            features.append(list(feature_dict.values()))
            
            # 提取标签
            if 'squad_evaluation' in item:
                is_hallucination = item['squad_evaluation'].get('squad_is_hallucination', False)
            else:
                is_hallucination = not item.get('metrics', {}).get('hit@1', False)
            labels.append(int(is_hallucination))
        
        X = np.array(features)
        y = np.array(labels)
        
        print(f"SAS+PRD特征: {len(self.feature_names)}")
        print(f"样本数量: {len(X)}")
        print(f"标签分布: {np.bincount(y)}")
        print(f"幻觉率: {np.mean(y):.3f}")
        
        return X, y
    
    def optimize_threshold(self, y_true, y_proba, method_name):
        """优化分类阈值以最大化F1分数"""
        from sklearn.metrics import f1_score
        
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
    
    def train_models(self, X_train, y_train, X_val, y_val):
        """训练多种模型"""
        print("\n🚀 训练SAS+PRD检测器...")
        
        # 定义模型参数
        model_configs = {
            'LogisticRegression': {
                'model': LogisticRegression(random_state=42, class_weight='balanced'),
                'params': {
                    'C': [0.01, 0.1, 1, 10, 100],
                    'penalty': ['l1', 'l2'],
                    'solver': ['liblinear']
                }
            },
            'RandomForest': {
                'model': RandomForestClassifier(random_state=42, class_weight='balanced'),
                'params': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [3, 5, 7, None],
                    'min_samples_split': [2, 5, 10]
                }
            },
            'XGBoost': {
                'model': xgb.XGBClassifier(random_state=42, eval_metric='logloss'),
                'params': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [3, 5, 7],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'subsample': [0.8, 1.0]
                }
            },
            'LightGBM': {
                'model': lgb.LGBMClassifier(random_state=42, verbose=-1),
                'params': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [3, 5, 7],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'num_leaves': [31, 50, 100]
                }
            }
        }
        
        # 训练并优化每个模型
        for name, config in model_configs.items():
            print(f"\n训练 {name}...")
            
            # 网格搜索
            grid_search = GridSearchCV(
                config['model'], 
                config['params'], 
                cv=5, 
                scoring='f1',
                n_jobs=-1
            )
            
            grid_search.fit(X_train, y_train)
            
            # 保存最佳模型
            self.best_models[name] = grid_search.best_estimator_
            
            # 在验证集上评估
            y_val_proba = grid_search.best_estimator_.predict_proba(X_val)[:, 1]
            
            # 优化阈值
            threshold, f1_opt = self.optimize_threshold(y_val, y_val_proba, name)
            self.best_threshold[name] = threshold
            
            print(f"  最佳参数: {grid_search.best_params_}")
            print(f"  验证集F1: {f1_opt:.4f}")
            print(f"  最优阈值: {threshold:.3f}")
    
    def evaluate_models(self, X_test, y_test):
        """在测试集上评估所有模型"""
        print("\n📊 测试集评估结果:")
        
        for name, model in self.best_models.items():
            print(f"\n{name}:")
            
            # 预测
            y_test_proba = model.predict_proba(X_test)[:, 1]
            y_test_pred = (y_test_proba >= self.best_threshold[name]).astype(int)
            
            # 计算指标
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
            
            metrics = {
                'accuracy': accuracy_score(y_test, y_test_pred),
                'precision': precision_score(y_test, y_test_pred),
                'recall': recall_score(y_test, y_test_pred),
                'f1_score': f1_score(y_test, y_test_pred),
                'auc': roc_auc_score(y_test, y_test_proba)
            }
            
            self.results[name] = metrics
            
            print(f"  AUC: {metrics['auc']:.4f}")
            print(f"  F1-Score: {metrics['f1_score']:.4f}")
            print(f"  Precision: {metrics['precision']:.4f}")
            print(f"  Recall: {metrics['recall']:.4f}")
            print(f"  Accuracy: {metrics['accuracy']:.4f}")
    
    def plot_results(self):
        """绘制结果对比图"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        models = list(self.results.keys())
        metrics = ['auc', 'f1_score', 'precision', 'recall']
        metric_names = ['AUC', 'F1-Score', 'Precision', 'Recall']
        
        for i, (metric, metric_name) in enumerate(zip(metrics, metric_names)):
            ax = axes[i//2, i%2]
            
            values = [self.results[model][metric] for model in models]
            bars = ax.bar(models, values, alpha=0.7)
            
            # 添加数值标签
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{value:.3f}', ha='center', va='bottom')
            
            ax.set_ylabel(metric_name)
            ax.set_title(f'{metric_name} Comparison (SAS+PRD Only)')
            ax.set_ylim(0, 1.1)
            ax.grid(axis='y', alpha=0.3)
            
            # 旋转x轴标签
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        
        # 保存图片
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"results/sas_prd_only_results/sas_prd_only_comparison_{timestamp}.png"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\n💾 结果对比图保存至: {output_path}")
        
        plt.show()
    
    def save_models_and_results(self):
        """保存模型和结果"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S") 
        
        # 创建保存目录
        save_dir = f"results/sas_prd_only_results/models"
        os.makedirs(save_dir, exist_ok=True)
        
        # 保存模型
        for name, model in self.best_models.items():
            model_path = os.path.join(save_dir, f"{name}_sas_prd_only_{timestamp}.joblib")
            joblib.dump(model, model_path)
        
        # 保存scaler
        scaler_path = os.path.join(save_dir, f"scaler_sas_prd_only_{timestamp}.joblib")
        joblib.dump(self.scaler, scaler_path)
        
        # 保存阈值
        threshold_path = os.path.join(save_dir, f"thresholds_sas_prd_only_{timestamp}.json")
        with open(threshold_path, 'w') as f:
            json.dump(self.best_threshold, f, indent=2)
        
        # 保存结果
        results_path = f"results/sas_prd_only_results/sas_prd_only_results_{timestamp}.json"
        results_data = {
            'timestamp': timestamp,
            'feature_names': self.feature_names,
            'model_results': self.results,
            'best_thresholds': self.best_threshold,
            'experiment_type': 'sas_prd_only_detection'
        }
        
        with open(results_path, 'w') as f:
            json.dump(results_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 模型和结果已保存:")
        print(f"  模型目录: {save_dir}")
        print(f"  结果文件: {results_path}")
        
        return results_data
    
    def run_experiment(self):
        """运行完整实验"""
        print("🚀 开始SAS+PRD检测器实验")
        print("="*50)
        
        # 加载数据
        train_files = [
            '/mnt/d/experiments/GraphDeEP/experiment_records/inference_results/llama2-7b/colab_train_simple_part1&2.jsonl',
            '/mnt/d/experiments/GraphDeEP/experiment_records/inference_results/llama2-7b/colab_train_simple_part1.jsonl'
        ]
        
        test_files = [
            '/mnt/d/experiments/GraphDeEP/experiment_records/inference_results/llama2-7b/colab_test_simple.jsonl'
        ]
        
        # 加载训练数据
        train_data = []
        for train_file in train_files:
            if os.path.exists(train_file):
                print(f"📥 加载训练数据: {train_file}")
                with open(train_file, 'r', encoding='utf-8') as f:
                    next(f)  # 跳过配置行
                    for line in f:
                        if line.strip():
                            train_data.append(json.loads(line))
                break
        
        # 加载测试数据
        test_data = []
        for test_file in test_files:
            if os.path.exists(test_file):
                print(f"📥 加载测试数据: {test_file}")
                with open(test_file, 'r', encoding='utf-8') as f:
                    next(f)  # 跳过配置行
                    for line in f:
                        if line.strip():
                            test_data.append(json.loads(line))
                break
        
        if not train_data or not test_data:
            print("❌ 数据加载失败")
            return None
        
        # 提取特征
        print(f"\n📊 提取SAS+PRD特征...")
        X_train, y_train = self.extract_features(train_data)
        X_test, y_test = self.extract_features(test_data)
        
        # 特征缩放
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # 训练验证分割
        X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
            X_train_scaled, y_train, test_size=0.2, random_state=42, stratify=y_train
        )
        
        print(f"训练集: {len(X_train_split)} 样本")
        print(f"验证集: {len(X_val_split)} 样本") 
        print(f"测试集: {len(X_test_scaled)} 样本")
        
        # 处理类别不平衡
        smote = SMOTE(random_state=42)
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train_split, y_train_split)
        
        print(f"SMOTE后训练集: {len(X_train_balanced)} 样本")
        print(f"平衡后标签分布: {np.bincount(y_train_balanced)}")
        
        # 训练模型
        self.train_models(X_train_balanced, y_train_balanced, X_val_split, y_val_split)
        
        # 评估模型
        self.evaluate_models(X_test_scaled, y_test)
        
        # 绘制结果
        self.plot_results()
        
        # 保存模型和结果
        results_data = self.save_models_and_results()
        
        print(f"\n✅ SAS+PRD检测器实验完成!")
        
        # 显示最佳模型
        best_model_name = max(self.results.keys(), key=lambda x: self.results[x]['f1_score'])
        best_metrics = self.results[best_model_name]
        
        print(f"\n🏆 最佳模型: {best_model_name}")
        print(f"  AUC: {best_metrics['auc']:.4f}")
        print(f"  F1-Score: {best_metrics['f1_score']:.4f}")
        print(f"  Precision: {best_metrics['precision']:.4f}")
        print(f"  Recall: {best_metrics['recall']:.4f}")
        
        return results_data

def main():
    # 切换到检测器目录
    os.chdir('/mnt/d/experiments/GraphDeEP/detector/llama2-7b/metaqa-1hop')
    
    # 运行实验
    detector = SasPrdOnlyDetector()
    results = detector.run_experiment()
    
    return results

if __name__ == "__main__":
    main()