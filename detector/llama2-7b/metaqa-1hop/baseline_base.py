#!/usr/bin/env python3
"""
Baseline方法基础类 - 用于单独调试每个baseline方法
"""

import json
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
from sklearn.metrics import classification_report, roc_auc_score, f1_score, precision_score, recall_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class BaselineBase:
    """基础baseline检测器"""
    
    def __init__(self, model_name="meta-llama/Llama-2-7b-chat-hf"):
        """初始化基础检测器 - Colab L4 GPU优化版"""
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 🚀 Colab L4 GPU优化配置 (22.5GB显存)
        if torch.cuda.is_available():
            gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"💾 GPU Memory: {gpu_memory_gb:.1f} GB")
            
            if gpu_memory_gb > 20:  # L4 GPU
                self.batch_size = 32  # 进一步增大batch size
                self.max_length = 256  # 减少序列长度提速
                print("🚀 L4 GPU优化: batch_size=32, max_length=256")
            else:
                self.batch_size = 8
                self.max_length = 512
                print("⚡ 标准GPU配置: batch_size=8, max_length=512")
        else:
            self.batch_size = 4
            self.max_length = 256
            print("🖥️ CPU模式: batch_size=4, max_length=256")
        
        print(f"📱 Device: {self.device}")
    
    def load_data(self, max_train_samples=100, max_test_samples=50):
        """加载训练和测试数据"""
        print("📥 Loading training and test data...")
        
        # 加载训练数据 - 自动检测路径
        train_samples = []
        
        # 🚀 智能路径检测：支持本地和Colab环境
        base_paths = [
            '/content/GraphDeep/experiment_records/inference_results/llama2-7b/',  # Colab路径
            '/mnt/d/experiments/GraphDeEP/experiment_records/inference_results/llama2-7b/',  # 本地路径
        ]
        
        train_files = []
        for base_path in base_paths:
            if os.path.exists(base_path):
                train_files = [
                    os.path.join(base_path, 'colab_train_simple_part1&2.jsonl'),
                    os.path.join(base_path, 'colab_train_simple_part1.jsonl'),
                    os.path.join(base_path, 'colab_train_simple_part2.jsonl'),
                ]
                break
        
        train_file = None
        for tf in train_files:
            if os.path.exists(tf):
                train_file = tf
                break
        
        if train_file is None:
            print("❌ No training data file found")
            return None, None
        
        try:
            with open(train_file, 'r', encoding='utf-8') as f:
                next(f)  # 跳过第一行配置信息
                for line in f:
                    if line.strip():
                        data = json.loads(line)
                        train_samples.append(data)
                        if len(train_samples) >= max_train_samples:
                            break
        except Exception as e:
            print(f"❌ Error loading training data from {train_file}: {e}")
            return None, None
        
        # 加载测试数据 - 使用相同的base_path
        test_samples = []
        test_files = []
        for base_path in base_paths:
            if os.path.exists(base_path):
                test_files = [
                    os.path.join(base_path, 'colab_test_simple.jsonl')
                ]
                break
        
        test_file = None
        for tf in test_files:
            if os.path.exists(tf):
                test_file = tf
                break
        
        if test_file is None:
            print("❌ No test data file found")
            return None, None
        
        try:
            with open(test_file, 'r', encoding='utf-8') as f:
                next(f)  # 跳过第一行配置信息
                for line in f:
                    if line.strip():
                        data = json.loads(line)
                        test_samples.append(data)
                        if len(test_samples) >= max_test_samples:
                            break
        except Exception as e:
            print(f"❌ Error loading test data from {test_file}: {e}")
            return None, None
        
        print(f"✅ Loaded {len(train_samples)} training samples from {train_file}")
        print(f"✅ Loaded {len(test_samples)} test samples from {test_file}")
        return train_samples, test_samples
    
    def extract_labels(self, samples):
        """从样本中提取标签（0=正确，1=幻觉），改为用SQuAD评估"""
        labels = []
        for sample in samples:
            # 用SQuAD评估的标签
            squad_eval = sample.get('squad_evaluation', {})
            is_hallucination = squad_eval.get('squad_is_hallucination', False)
            labels.append(1 if is_hallucination else 0)  # 1=幻觉，0=正确
        return np.array(labels)
    
    def optimize_threshold(self, y_true, y_proba, method_name=None):
        """使用PRD+SAS的简单F1优化策略"""
        from sklearn.metrics import precision_recall_curve
        
        # 🎯 使用PRD+SAS的简单策略：直接最大化F1
        precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)
        f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
        
        best_idx = np.argmax(f1_scores)
        best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
        best_f1 = f1_scores[best_idx]
        
        # 获取最优阈值下的所有指标
        y_pred = (y_proba >= best_threshold).astype(int)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        
        # 输出选择的指标信息
        print(f"    📊 F1-optimized - P:{precision:.3f}, R:{recall:.3f}, F1:{best_f1:.3f}")
        print(f"    📊 Threshold: {best_threshold:.3f}")
        
        return best_threshold, best_f1
    
    def evaluate_method(self, train_samples, test_samples, method_name, feature_extractor):
        """评估单个方法"""
        print(f"\n🔍 Evaluating {method_name}...")
        
        # 提取特征和标签
        print("📊 Extracting features...")
        X_train = feature_extractor(train_samples)
        y_train = self.extract_labels(train_samples)
        
        X_test = feature_extractor(test_samples)
        y_test = self.extract_labels(test_samples)
        
        # 数据预处理
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train.reshape(-1, 1))
        X_test_scaled = scaler.transform(X_test.reshape(-1, 1))
        
        # 训练验证分割
        X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
            X_train_scaled, y_train, test_size=0.2, random_state=42, stratify=y_train
        )
        
        print(f"📊 Data splits: Train={len(X_train_split)}, Val={len(X_val_split)}, Test={len(X_test_scaled)}")
        print(f"📊 Label distribution - Train: {np.bincount(y_train_split)}, Val: {np.bincount(y_val_split)}, Test: {np.bincount(y_test)}")
        
        # 训练模型
        model = LogisticRegression(random_state=42, class_weight='balanced')
        model.fit(X_train_split, y_train_split)
        
        # 在验证集上优化阈值
        y_proba_val = model.predict_proba(X_val_split)[:, 1]
        threshold, f1_opt = self.optimize_threshold(y_val_split, y_proba_val, method_name)
        
        print(f"📊 Optimized threshold: {threshold:.3f} (Val F1: {f1_opt:.3f})")
        
        # 在测试集上评估
        y_proba_test = model.predict_proba(X_test_scaled)[:, 1]
        y_pred_test = (y_proba_test >= threshold).astype(int)
        
        # 🔍 详细调试信息
        print(f"\n🔍 DEBUG: Test predictions analysis")
        print(f"   Test probabilities range: [{y_proba_test.min():.3f}, {y_proba_test.max():.3f}]")
        print(f"   Threshold: {threshold:.3f}")
        print(f"   Predicted as hallucination: {np.sum(y_pred_test)}/{len(y_pred_test)}")
        print(f"   Actually hallucination: {np.sum(y_test)}/{len(y_test)}")
        
        # 显示混淆矩阵信息
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_test, y_pred_test)
        print(f"   Confusion Matrix:")
        print(f"   TN={cm[0,0]}, FP={cm[0,1]}")
        print(f"   FN={cm[1,0]}, TP={cm[1,1]}")
        
        # 计算指标
        auc = roc_auc_score(y_test, y_proba_test)
        precision = precision_score(y_test, y_pred_test, zero_division=0)
        recall = recall_score(y_test, y_pred_test, zero_division=0)
        f1 = f1_score(y_test, y_pred_test, zero_division=0)
        accuracy = accuracy_score(y_test, y_pred_test)
        
        print(f"   Precision = TP/(TP+FP) = {cm[1,1]}/({cm[1,1]}+{cm[0,1]}) = {precision:.3f}")
        print(f"   Recall = TP/(TP+FN) = {cm[1,1]}/({cm[1,1]}+{cm[1,0]}) = {recall:.3f}")
        
        # 详细分类报告
        report = classification_report(y_test, y_pred_test, output_dict=True, zero_division=0)
        
        results = {
            'auc': float(auc),
            'threshold': float(threshold),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'accuracy': float(accuracy),
            'classification_report': report
        }
        
        # 打印结果
        print(f"✅ Results for {method_name}:")
        print(f"   AUC: {auc:.4f}")
        print(f"   Precision: {precision:.4f}")
        print(f"   Recall: {recall:.4f}")
        print(f"   F1-Score: {f1:.4f}")
        print(f"   Accuracy: {accuracy:.4f}")
        
        return results
    
    def save_results(self, method_name, results):
        """保存结果"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(os.path.dirname(__file__), "results", "individual_baselines")
        os.makedirs(output_dir, exist_ok=True)
        
        output_file = os.path.join(output_dir, f"{method_name.lower().replace(' ', '_')}_results_{timestamp}.json")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({method_name: results}, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Results saved to: {output_file}")
        return output_file