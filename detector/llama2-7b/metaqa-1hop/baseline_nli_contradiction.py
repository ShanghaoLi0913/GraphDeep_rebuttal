#!/usr/bin/env python3
"""
NLI-based Contradiction Detection Baseline - 独立调试版本
基于自然语言推理检测问题与答案之间的矛盾来识别幻觉
"""

import numpy as np
import torch
from transformers import pipeline
from baseline_base import BaselineBase
import argparse
from tqdm import tqdm
import gc

class NLIContradictionBaseline(BaselineBase):
    """NLI-based Contradiction Detection baseline检测器"""
    
    def __init__(self):
        super().__init__()
        print("📥 Loading NLI model for contradiction detection...")
        
        # 加载NLI pipeline
        try:
            self.nli_pipeline = pipeline(
                "text-classification",
                model="facebook/bart-large-mnli",
                device=0 if torch.cuda.is_available() else -1
            )
            print("✅ NLI model loaded successfully!")
        except Exception as e:
            print(f"❌ Error loading NLI model: {e}")
            raise e
    
    def calculate_nli_contradiction_score(self, question, answer):
        """计算问题与答案之间的矛盾分数"""
        if not question or not answer:
            return 0.0  # 无内容时默认无矛盾
        
        try:
            # 构建NLI输入：问题作为前提，答案作为假设
            premise = question.strip()
            hypothesis = answer.strip()
            
            # 使用NLI模型进行推理
            result = self.nli_pipeline(f"{premise} [SEP] {hypothesis}")
            
            # 提取矛盾概率
            contradiction_score = 0.0
            for item in result:
                if item['label'].upper() == 'CONTRADICTION':
                    contradiction_score = item['score']
                    break
            
            # 返回原始NLI模型输出，不加任何人工修改
            return float(contradiction_score)
            
        except Exception as e:
            print(f"⚠️ Error in NLI contradiction detection: {e}")
            return 0.0
    
    def extract_nli_contradiction_features(self, samples):
        """提取NLI Contradiction特征"""
        features = []
        
        print("🔍 Calculating NLI contradiction scores...")
        for sample in tqdm(samples, desc="NLI Contradiction"):
            question = sample.get('question', '')
            answer = sample.get('answer', '')
            
            contradiction_score = self.calculate_nli_contradiction_score(question, answer)
            features.append(contradiction_score)
            
            # 定期清理GPU内存
            if len(features) % 20 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
        
        return np.array(features)
    
    def run_evaluation(self, max_train_samples=100, max_test_samples=50):
        """运行NLI Contradiction评估"""
        print("🚀 NLI-based Contradiction Detection Baseline Evaluation")
        print("=" * 60)
        
        # 加载数据
        train_samples, test_samples = self.load_data(max_train_samples, max_test_samples)
        if train_samples is None or test_samples is None:
            return None
        
        # 评估方法
        results = self.evaluate_method(
            train_samples, 
            test_samples, 
            "NLI-based Contradiction Detection",
            self.extract_nli_contradiction_features
        )
        
        # 保存结果
        output_file = self.save_results("NLI_based_Contradiction_Detection", results)
        
        # 额外的分析
        print("\n📊 Feature Analysis:")
        train_features = self.extract_nli_contradiction_features(train_samples)
        test_features = self.extract_nli_contradiction_features(test_samples)
        
        print(f"📈 Train contradiction - Range: [{train_features.min():.3f}, {train_features.max():.3f}], Mean: {train_features.mean():.3f}")
        print(f"📈 Test contradiction - Range: [{test_features.min():.3f}, {test_features.max():.3f}], Mean: {test_features.mean():.3f}")
        
        # 按标签分析
        train_labels = self.extract_labels(train_samples)
        test_labels = self.extract_labels(test_samples)
        
        train_correct_features = train_features[train_labels == 0]
        train_halluc_features = train_features[train_labels == 1]
        test_correct_features = test_features[test_labels == 0]
        test_halluc_features = test_features[test_labels == 1]
        
        print(f"🔍 Train - Correct: {train_correct_features.mean():.3f}±{train_correct_features.std():.3f}")
        print(f"🔍 Train - Hallucination: {train_halluc_features.mean():.3f}±{train_halluc_features.std():.3f}")
        print(f"🔍 Test - Correct: {test_correct_features.mean():.3f}±{test_correct_features.std():.3f}")
        print(f"🔍 Test - Hallucination: {test_halluc_features.mean():.3f}±{test_halluc_features.std():.3f}")
        
        # 显示一些样本分析
        print("\n🔍 Sample Analysis:")
        for i, sample in enumerate(train_samples[:3]):
            question = sample.get('question', '')
            answer = sample.get('answer', '')
            label = "Correct" if train_labels[i] == 0 else "Hallucination"
            contradiction_score = train_features[i]
            
            print(f"  Sample {i+1} ({label}):")
            print(f"    Question: {question[:80]}...")
            print(f"    Answer: {answer[:80]}...")
            print(f"    Contradiction Score: {contradiction_score:.3f}")
            print()
        
        return results

def main():
    parser = argparse.ArgumentParser(description='NLI-based Contradiction Detection Baseline Evaluation')
    parser.add_argument('--train_samples', type=int, default=100, help='Number of training samples')
    parser.add_argument('--test_samples', type=int, default=50, help='Number of test samples')
    args = parser.parse_args()
    
    baseline = NLIContradictionBaseline()
    results = baseline.run_evaluation(args.train_samples, args.test_samples)
    
    if results:
        print(f"\n🎯 Final F1-Score: {results['f1_score']:.4f}")
        print(f"🎯 Final AUC: {results['auc']:.4f}")

if __name__ == "__main__":
    main()