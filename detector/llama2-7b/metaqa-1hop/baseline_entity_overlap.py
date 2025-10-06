#!/usr/bin/env python3
"""
Entity Overlap Baseline - 独立调试版本
基于答案与问题实体重叠度检测幻觉
"""

import numpy as np
import re
from baseline_base import BaselineBase
import argparse
from tqdm import tqdm

class EntityOverlapBaseline(BaselineBase):
    """Entity Overlap baseline检测器"""
    
    def __init__(self):
        super().__init__()
        print("✅ Entity Overlap detector initialized!")
    
    def extract_entities(self, text):
        """简单的实体提取：提取大写字母开头的词和引号内容"""
        if not text:
            return set()
        
        entities = set()
        
        # 提取引号内的内容
        quoted_entities = re.findall(r'"([^"]*)"', text)
        entities.update([e.lower().strip() for e in quoted_entities if e.strip()])
        
        # 提取大写字母开头的词（可能是专有名词/实体）
        capitalized_words = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
        entities.update([e.lower().strip() for e in capitalized_words])
        
        # 提取数字
        numbers = re.findall(r'\b\d+\b', text)
        entities.update(numbers)
        
        return entities
    
    def calculate_entity_overlap(self, question, answer):
        """计算问题和答案之间的实体重叠度"""
        if not question or not answer:
            return 0.5  # 默认中等重叠度
        
        question_entities = self.extract_entities(question)
        answer_entities = self.extract_entities(answer)
        
        if not question_entities and not answer_entities:
            return 0.5
        
        if not question_entities or not answer_entities:
            return 0.1  # 一方没有实体，重叠度很低
        
        # 计算Jaccard相似度
        intersection = len(question_entities.intersection(answer_entities))
        union = len(question_entities.union(answer_entities))
        
        if union == 0:
            return 0.5
        
        overlap = intersection / union
        
        # 返回真实的重叠度，不加任何人工修改
        return max(0.0, min(1.0, overlap))
    
    def extract_entity_overlap_features(self, samples):
        """提取Entity Overlap特征"""
        features = []
        
        print("🔍 Calculating entity overlap...")
        for sample in tqdm(samples, desc="Entity Overlap"):
            question = sample.get('question', '')
            answer = sample.get('answer', '')
            
            overlap = self.calculate_entity_overlap(question, answer)
            # 🔧 反转逻辑：高重叠可能表示幻觉（重复问题中的实体）
            # 低重叠可能表示正确（提供了不同的信息）
            inverted_overlap = 1.0 - overlap
            features.append(inverted_overlap)
        
        return np.array(features)
    
    def run_evaluation(self, max_train_samples=100, max_test_samples=50):
        """运行Entity Overlap评估"""
        print("🚀 Entity Overlap Baseline Evaluation")
        print("=" * 50)
        
        # 加载数据
        train_samples, test_samples = self.load_data(max_train_samples, max_test_samples)
        if train_samples is None or test_samples is None:
            return None
        
        # 评估方法
        results = self.evaluate_method(
            train_samples, 
            test_samples, 
            "Entity Question Overlap",
            self.extract_entity_overlap_features
        )
        
        # 保存结果
        output_file = self.save_results("Entity_Question_Overlap", results)
        
        # 额外的分析
        print("\n📊 Feature Analysis:")
        train_features = self.extract_entity_overlap_features(train_samples)
        test_features = self.extract_entity_overlap_features(test_samples)
        
        print(f"📈 Train overlap - Range: [{train_features.min():.3f}, {train_features.max():.3f}], Mean: {train_features.mean():.3f}")
        print(f"📈 Test overlap - Range: [{test_features.min():.3f}, {test_features.max():.3f}], Mean: {test_features.mean():.3f}")
        
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
            overlap = train_features[i]
            
            q_entities = self.extract_entities(question)
            a_entities = self.extract_entities(answer)
            
            print(f"  Sample {i+1} ({label}):")
            print(f"    Question: {question[:80]}...")
            print(f"    Answer: {answer[:80]}...")
            print(f"    Q-entities: {q_entities}")
            print(f"    A-entities: {a_entities}")
            print(f"    Overlap: {overlap:.3f}")
            print()
        
        return results

def main():
    parser = argparse.ArgumentParser(description='Entity Overlap Baseline Evaluation')
    parser.add_argument('--train_samples', type=int, default=100, help='Number of training samples')
    parser.add_argument('--test_samples', type=int, default=50, help='Number of test samples')
    args = parser.parse_args()
    
    baseline = EntityOverlapBaseline()
    results = baseline.run_evaluation(args.train_samples, args.test_samples)
    
    if results:
        print(f"\n🎯 Final F1-Score: {results['f1_score']:.4f}")
        print(f"🎯 Final AUC: {results['auc']:.4f}")

if __name__ == "__main__":
    main()