#!/usr/bin/env python3
"""
BERTScore Baseline - 独立调试版本
基于回答与问题的语义相似度检测幻觉
"""

import numpy as np
from sentence_transformers import SentenceTransformer
from baseline_base import BaselineBase
import argparse
from tqdm import tqdm

class BERTScoreBaseline(BaselineBase):
    """BERTScore相似度baseline检测器"""
    
    def __init__(self):
        super().__init__()
        print("📥 Loading BERTScore model...")
        try:
            from bert_score import score
            print("✅ BERTScore library loaded!")
            self.use_real_bertscore = True
        except ImportError:
            print("⚠️ bert_score not available, using SentenceTransformer similarity")
            self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
            self.use_real_bertscore = False
        print("✅ BERTScore model ready!")
    
    def extract_bertscore_features(self, samples):
        """提取BERTScore特征"""
        features = []
        
        if self.use_real_bertscore:
            print("🔍 Calculating real BERTScore F1...")
            try:
                from bert_score import score
                questions = [sample.get('question', '') for sample in samples]
                answers = [sample.get('answer', '') for sample in samples]
                
                # 过滤空文本
                valid_pairs = [(q, a) for q, a in zip(questions, answers) if q.strip() and a.strip()]
                if not valid_pairs:
                    return np.array([0.0] * len(samples))
                
                valid_questions, valid_answers = zip(*valid_pairs)
                
                # 计算真正的BERTScore (candidate, reference)
                P, R, F1 = score(list(valid_answers), list(valid_questions), 
                                lang="en", verbose=False, device=self.device)
                
                # 将结果映射回原始样本
                bert_scores = F1.cpu().numpy().tolist()
                
                # 处理无效样本
                result_idx = 0
                for sample in samples:
                    q, a = sample.get('question', ''), sample.get('answer', '')
                    if q.strip() and a.strip():
                        features.append(bert_scores[result_idx])
                        result_idx += 1
                    else:
                        features.append(0.0)
                        
                print(f"✅ Calculated BERTScore for {len(valid_pairs)} valid pairs")
                
            except Exception as e:
                print(f"⚠️ Error with real BERTScore: {e}, falling back to similarity")
                self.use_real_bertscore = False
                
        if not self.use_real_bertscore:
            print("🔍 Calculating sentence similarity (fallback)...")
            for sample in tqdm(samples, desc="Sentence Similarity"):
                question = sample.get('question', '')
                answer = sample.get('answer', '')
                
                if not question or not answer:
                    features.append(0.0)
                    continue
                
                try:
                    # 计算问题和答案的向量表示
                    question_embedding = self.sentence_model.encode([question])
                    answer_embedding = self.sentence_model.encode([answer])
                    
                    # 计算余弦相似度
                    similarity = np.dot(question_embedding[0], answer_embedding[0]) / (
                        np.linalg.norm(question_embedding[0]) * np.linalg.norm(answer_embedding[0])
                    )
                    
                    features.append(float(similarity))
                    
                except Exception as e:
                    print(f"⚠️ Error calculating similarity for sample: {e}")
                    features.append(0.0)
        
        return np.array(features)
    
    def run_evaluation(self, max_train_samples=100, max_test_samples=50):
        """运行BERTScore评估"""
        print("🚀 BERTScore Baseline Evaluation")
        print("=" * 50)
        
        # 加载数据
        train_samples, test_samples = self.load_data(max_train_samples, max_test_samples)
        if train_samples is None or test_samples is None:
            return None
        
        # 评估方法
        results = self.evaluate_method(
            train_samples, 
            test_samples, 
            "BERTScore vs Question",
            self.extract_bertscore_features
        )
        
        # 保存结果
        output_file = self.save_results("BERTScore_vs_Question", results)
        
        # 额外的分析
        print("\n📊 Feature Analysis:")
        train_features = self.extract_bertscore_features(train_samples)
        test_features = self.extract_bertscore_features(test_samples)
        
        print(f"📈 Train features - Range: [{train_features.min():.3f}, {train_features.max():.3f}], Mean: {train_features.mean():.3f}")
        print(f"📈 Test features - Range: [{test_features.min():.3f}, {test_features.max():.3f}], Mean: {test_features.mean():.3f}")
        
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
        
        return results

def main():
    parser = argparse.ArgumentParser(description='BERTScore Baseline Evaluation')
    parser.add_argument('--train_samples', type=int, default=100, help='Number of training samples')
    parser.add_argument('--test_samples', type=int, default=50, help='Number of test samples')
    args = parser.parse_args()
    
    baseline = BERTScoreBaseline()
    results = baseline.run_evaluation(args.train_samples, args.test_samples)
    
    if results:
        print(f"\n🎯 Final F1-Score: {results['f1_score']:.4f}")
        print(f"🎯 Final AUC: {results['auc']:.4f}")

if __name__ == "__main__":
    main()