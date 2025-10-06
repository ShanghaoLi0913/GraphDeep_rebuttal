#!/usr/bin/env python3
"""
Embedding-Based Semantic Divergence Baseline - 独立调试版本
基于问题和答案在语义空间中的偏离程度检测幻觉
"""

import numpy as np
from sentence_transformers import SentenceTransformer
from baseline_base import BaselineBase
import argparse
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import jensenshannon
import torch

class EmbeddingSemanticDivergenceBaseline(BaselineBase):
    """Embedding-Based Semantic Divergence baseline检测器"""
    
    def __init__(self):
        super().__init__()
        print("📥 Loading SentenceTransformer model for semantic divergence...")
        # 🚀 使用更快的模型和GPU优化
        self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2', device=self.device)
        if torch.cuda.is_available():
            print(f"🚀 SentenceTransformer loaded on {self.device}")
        print("✅ SentenceTransformer loaded!")
    
    def calculate_semantic_divergence(self, question, answer):
        """计算问题和答案之间的语义偏离度"""
        if not question or not answer:
            return 0.5  # 默认中等偏离度
        
        try:
            # 获取问题和答案的语义向量
            question_embedding = self.sentence_model.encode([question])
            answer_embedding = self.sentence_model.encode([answer])
            
            # 方法1: 余弦距离 (1 - 余弦相似度)
            cosine_sim = cosine_similarity(question_embedding, answer_embedding)[0][0]
            cosine_divergence = 1.0 - cosine_sim
            
            # 方法2: 欧几里得距离归一化
            euclidean_dist = np.linalg.norm(question_embedding[0] - answer_embedding[0])
            # 归一化到[0,1]，假设最大距离约为2*sqrt(384) ≈ 39.2 (对于384维向量)
            max_possible_dist = 2 * np.sqrt(question_embedding[0].shape[0])
            euclidean_divergence = min(euclidean_dist / max_possible_dist, 1.0)
            
            # 方法3: 向量角度偏离
            # 计算两个向量之间的角度(弧度)，然后归一化到[0,1]
            dot_product = np.dot(question_embedding[0], answer_embedding[0])
            norms = np.linalg.norm(question_embedding[0]) * np.linalg.norm(answer_embedding[0])
            cos_angle = np.clip(dot_product / norms, -1.0, 1.0)
            angle_radians = np.arccos(cos_angle)
            angle_divergence = angle_radians / np.pi  # 归一化到[0,1]
            
            # 方法4: 基于概率分布的Jensen-Shannon散度
            # 将embedding转换为概率分布（通过softmax）
            q_probs = torch.softmax(torch.tensor(question_embedding[0]), dim=0).numpy()
            a_probs = torch.softmax(torch.tensor(answer_embedding[0]), dim=0).numpy()
            js_divergence = jensenshannon(q_probs, a_probs)
            
            # 综合多种偏离度度量，给不同方法不同权重
            combined_divergence = (
                0.4 * cosine_divergence +      # 余弦距离权重最高
                0.2 * euclidean_divergence +   # 欧几里得距离
                0.2 * angle_divergence +       # 角度偏离
                0.2 * js_divergence           # JS散度
            )
            
            return max(0.0, min(1.0, combined_divergence))
            
        except Exception as e:
            print(f"⚠️ Error calculating semantic divergence: {e}")
            return 0.5
    
    def extract_semantic_divergence_features(self, samples):
        """提取Embedding-Based Semantic Divergence特征"""
        features = []
        
        print("🔍 Calculating embedding-based semantic divergence...")
        for sample in tqdm(samples, desc="Semantic Divergence"):
            question = sample.get('question', '')
            answer = sample.get('answer', '')
            
            divergence = self.calculate_semantic_divergence(question, answer)
            # 🔧 直接使用divergence：高偏离度表示幻觉，低偏离度表示正确
            # divergence越大 = 语义差异越大 = 更可能是幻觉
            features.append(divergence)
            
            # 定期清理内存
            if len(features) % 20 == 0:
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        return np.array(features)
    
    def run_evaluation(self, max_train_samples=100, max_test_samples=50):
        """运行Embedding-Based Semantic Divergence评估"""
        print("🚀 Embedding-Based Semantic Divergence Baseline Evaluation")
        print("=" * 60)
        
        # 加载数据
        train_samples, test_samples = self.load_data(max_train_samples, max_test_samples)
        if train_samples is None or test_samples is None:
            return None
        
        # 评估方法
        results = self.evaluate_method(
            train_samples, 
            test_samples, 
            "Embedding-Based Semantic Divergence",
            self.extract_semantic_divergence_features
        )
        
        # 保存结果
        output_file = self.save_results("Embedding_Based_Semantic_Divergence", results)
        
        # 额外的分析
        print("\n📊 Feature Analysis:")
        train_features = self.extract_semantic_divergence_features(train_samples)
        test_features = self.extract_semantic_divergence_features(test_samples)
        
        print(f"📈 Train divergence - Range: [{train_features.min():.3f}, {train_features.max():.3f}], Mean: {train_features.mean():.3f}")
        print(f"📈 Test divergence - Range: [{test_features.min():.3f}, {test_features.max():.3f}], Mean: {test_features.mean():.3f}")
        
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
        
        # 显示方法组成分析
        print("\n🔍 Divergence Method Analysis:")
        sample_q = test_samples[0].get('question', '') if test_samples else "What is AI?"
        sample_a = test_samples[0].get('answer', '') if test_samples else "Artificial Intelligence"
        
        if sample_q and sample_a:
            try:
                q_emb = self.sentence_model.encode([sample_q])
                a_emb = self.sentence_model.encode([sample_a])
                
                cosine_sim = cosine_similarity(q_emb, a_emb)[0][0]
                cosine_div = 1.0 - cosine_sim
                
                euclidean_dist = np.linalg.norm(q_emb[0] - a_emb[0])
                max_dist = 2 * np.sqrt(q_emb[0].shape[0])
                euclidean_div = min(euclidean_dist / max_dist, 1.0)
                
                print(f"  Sample question: {sample_q[:60]}...")
                print(f"  Sample answer: {sample_a[:60]}...")
                print(f"  Cosine divergence: {cosine_div:.3f}")
                print(f"  Euclidean divergence: {euclidean_div:.3f}")
                print(f"  Combined divergence: {self.calculate_semantic_divergence(sample_q, sample_a):.3f}")
                
            except Exception as e:
                print(f"  Analysis error: {e}")
        
        return results

def main():
    parser = argparse.ArgumentParser(description='Embedding-Based Semantic Divergence Baseline Evaluation')
    parser.add_argument('--train_samples', type=int, default=100, help='Number of training samples')
    parser.add_argument('--test_samples', type=int, default=50, help='Number of test samples')
    args = parser.parse_args()
    
    baseline = EmbeddingSemanticDivergenceBaseline()
    results = baseline.run_evaluation(args.train_samples, args.test_samples)
    
    if results:
        print(f"\n🎯 Final F1-Score: {results['f1_score']:.4f}")
        print(f"🎯 Final AUC: {results['auc']:.4f}")

if __name__ == "__main__":
    main()