#!/usr/bin/env python3
"""
Token Confidence Baseline - 独立调试版本
基于模型对答案token的置信度检测幻觉
"""

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from baseline_base import BaselineBase
import argparse
from tqdm import tqdm
import gc

class TokenConfidenceBaseline(BaselineBase):
    """Token Confidence baseline检测器"""
    
    def __init__(self):
        super().__init__()
        print(f"📥 Loading model {self.model_name}...")
        
        # 加载tokenizer和model
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=False)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 🚀 Colab L4 GPU优化配置
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=False,
            low_cpu_mem_usage=True,  # 减少CPU内存使用
            use_cache=False  # 禁用KV缓存节省显存
        )
        self.model.eval()
        
        print("✅ Model loaded!")
    
    def calculate_token_confidence(self, text):
        """计算真正的token置信度 - 每个实际token的预测概率"""
        try:
            inputs = self.tokenizer.encode(text, return_tensors="pt", max_length=self.max_length, truncation=True)
            inputs = inputs.to(self.device)
            
            with torch.no_grad():
                logits = self.model(inputs).logits[0]  # [seq_len, vocab_size]
                probs = torch.softmax(logits, dim=-1)
                
                # 计算每个实际token的预测概率
                token_probs = []
                for i in range(1, len(inputs[0])):  # 跳过第一个token
                    if i < len(probs):
                        actual_token_id = inputs[0][i].item()
                        token_prob = probs[i-1, actual_token_id].cpu().item()
                        token_probs.append(token_prob)
                
                if token_probs:
                    # 使用实际token概率的平均值作为置信度
                    confidence = np.mean(token_probs)
                else:
                    confidence = 0.5
            
            # 清理内存
            del inputs, logits, probs
            
            # 只处理异常值，保持真实性
            return max(0.001, min(1.0, confidence))
            
        except Exception as e:
            print(f"⚠️ Error calculating token confidence: {e}")
            return 0.5
    
    def extract_token_confidence_features(self, samples):
        """提取Token Confidence特征 - 批处理优化版"""
        features = []
        
        print("🔍 Calculating token confidence...")
        
        # 🚀 批处理优化：按batch_size分组处理
        batch_size = min(self.batch_size, 16)  # Token confidence提升batch限制
        
        for i in tqdm(range(0, len(samples), batch_size), desc="Token Confidence Batches"):
            batch_samples = samples[i:i+batch_size]
            batch_features = []
            
            for sample in batch_samples:
                answer = sample.get('answer', '').strip()
                
                if not answer:
                    batch_features.append(0.5)
                    continue
                
                confidence = self.calculate_token_confidence(answer)
                inverted_confidence = 1.0 - confidence
                batch_features.append(inverted_confidence)
            
            features.extend(batch_features)
            
            # 每个batch后清理GPU内存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
        
        return np.array(features)
    
    def run_evaluation(self, max_train_samples=100, max_test_samples=50):
        """运行Token Confidence评估"""
        print("🚀 Token Confidence Baseline Evaluation")
        print("=" * 50)
        
        # 加载数据
        train_samples, test_samples = self.load_data(max_train_samples, max_test_samples)
        if train_samples is None or test_samples is None:
            return None
        
        # 评估方法
        results = self.evaluate_method(
            train_samples, 
            test_samples, 
            "Token Confidence",
            self.extract_token_confidence_features
        )
        
        # 保存结果
        output_file = self.save_results("Token_Confidence", results)
        
        # 额外的分析
        print("\n📊 Feature Analysis:")
        train_features = self.extract_token_confidence_features(train_samples)
        test_features = self.extract_token_confidence_features(test_samples)
        
        print(f"📈 Train confidence - Range: [{train_features.min():.3f}, {train_features.max():.3f}], Mean: {train_features.mean():.3f}")
        print(f"📈 Test confidence - Range: [{test_features.min():.3f}, {test_features.max():.3f}], Mean: {test_features.mean():.3f}")
        
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
    parser = argparse.ArgumentParser(description='Token Confidence Baseline Evaluation')
    parser.add_argument('--train_samples', type=int, default=100, help='Number of training samples')
    parser.add_argument('--test_samples', type=int, default=50, help='Number of test samples')
    args = parser.parse_args()
    
    baseline = TokenConfidenceBaseline()
    results = baseline.run_evaluation(args.train_samples, args.test_samples)
    
    if results:
        print(f"\n🎯 Final F1-Score: {results['f1_score']:.4f}")
        print(f"🎯 Final AUC: {results['auc']:.4f}")

if __name__ == "__main__":
    main()