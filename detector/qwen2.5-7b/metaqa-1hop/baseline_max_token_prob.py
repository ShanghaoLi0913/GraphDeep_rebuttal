#!/usr/bin/env python3
"""
Max Token Probability Baseline - 独立调试版本
基于模型对答案中最大token概率检测幻觉
"""

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from baseline_base import BaselineBase
import argparse
from tqdm import tqdm
import gc

class MaxTokenProbBaseline(BaselineBase):
    """Max Token Probability baseline检测器"""
    
    def __init__(self):
        super().__init__()
        print(f"📥 Loading model {self.model_name}...")
        
        # 加载tokenizer和model
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=False)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=False
        )
        self.model.eval()
        
        print("✅ Model loaded!")
    
    def calculate_max_token_probability(self, text):
        """计算答案文本的最大token概率 - 改进版本"""
        try:
            # 对答案文本进行概率计算
            inputs = self.tokenizer.encode(text, return_tensors="pt", max_length=self.max_length, truncation=True)
            inputs = inputs.to(self.device)
            
            with torch.no_grad():
                logits = self.model(inputs).logits[0]  # [seq_len, vocab_size]
                probs = torch.softmax(logits, dim=-1)
                
                # 计算每个token的实际概率（而不是位置最大概率）
                token_probs = []
                for i in range(1, len(inputs[0])):  # 跳过第一个token
                    if i < len(probs):
                        actual_token_id = inputs[0][i].item()
                        token_prob = probs[i-1, actual_token_id].cpu().item()
                        token_probs.append(token_prob)
                
                if token_probs:
                    # 使用真实token概率的最大值
                    max_token_prob = max(token_probs)
                else:
                    # 回退到原方法
                    max_probs = torch.max(probs, dim=-1)[0]
                    max_token_prob = torch.max(max_probs).cpu().item()
            
            # 清理内存
            del inputs, logits, probs
            
            # 移除人工范围限制，让模型自然区分
            return max(0.001, min(1.0, max_token_prob))
            
        except Exception as e:
            print(f"⚠️ Error calculating max token probability: {e}")
            return 0.5
    
    def extract_max_token_prob_features(self, samples):
        """提取Max Token Probability特征"""
        features = []
        
        print("🔍 Calculating max token probability...")
        for sample in tqdm(samples, desc="Max Token Prob"):
            answer = sample.get('answer', '').strip()
            
            if not answer:
                features.append(0.7)  # 默认值
                continue
            
            max_prob = self.calculate_max_token_probability(answer)
            # 🔧 反转逻辑：高概率可能表示模型过度自信的错误答案
            # 低概率可能表示模型对正确答案的合理不确定性
            inverted_prob = 1.0 - max_prob
            features.append(inverted_prob)
            
            # 定期清理GPU内存
            if len(features) % 10 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
        
        return np.array(features)
    
    def run_evaluation(self, max_train_samples=100, max_test_samples=50):
        """运行Max Token Probability评估"""
        print("🚀 Max Token Probability Baseline Evaluation")
        print("=" * 50)
        
        # 加载数据
        train_samples, test_samples = self.load_data(max_train_samples, max_test_samples)
        if train_samples is None or test_samples is None:
            return None
        
        # 评估方法
        results = self.evaluate_method(
            train_samples, 
            test_samples, 
            "Max Token Probability",
            self.extract_max_token_prob_features
        )
        
        # 保存结果
        output_file = self.save_results("Max_Token_Probability", results)
        
        # 额外的分析
        print("\n📊 Feature Analysis:")
        train_features = self.extract_max_token_prob_features(train_samples)
        test_features = self.extract_max_token_prob_features(test_samples)
        
        print(f"📈 Train max prob - Range: [{train_features.min():.3f}, {train_features.max():.3f}], Mean: {train_features.mean():.3f}")
        print(f"📈 Test max prob - Range: [{test_features.min():.3f}, {test_features.max():.3f}], Mean: {test_features.mean():.3f}")
        
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
    parser = argparse.ArgumentParser(description='Max Token Probability Baseline Evaluation')
    parser.add_argument('--train_samples', type=int, default=100, help='Number of training samples')
    parser.add_argument('--test_samples', type=int, default=50, help='Number of test samples')
    args = parser.parse_args()
    
    baseline = MaxTokenProbBaseline()
    results = baseline.run_evaluation(args.train_samples, args.test_samples)
    
    if results:
        print(f"\n🎯 Final F1-Score: {results['f1_score']:.4f}")
        print(f"🎯 Final AUC: {results['auc']:.4f}")

if __name__ == "__main__":
    main()