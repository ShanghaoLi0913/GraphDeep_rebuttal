#!/usr/bin/env python3
"""
Perplexity Baseline - 独立调试版本
基于模型对答案的困惑度检测幻觉
"""

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from baseline_base import BaselineBase
import argparse
from tqdm import tqdm
import gc

class PerplexityBaseline(BaselineBase):
    """Perplexity baseline检测器"""
    
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
    
    def calculate_perplexity(self, text):
        """计算单个文本的perplexity"""
        try:
            inputs = self.tokenizer.encode(text, return_tensors="pt", max_length=self.max_length, truncation=True)
            inputs = inputs.to(self.device)
            
            with torch.no_grad():
                outputs = self.model(inputs, labels=inputs)
                loss = outputs.loss
                perplexity = torch.exp(loss).cpu().item()
            
            # 清理内存
            del inputs, outputs
            
            # 处理异常值但保持真实性
            if perplexity == float('inf') or perplexity > 10000:
                perplexity = 10000.0  # 设为上限但不加噪声
            elif perplexity < 1.0:
                perplexity = 1.0  # 设为下限但不加噪声
            
            # 转换为log space增加稳定性
            return np.log(perplexity)
            
        except Exception as e:
            print(f"⚠️ Error calculating perplexity: {e}")
            return np.log(15.0)  # 默认中等perplexity的log值
    
    def extract_perplexity_features(self, samples):
        """提取Perplexity特征"""
        features = []
        
        print("🔍 Calculating perplexity...")
        for sample in tqdm(samples, desc="Perplexity"):
            answer = sample.get('answer', '').strip()
            
            if not answer:
                features.append(np.log(15.0))  # 默认值
                continue
            
            perplexity = self.calculate_perplexity(answer)
            features.append(perplexity)
            
            # 定期清理GPU内存
            if len(features) % 10 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
        
        return np.array(features)
    
    def run_evaluation(self, max_train_samples=100, max_test_samples=50):
        """运行Perplexity评估"""
        print("🚀 Perplexity Baseline Evaluation")
        print("=" * 50)
        
        # 加载数据
        train_samples, test_samples = self.load_data(max_train_samples, max_test_samples)
        if train_samples is None or test_samples is None:
            return None
        
        # 评估方法
        results = self.evaluate_method(
            train_samples, 
            test_samples, 
            "Perplexity",
            self.extract_perplexity_features
        )
        
        # 保存结果
        output_file = self.save_results("Perplexity", results)
        
        # 额外的分析
        print("\n📊 Feature Analysis:")
        train_features = self.extract_perplexity_features(train_samples)
        test_features = self.extract_perplexity_features(test_samples)
        
        print(f"📈 Train perplexity - Range: [{np.exp(train_features.min()):.3f}, {np.exp(train_features.max()):.3f}], Mean: {np.exp(train_features.mean()):.3f}")
        print(f"📈 Test perplexity - Range: [{np.exp(test_features.min()):.3f}, {np.exp(test_features.max()):.3f}], Mean: {np.exp(test_features.mean()):.3f}")
        
        # 按标签分析
        train_labels = self.extract_labels(train_samples)
        test_labels = self.extract_labels(test_samples)
        
        train_correct_features = train_features[train_labels == 0]
        train_halluc_features = train_features[train_labels == 1]
        test_correct_features = test_features[test_labels == 0]
        test_halluc_features = test_features[test_labels == 1]
        
        print(f"🔍 Train - Correct: {np.exp(train_correct_features.mean()):.3f}±{np.exp(train_correct_features.std()):.3f}")
        print(f"🔍 Train - Hallucination: {np.exp(train_halluc_features.mean()):.3f}±{np.exp(train_halluc_features.std()):.3f}")
        print(f"🔍 Test - Correct: {np.exp(test_correct_features.mean()):.3f}±{np.exp(test_correct_features.std()):.3f}")
        print(f"🔍 Test - Hallucination: {np.exp(test_halluc_features.mean()):.3f}±{np.exp(test_halluc_features.std()):.3f}")
        
        return results

def main():
    parser = argparse.ArgumentParser(description='Perplexity Baseline Evaluation')
    parser.add_argument('--train_samples', type=int, default=100, help='Number of training samples')
    parser.add_argument('--test_samples', type=int, default=50, help='Number of test samples')
    args = parser.parse_args()
    
    baseline = PerplexityBaseline()
    results = baseline.run_evaluation(args.train_samples, args.test_samples)
    
    if results:
        print(f"\n🎯 Final F1-Score: {results['f1_score']:.4f}")
        print(f"🎯 Final AUC: {results['auc']:.4f}")

if __name__ == "__main__":
    main()