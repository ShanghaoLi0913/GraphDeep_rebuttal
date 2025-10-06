#!/usr/bin/env python3
"""
Uncertainty Quantification Baseline - 独立调试版本
基于模型预测的不确定性检测幻觉
"""

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from baseline_base import BaselineBase
import argparse
from tqdm import tqdm
import gc
from scipy.stats import entropy

class UncertaintyQuantificationBaseline(BaselineBase):
    """Uncertainty Quantification baseline检测器"""
    
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
    
    def calculate_uncertainty(self, text):
        """计算单个文本的不确定性（基于熵）"""
        try:
            inputs = self.tokenizer.encode(text, return_tensors="pt", max_length=self.max_length, truncation=True)
            inputs = inputs.to(self.device)
            
            with torch.no_grad():
                logits = self.model(inputs).logits[0]  # [seq_len, vocab_size]
                probs = torch.softmax(logits, dim=-1)
                
                # 计算每个位置的熵（不确定性）
                entropies = []
                for pos_probs in probs:
                    pos_probs_np = pos_probs.cpu().numpy()
                    # 过滤掉0概率避免log(0)
                    pos_probs_np = pos_probs_np[pos_probs_np > 1e-10]
                    if len(pos_probs_np) > 0:
                        ent = entropy(pos_probs_np)
                        entropies.append(ent)
                
                # 计算平均不确定性
                if entropies:
                    avg_uncertainty = np.mean(entropies)
                else:
                    avg_uncertainty = 5.0  # 默认中等不确定性
            
            # 清理内存
            del inputs, logits, probs
            
            # 归一化到[0, 1]范围，熵的理论最大值约为log(vocab_size)
            max_entropy = np.log(self.tokenizer.vocab_size)
            normalized_uncertainty = min(avg_uncertainty / max_entropy, 1.0)
            
            # 🔧 移除人工噪声，保持计算纯净性
            return max(0.0, min(1.0, normalized_uncertainty))
            
        except Exception as e:
            print(f"⚠️ Error calculating uncertainty: {e}")
            return 0.3  # 固定默认值，无噪声
    
    def extract_uncertainty_features(self, samples):
        """提取Uncertainty Quantification特征"""
        features = []
        
        print("🔍 Calculating uncertainty scores...")
        for sample in tqdm(samples, desc="Uncertainty"):
            answer = sample.get('answer', '').strip()
            
            if not answer:
                features.append(0.3)  # 默认值
                continue
            
            uncertainty = self.calculate_uncertainty(answer)
            features.append(uncertainty)
            
            # 定期清理GPU内存
            if len(features) % 10 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
        
        return np.array(features)
    
    def run_evaluation(self, max_train_samples=100, max_test_samples=50):
        """运行Uncertainty Quantification评估"""
        print("🚀 Uncertainty Quantification Baseline Evaluation")
        print("=" * 55)
        
        # 加载数据
        train_samples, test_samples = self.load_data(max_train_samples, max_test_samples)
        if train_samples is None or test_samples is None:
            return None
        
        # 评估方法
        results = self.evaluate_method(
            train_samples, 
            test_samples, 
            "Uncertainty Quantification",
            self.extract_uncertainty_features
        )
        
        # 保存结果
        output_file = self.save_results("Uncertainty_Quantification", results)
        
        # 额外的分析
        print("\n📊 Feature Analysis:")
        train_features = self.extract_uncertainty_features(train_samples)
        test_features = self.extract_uncertainty_features(test_samples)
        
        print(f"📈 Train uncertainty - Range: [{train_features.min():.3f}, {train_features.max():.3f}], Mean: {train_features.mean():.3f}")
        print(f"📈 Test uncertainty - Range: [{test_features.min():.3f}, {test_features.max():.3f}], Mean: {test_features.mean():.3f}")
        
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
        
        # 理论期望：幻觉应该有更高的不确定性
        if len(train_halluc_features) > 0 and len(train_correct_features) > 0:
            halluc_mean = train_halluc_features.mean()
            correct_mean = train_correct_features.mean()
            if halluc_mean > correct_mean:
                print("✅ Expected pattern: Hallucinations have higher uncertainty")
            else:
                print("⚠️ Unexpected pattern: Correct answers have higher uncertainty")
        
        return results

def main():
    parser = argparse.ArgumentParser(description='Uncertainty Quantification Baseline Evaluation')
    parser.add_argument('--train_samples', type=int, default=100, help='Number of training samples')
    parser.add_argument('--test_samples', type=int, default=50, help='Number of test samples')
    args = parser.parse_args()
    
    baseline = UncertaintyQuantificationBaseline()
    results = baseline.run_evaluation(args.train_samples, args.test_samples)
    
    if results:
        print(f"\n🎯 Final F1-Score: {results['f1_score']:.4f}")
        print(f"🎯 Final AUC: {results['auc']:.4f}")

if __name__ == "__main__":
    main()