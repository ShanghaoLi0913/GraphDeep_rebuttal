"""
平衡采样校准：解决极端数据不平衡问题
用平衡的正确/幻觉样本重新训练校准器

核心思想：
1. 从大量正确样本中随机采样，与幻觉样本数量匹配
2. 用平衡数据训练校准器
3. 应用到全部数据上

作者: AI Assistant
日期: 2025年7月7日
"""

import json
import numpy as np
import random
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import joblib
import os
from datetime import datetime

class BalancedCalibrator:
    """平衡采样校准器"""
    
    def __init__(self, balance_ratio=1.0, random_seed=42):
        """
        Args:
            balance_ratio: 正确:幻觉的比例，1.0表示1:1平衡
            random_seed: 随机种子
        """
        self.balance_ratio = balance_ratio
        self.random_seed = random_seed
        self.scaler = StandardScaler()
        self.classifier = LogisticRegression(random_state=random_seed, max_iter=1000)
        self.is_trained = False
        
    def extract_training_data(self, results_file):
        """从结果文件中提取训练数据"""
        correct_scores = []
        hallucinated_scores = []
        
        print(f"📖 Loading data from: {results_file}")
        
        with open(results_file, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                    
                try:
                    data = json.loads(line)
                    if 'config' in data or 'stats' in data:
                        continue
                    
                    gass_score = data.get('gass_score', 0.0)
                    
                    # 判断是否为幻觉（优先使用SQuAD评估）
                    squad_eval = data.get('squad_evaluation')
                    if squad_eval is not None:
                        is_correct = not squad_eval.get('squad_is_hallucination', True)
                    else:
                        is_correct = data.get('metrics', {}).get('hit@1', False)
                    
                    if is_correct:
                        correct_scores.append(gass_score)
                    else:
                        hallucinated_scores.append(gass_score)
                        
                except Exception as e:
                    continue
        
        print(f"📊 Original data: {len(correct_scores)} correct, {len(hallucinated_scores)} hallucinated")
        return correct_scores, hallucinated_scores
    
    def create_balanced_dataset(self, correct_scores, hallucinated_scores):
        """创建平衡数据集"""
        random.seed(self.random_seed)
        
        num_hallucinated = len(hallucinated_scores)
        num_correct_needed = int(num_hallucinated * self.balance_ratio)
        
        if len(correct_scores) < num_correct_needed:
            print(f"⚠️ Warning: Not enough correct samples ({len(correct_scores)}) for desired ratio")
            num_correct_needed = len(correct_scores)
        
        # 随机采样正确样本
        sampled_correct = random.sample(correct_scores, num_correct_needed)
        
        # 组合数据
        X = np.array(sampled_correct + hallucinated_scores).reshape(-1, 1)
        y = np.array([1] * len(sampled_correct) + [0] * len(hallucinated_scores))
        
        print(f"🎯 Balanced dataset: {len(sampled_correct)} correct, {len(hallucinated_scores)} hallucinated")
        print(f"📈 Balance ratio: {len(sampled_correct)/len(hallucinated_scores):.2f}:1")
        
        return X, y
    
    def train(self, results_file):
        """训练校准器"""
        print("🚀 Starting balanced calibration training...")
        
        # 提取数据
        correct_scores, hallucinated_scores = self.extract_training_data(results_file)
        
        if len(hallucinated_scores) == 0:
            raise ValueError("No hallucinated samples found for training!")
        
        # 创建平衡数据集
        X, y = self.create_balanced_dataset(correct_scores, hallucinated_scores)
        
        # 标准化特征
        X_scaled = self.scaler.fit_transform(X)
        
        # 训练分类器
        print("🎓 Training logistic regression classifier...")
        self.classifier.fit(X_scaled, y)
        
        # 评估训练效果
        y_pred = self.classifier.predict(X_scaled)
        print("📊 Training performance:")
        print(classification_report(y, y_pred, target_names=['Hallucinated', 'Correct']))
        
        self.is_trained = True
        print("✅ Training completed!")
        
    def calibrate_score(self, original_gass_score):
        """校准单个GASS分数"""
        if not self.is_trained:
            raise ValueError("Calibrator not trained yet!")
        
        # 标准化输入
        X = np.array([[original_gass_score]])
        X_scaled = self.scaler.transform(X)
        
        # 获取正确类的概率
        prob_correct = self.classifier.predict_proba(X_scaled)[0][1]
        
        # 将概率转换为校准分数
        # 使用 logit 变换：score = log(p/(1-p))
        if prob_correct >= 0.999:
            prob_correct = 0.999
        elif prob_correct <= 0.001:
            prob_correct = 0.001
            
        calibrated_score = np.log(prob_correct / (1 - prob_correct))
        
        return float(calibrated_score)
    
    def save_model(self, model_dir):
        """保存训练好的模型"""
        if not self.is_trained:
            raise ValueError("No trained model to save!")
        
        os.makedirs(model_dir, exist_ok=True)
        
        # 保存scaler和classifier
        scaler_path = os.path.join(model_dir, 'balanced_scaler.joblib')
        classifier_path = os.path.join(model_dir, 'balanced_classifier.joblib')
        
        joblib.dump(self.scaler, scaler_path)
        joblib.dump(self.classifier, classifier_path)
        
        # 保存元数据
        metadata = {
            'balance_ratio': self.balance_ratio,
            'random_seed': self.random_seed,
            'training_time': datetime.now().isoformat(),
            'scaler_path': scaler_path,
            'classifier_path': classifier_path
        }
        
        metadata_path = os.path.join(model_dir, 'balanced_calibration_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"💾 Model saved to: {model_dir}")
        return model_dir
    
    def load_model(self, model_dir):
        """加载训练好的模型"""
        scaler_path = os.path.join(model_dir, 'balanced_scaler.joblib')
        classifier_path = os.path.join(model_dir, 'balanced_classifier.joblib')
        
        if not os.path.exists(scaler_path) or not os.path.exists(classifier_path):
            raise ValueError(f"Model files not found in {model_dir}")
        
        self.scaler = joblib.load(scaler_path)
        self.classifier = joblib.load(classifier_path)
        self.is_trained = True
        
        print(f"📂 Model loaded from: {model_dir}")

def main():
    """演示平衡校准"""
    # 训练数据文件
    training_file = "experiment_records/inference_results/Llama-2-7b/colab_dev_simple.jsonl"
    
    # 创建校准器
    calibrator = BalancedCalibrator(balance_ratio=1.0, random_seed=42)
    
    # 训练
    calibrator.train(training_file)
    
    # 保存模型
    model_dir = calibrator.save_model("calibration_models/balanced_calibration_llama3")
    
    # 测试几个样本
    test_scores = [0.15, 0.18, 0.20, 0.25]
    print("\\n🧪 Testing calibration:")
    for score in test_scores:
        calibrated = calibrator.calibrate_score(score)
        print(f"Original: {score:.4f} → Calibrated: {calibrated:.4f}")

if __name__ == "__main__":
    main()