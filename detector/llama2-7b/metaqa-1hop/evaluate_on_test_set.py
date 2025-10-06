"""
在真正的test set上评估训练好的detector模型
获得真实、无偏的性能评估
"""

import json
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
import joblib
import glob

class TestSetEvaluator:
    def __init__(self):
        self.results = {}
        
    def load_data(self, train_file, test_file):
        """加载训练和测试数据"""
        print("Loading data...")
        
        # 加载训练数据
        train_data = []
        with open(train_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip() and not line.startswith('{"config"'):
                    train_data.append(json.loads(line))
        
        # 加载测试数据
        test_data = []
        with open(test_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip() and not line.startswith('{"config"'):
                    test_data.append(json.loads(line))
        
        print(f"Loaded {len(train_data)} training samples, {len(test_data)} test samples")
        return train_data, test_data
    
    def extract_enhanced_features(self, data):
        """提取增强特征 - 与训练时保持一致"""
        features = []
        labels = []
        
        for item in data:
            # 基础特征
            feature_dict = {
                'gass_score': item.get('gass_score', 0),
                'prd_score': item.get('prd_score', 0),
                'balanced_calibrated_gass': item.get('balanced_calibrated_gass', 0),
                'tus_score': item.get('tus_score', 0),
                'gass_jsd_score': item.get('gass_jsd_score', 0),
            }
            
            # 特征工程：组合特征
            feature_dict['gass_prd_ratio'] = feature_dict['gass_score'] / (feature_dict['prd_score'] + 1e-8)
            feature_dict['gass_tus_diff'] = feature_dict['gass_score'] - feature_dict['tus_score']
            feature_dict['balanced_original_gass_diff'] = feature_dict['balanced_calibrated_gass'] - feature_dict['gass_score']
            
            # 表面特征
            if 'model_output' in item:
                output = item['model_output']
                words = output.split()
                feature_dict.update({
                    'output_length': len(words),
                    'repetition_score': self.calculate_repetition(output),
                    'avg_word_length': np.mean([len(w) for w in words]) if words else 0,
                    'unique_word_ratio': len(set(words)) / len(words) if words else 0,
                    'has_ans_prefix': 1 if 'ans:' in output.lower() else 0,
                    'comma_count': output.count(','),
                    'question_mark_count': output.count('?'),
                })
            
            # 问题特征
            if 'question' in item:
                question = item['question']
                q_words = question.split()
                feature_dict.update({
                    'question_length': len(q_words),
                    'question_has_what': 1 if 'what' in question.lower() else 0,
                    'question_has_which': 1 if 'which' in question.lower() else 0,
                    'question_has_who': 1 if 'who' in question.lower() else 0,
                })
            
            features.append(feature_dict)
            
            # 标签
            if 'squad_evaluation' in item:
                is_hallucination = item['squad_evaluation'].get('squad_is_hallucination', False)
            else:
                is_hallucination = not item.get('metrics', {}).get('hit@1', False)
            
            labels.append(int(is_hallucination))
        
        df = pd.DataFrame(features)
        return df, np.array(labels)
    
    def calculate_repetition(self, text):
        """计算文本重复度"""
        words = text.lower().split()
        if len(words) <= 1:
            return 0
        unique_words = len(set(words))
        return 1 - (unique_words / len(words))
    
    def load_trained_models(self, model_dir):
        """加载训练好的模型"""
        print("🔄 加载训练好的模型...")
        
        # 查找最新的模型元数据文件
        metadata_files = glob.glob(f"{model_dir}/models_metadata_*.json")
        if not metadata_files:
            print("❌ 没有找到保存的模型元数据文件")
            print("请先运行 train_detector_optimized.py 训练模型")
            return None
        
        # 使用最新的模型文件
        latest_metadata_file = max(metadata_files)
        print(f"📁 使用模型元数据文件: {latest_metadata_file}")
        
        # 加载元数据
        with open(latest_metadata_file, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        # 加载scaler
        scaler_path = metadata['scaler_path']
        if not os.path.exists(scaler_path):
            print(f"❌ Scaler文件不存在: {scaler_path}")
            return None
        
        scaler = joblib.load(scaler_path)
        print(f"✅ 加载scaler: {scaler_path}")
        
        # 加载特征名称
        feature_names_path = metadata['feature_names_path']
        if not os.path.exists(feature_names_path):
            print(f"❌ 特征名称文件不存在: {feature_names_path}")
            return None
        
        with open(feature_names_path, 'r', encoding='utf-8') as f:
            feature_names = json.load(f)
        print(f"✅ 加载特征名称: {len(feature_names)} 个特征")
        
        # 加载优化阈值（如果存在）
        thresholds = {}
        if 'threshold_path' in metadata and os.path.exists(metadata['threshold_path']):
            with open(metadata['threshold_path'], 'r', encoding='utf-8') as f:
                thresholds = json.load(f)
            print(f"✅ 加载优化阈值: {len(thresholds)} 个模型")
        else:
            print("⚠️ 未找到优化阈值，将使用默认阈值0.5")
        
        # 加载所有模型
        models = {}
        model_paths = metadata['model_paths']
        
        for model_name, model_path in model_paths.items():
            if not os.path.exists(model_path):
                print(f"❌ 模型文件不存在: {model_path}")
                continue
            
            try:
                model = joblib.load(model_path)
                models[model_name] = model
                print(f"✅ 加载模型: {model_name}")
            except Exception as e:
                print(f"❌ 加载模型 {model_name} 失败: {e}")
                continue
        
        if not models:
            print("❌ 没有成功加载任何模型")
            return None
        
        print(f"🎉 成功加载 {len(models)} 个模型")
        print(f"📈 训练时最佳模型: {metadata['best_model']} (AUC: {metadata['best_auc']:.4f})")
        
        return {
            'models': models,
            'scaler': scaler,
            'feature_names': feature_names,
            'thresholds': thresholds,
            'metadata': metadata
        }
    
    def evaluate_on_test_set(self, test_data, model_dir):
        """在真正的test set上评估"""
        print("\n🧪 在真正的test set上评估模型性能...")
        
        # 加载训练好的模型
        model_data = self.load_trained_models(model_dir)
        if model_data is None:
            return None
        
        trained_models = model_data['models']
        scaler = model_data['scaler']
        feature_names = model_data['feature_names']
        thresholds = model_data['thresholds']
        metadata = model_data['metadata']
        
        # 提取测试集特征
        X_test, y_test = self.extract_enhanced_features(test_data)
        
        print(f"Test set: {len(X_test)} samples")
        print(f"Test set hallucination rate: {np.mean(y_test):.3f}")
        
        # 确保特征顺序一致
        if list(X_test.columns) != feature_names:
            print("⚠️ 特征顺序不一致，重新排序...")
            X_test = X_test[feature_names]
        
        # 使用训练时的scaler进行标准化
        X_test_scaled = scaler.transform(X_test)
        
        # 在test set上评估
        print("\n📊 在Test Set上的真实性能:")
        print("="*60)
        
        test_results = {}
        for name, model in trained_models.items():
            try:
                y_proba = model.predict_proba(X_test_scaled)[:, 1]
                auc = roc_auc_score(y_test, y_proba)
                
                # 使用优化阈值（如果存在），否则使用默认阈值
                optimal_threshold = thresholds.get(name, 0.5)
                
                # 默认阈值预测
                y_pred_default = model.predict(X_test_scaled)
                report_default = classification_report(y_test, y_pred_default, output_dict=True)
                
                # 优化阈值预测
                y_pred_optimal = (y_proba >= optimal_threshold).astype(int)
                report_optimal = classification_report(y_test, y_pred_optimal, output_dict=True)
                
                test_results[name] = {
                    'auc': auc,
                    'optimal_threshold': optimal_threshold,
                    'default_results': {
                        'precision': report_default['1']['precision'] if '1' in report_default else 0,
                        'recall': report_default['1']['recall'] if '1' in report_default else 0,
                        'f1_score': report_default['1']['f1-score'] if '1' in report_default else 0,
                        'accuracy': report_default['accuracy']
                    },
                    'optimal_results': {
                        'precision': report_optimal['1']['precision'] if '1' in report_optimal else 0,
                        'recall': report_optimal['1']['recall'] if '1' in report_optimal else 0,
                        'f1_score': report_optimal['1']['f1-score'] if '1' in report_optimal else 0,
                        'accuracy': report_optimal['accuracy']
                    }
                }
                
                print(f"\n{name}:")
                print(f"  AUC: {auc:.4f}")
                print(f"  优化阈值: {optimal_threshold:.3f}")
                print(f"  默认阈值(0.5) - P: {report_default['1']['precision']:.3f}, R: {report_default['1']['recall']:.3f}, F1: {report_default['1']['f1-score']:.3f}")
                print(f"  优化阈值({optimal_threshold:.3f}) - P: {report_optimal['1']['precision']:.3f}, R: {report_optimal['1']['recall']:.3f}, F1: {report_optimal['1']['f1-score']:.3f}")
                
            except Exception as e:
                print(f"❌ 评估模型 {name} 时出错: {e}")
                continue
        
        return test_results
    
    def plot_test_results(self, test_results, output_dir):
        """绘制test set结果"""
        plt.figure(figsize=(12, 8))
        
        # AUC对比
        plt.subplot(2, 2, 1)
        names = list(test_results.keys())
        aucs = [test_results[name]['auc'] for name in names]
        colors = ['lightcoral' if name == 'VotingEnsemble' else 'skyblue' for name in names]
        bars = plt.bar(names, aucs, color=colors)
        plt.ylabel('AUC Score')
        plt.title('Test Set AUC Performance')
        plt.xticks(rotation=45)
        plt.ylim(0, 1)
        
        for bar, auc in zip(bars, aucs):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{auc:.3f}', ha='center', va='bottom')
        
        # F1对比 (使用优化阈值结果)
        plt.subplot(2, 2, 2)
        f1s = [test_results[name]['optimal_results']['f1_score'] for name in names]
        bars = plt.bar(names, f1s, color=colors)
        plt.ylabel('F1 Score')
        plt.title('Test Set F1 Performance (优化阈值)')
        plt.xticks(rotation=45)
        plt.ylim(0, 1)
        
        for bar, f1 in zip(bars, f1s):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{f1:.3f}', ha='center', va='bottom')
        
        # Precision vs Recall (使用优化阈值结果)
        plt.subplot(2, 2, 3)
        precisions = [test_results[name]['optimal_results']['precision'] for name in names]
        recalls = [test_results[name]['optimal_results']['recall'] for name in names]
        
        plt.scatter(recalls, precisions, s=100, alpha=0.7)
        for i, name in enumerate(names):
            plt.annotate(name, (recalls[i], precisions[i]), 
                        xytext=(5, 5), textcoords='offset points')
        
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision vs Recall (Test Set)')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存图片
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.savefig(f"{output_dir}/test_set_evaluation_{timestamp}.png", dpi=300, bbox_inches='tight')
        plt.show()

def main():
    """主函数"""
    # 文件路径 - 使用llama2-7b测试数据，从llama2-7b模型目录加载模型
    test_file = "/mnt/d/experiments/GraphDeEP/experiment_records/inference_results/llama2-7b/colab_test_simple.jsonl"
    output_dir = "/mnt/d/experiments/GraphDeEP/detector/llama2-7b/metaqa-1hop/results"
    model_dir = "/mnt/d/experiments/GraphDeEP/detector/llama2-7b/metaqa-1hop/models"
    
    print("🎯 Test Set 真实性能评估")
    print("="*50)
    
    # 检查文件
    if not os.path.exists(test_file):
        print(f"❌ 测试文件不存在: {test_file}")
        return
    
    if not os.path.exists(model_dir):
        print(f"❌ 模型目录不存在: {model_dir}")
        print("请先运行 train_detector_optimized.py 训练模型")
        return
    
    # 初始化评估器
    evaluator = TestSetEvaluator()
    
    # 加载测试数据
    test_data = []
    with open(test_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip() and not line.startswith('{"config"'):
                test_data.append(json.loads(line))
    
    print(f"Loaded {len(test_data)} test samples")
    
    # 在test set上评估
    test_results = evaluator.evaluate_on_test_set(test_data, model_dir)
    
    if test_results is None:
        print("❌ 评估失败")
        return
    
    # 绘制结果
    evaluator.plot_test_results(test_results, output_dir)
    
    # 保存结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(f"{output_dir}/test_set_results_{timestamp}.json", 'w', encoding='utf-8') as f:
        json.dump(test_results, f, indent=2, ensure_ascii=False)
    
    # 总结
    print("\n" + "="*60)
    print("🏆 TEST SET 最终结果总结")
    print("="*60)
    
    best_model = max(test_results.keys(), key=lambda x: test_results[x]['auc'])
    best_auc = test_results[best_model]['auc']
    best_results = test_results[best_model]['optimal_results']
    
    print(f"最佳模型: {best_model}")
    print(f"Test Set AUC: {best_auc:.4f}")
    print(f"优化阈值结果 - P: {best_results['precision']:.4f}, R: {best_results['recall']:.4f}, F1: {best_results['f1_score']:.4f}")
    print("\n✅ 这是基于训练好的模型和优化阈值的真实、无偏的性能评估！")

if __name__ == "__main__":
    main()