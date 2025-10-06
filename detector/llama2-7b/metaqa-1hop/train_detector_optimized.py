"""
优化版幻觉检测器训练脚本
- 超参数调优
- 特征工程增强
- 集成学习
- 类别不平衡处理
- 交叉验证
"""
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, precision_recall_curve
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline
import xgboost as xgb
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
import warnings
warnings.filterwarnings('ignore')

class OptimizedHallucinationDetector:
    def __init__(self):
        self.models = {}
        self.best_models = {}
        self.scaler = StandardScaler()
        self.results = {}
        self.feature_names = None
        self.best_threshold = {}
        
    def load_data(self, train_file):
        """加载训练数据"""
        print("Loading training data...")
        
        # 加载训练数据
        train_data = []
        with open(train_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip() and not line.startswith('{"config"'):
                    train_data.append(json.loads(line))
        
        print(f"Loaded {len(train_data)} training samples")
        return train_data
    
    def extract_enhanced_features(self, data):
        """提取增强特征"""
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
        self.feature_names = list(df.columns)
        
        print(f"Enhanced features: {len(self.feature_names)}")
        print(f"Label distribution: {np.bincount(labels)}")
        print(f"Hallucination rate: {np.mean(labels):.3f}")
        
        return df, np.array(labels)
    
    def calculate_repetition(self, text):
        """计算文本重复度"""
        words = text.lower().split()
        if len(words) <= 1:
            return 0
        unique_words = len(set(words))
        return 1 - (unique_words / len(words))
    
    def optimize_hyperparameters(self, X_train, y_train, cv=5):
        """超参数优化"""
        print("\nOptimizing hyperparameters...")
        
        # XGBoost 参数网格 (简化版)
        xgb_params = {
            'n_estimators': [100, 200],
            'max_depth': [3, 5],
            'learning_rate': [0.1, 0.2],
            'subsample': [0.8, 1.0],
        }
        
        # RandomForest 参数网格 (简化版)
        rf_params = {
            'n_estimators': [100, 200],
            'max_depth': [10, 20],
            'min_samples_split': [2, 5],
            'max_features': ['sqrt', 'log2'],
        }
        
        # LightGBM 参数网格 (简化版)
        lgb_params = {
            'n_estimators': [100, 200],
            'max_depth': [3, 5],
            'learning_rate': [0.1, 0.2],
            'num_leaves': [31, 50],
        }
        
        # 逻辑回归参数网格
        lr_params = {
            'C': [0.01, 0.1, 1.0, 10.0],
            'penalty': ['l1', 'l2'],
            'solver': ['liblinear', 'saga'],
            'max_iter': [1000, 2000],
        }
        
        # 计算类别权重
        pos_weight = len(y_train) / (2 * np.sum(y_train))
        
        models_to_optimize = {
            'XGBoost': (xgb.XGBClassifier(random_state=42, scale_pos_weight=pos_weight, eval_metric='logloss'), xgb_params),
            'RandomForest': (RandomForestClassifier(random_state=42, class_weight='balanced'), rf_params),
            'LightGBM': (lgb.LGBMClassifier(random_state=42, class_weight='balanced', verbose=-1), lgb_params),
            'LogisticRegression': (LogisticRegression(random_state=42, class_weight='balanced'), lr_params),
        }
        
        for name, (model, params) in models_to_optimize.items():
            print(f"\nOptimizing {name}...")
            
            # 使用RandomizedSearchCV进行更快的搜索
            search = RandomizedSearchCV(
                model, params, 
                n_iter=10,  # 进一步减少搜索次数
                cv=3,  # 减少CV折数
                scoring='roc_auc',
                n_jobs=-1,
                random_state=42
            )
            
            search.fit(X_train, y_train)
            
            self.best_models[name] = search.best_estimator_
            print(f"Best {name} params: {search.best_params_}")
            print(f"Best {name} CV score: {search.best_score_:.4f}")
    
    def train_ensemble_models(self, X_train, y_train, X_val, y_val):
        """训练集成模型"""
        print("\nTraining ensemble models...")
        
        # 基础模型
        base_models = [
            ('xgb', self.best_models.get('XGBoost', xgb.XGBClassifier(random_state=42))),
            ('rf', self.best_models.get('RandomForest', RandomForestClassifier(random_state=42))),
            ('lgb', self.best_models.get('LightGBM', lgb.LGBMClassifier(random_state=42, verbose=-1))),
            ('lr', self.best_models.get('LogisticRegression', LogisticRegression(random_state=42))),
        ]
        
        # 投票集成
        voting_clf = VotingClassifier(
            estimators=base_models,
            voting='soft'
        )
        
        # 训练集成模型并优化阈值
        voting_clf.fit(X_train, y_train)
        
        # 优化阈值
        threshold, f1_opt = self.optimize_threshold(voting_clf, X_val, y_val)
        self.best_threshold['VotingEnsemble'] = threshold
        
        # 使用优化阈值预测
        y_proba = voting_clf.predict_proba(X_val)[:, 1]
        y_pred_default = voting_clf.predict(X_val)
        y_pred_optimal = (y_proba >= threshold).astype(int)
        
        auc = roc_auc_score(y_val, y_proba)
        
        self.models['VotingEnsemble'] = voting_clf
        self.results['VotingEnsemble'] = {
            'predictions': y_pred_optimal,  # 使用优化阈值的预测
            'probabilities': y_proba,
            'auc': auc,
            'threshold': threshold,
            'f1_optimized': f1_opt,
            'classification_report': classification_report(y_val, y_pred_optimal, output_dict=True),
            'classification_report_default': classification_report(y_val, y_pred_default, output_dict=True)
        }
        
        print(f"Voting Ensemble AUC: {auc:.4f}")
        print(f"Optimal threshold: {threshold:.3f}, F1: {f1_opt:.4f}")
        print("With optimal threshold:")
        print(classification_report(y_val, y_pred_optimal))
        
        # 训练个别最优模型并优化阈值
        for name, model in self.best_models.items():
            print(f"\nTraining optimized {name}...")
            model.fit(X_train, y_train)
            
            # 优化阈值
            threshold, f1_opt = self.optimize_threshold(model, X_val, y_val)
            self.best_threshold[f'Optimized_{name}'] = threshold
            
            # 使用优化阈值预测
            y_proba = model.predict_proba(X_val)[:, 1]
            y_pred_default = model.predict(X_val)
            y_pred_optimal = (y_proba >= threshold).astype(int)
            
            auc = roc_auc_score(y_val, y_proba)
            
            self.models[f'Optimized_{name}'] = model
            self.results[f'Optimized_{name}'] = {
                'predictions': y_pred_optimal,  # 使用优化阈值的预测
                'probabilities': y_proba,
                'auc': auc,
                'threshold': threshold,
                'f1_optimized': f1_opt,
                'classification_report': classification_report(y_val, y_pred_optimal, output_dict=True),
                'classification_report_default': classification_report(y_val, y_pred_default, output_dict=True)
            }
            
            print(f"Optimized {name} AUC: {auc:.4f}")
            print(f"Optimal threshold: {threshold:.3f}, F1: {f1_opt:.4f}")
            print("With optimal threshold:")
            print(classification_report(y_val, y_pred_optimal))
    
    def apply_resampling(self, X_train, y_train):
        """应用重采样技术处理类别不平衡"""
        print("\nApplying resampling techniques...")
        
        # SMOTE + RandomUnderSampler
        smote = SMOTE(random_state=42)
        undersampler = RandomUnderSampler(random_state=42)
        
        # 先过采样再欠采样
        X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
        X_resampled, y_resampled = undersampler.fit_resample(X_resampled, y_resampled)
        
        print(f"Original distribution: {np.bincount(y_train)}")
        print(f"Resampled distribution: {np.bincount(y_resampled)}")
        
        return X_resampled, y_resampled
    
    def evaluate_with_cross_validation(self, X, y, cv=5):
        """交叉验证评估"""
        print("\nCross-validation evaluation...")
        
        for name, model in self.best_models.items():
            cv_scores = cross_val_score(model, X, y, cv=cv, scoring='roc_auc')
            print(f"{name} CV AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    def optimize_threshold(self, model, X_val, y_val):
        """优化分类阈值"""
        y_proba = model.predict_proba(X_val)[:, 1]
        
        # 计算不同阈值下的precision和recall
        precisions, recalls, thresholds = precision_recall_curve(y_val, y_proba)
        
        # 找到F1最大的阈值
        f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
        best_idx = np.argmax(f1_scores)
        best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
        
        return best_threshold, f1_scores[best_idx]
    
    def plot_enhanced_results(self, y_val, output_dir):
        """绘制增强的结果图"""
        plt.figure(figsize=(20, 15))
        
        # ROC曲线
        plt.subplot(3, 4, 1)
        for name, result in self.results.items():
            if 'probabilities' in result:
                fpr, tpr, _ = roc_curve(y_val, result['probabilities'])
                plt.plot(fpr, tpr, label=f"{name} (AUC={result['auc']:.3f})")
        
        plt.plot([0, 1], [0, 1], 'k--', label='Random (AUC=0.500)')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # AUC对比
        plt.subplot(3, 4, 2)
        names = list(self.results.keys())
        aucs = [self.results[name]['auc'] for name in names]
        colors = ['lightcoral' if 'Optimized' in name or 'Ensemble' in name else 'skyblue' for name in names]
        bars = plt.bar(names, aucs, color=colors)
        plt.ylabel('AUC Score')
        plt.title('AUC Comparison')
        plt.xticks(rotation=45)
        plt.ylim(0, 1)
        
        for bar, auc in zip(bars, aucs):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{auc:.3f}', ha='center', va='bottom')
        
        # 特征重要性 (最优XGBoost)
        if 'Optimized_XGBoost' in self.models:
            plt.subplot(3, 4, 3)
            model = self.models['Optimized_XGBoost']
            feature_importance = model.feature_importances_
            
            # 选择前10个最重要的特征
            top_indices = np.argsort(feature_importance)[-10:]
            top_features = [self.feature_names[i] for i in top_indices]
            top_importance = feature_importance[top_indices]
            
            plt.barh(range(len(top_importance)), top_importance)
            plt.yticks(range(len(top_importance)), top_features)
            plt.xlabel('Feature Importance')
            plt.title('Top 10 Feature Importance (XGBoost)')
        
        # Precision-Recall对比
        plt.subplot(3, 4, 4)
        model_names = list(self.results.keys())
        precisions = []
        recalls = []
        f1s = []
        
        for name in model_names:
            report = self.results[name]['classification_report']
            if '1' in report:
                precisions.append(report['1']['precision'])
                recalls.append(report['1']['recall'])
                f1s.append(report['1']['f1-score'])
            else:
                precisions.append(0)
                recalls.append(0)
                f1s.append(0)
        
        x = np.arange(len(model_names))
        width = 0.25
        
        plt.bar(x - width, precisions, width, label='Precision', alpha=0.8)
        plt.bar(x, recalls, width, label='Recall', alpha=0.8)
        plt.bar(x + width, f1s, width, label='F1-Score', alpha=0.8)
        
        plt.xlabel('Models')
        plt.ylabel('Score')
        plt.title('Precision, Recall, F1-Score')
        plt.xticks(x, model_names, rotation=45)
        plt.legend()
        plt.ylim(0, 1)
        
        # 混淆矩阵 (最优模型)
        best_model_name = max(self.results.keys(), key=lambda x: self.results[x]['auc'])
        best_predictions = self.results[best_model_name]['predictions']
        
        plt.subplot(3, 4, 5)
        cm = confusion_matrix(y_val, best_predictions)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix ({best_model_name})')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        
        plt.tight_layout()
        
        # 保存图片
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.savefig(f"{output_dir}/detector_optimized_results_{timestamp}.png", dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_optimized_results(self, output_dir):
        """保存优化结果"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        results_summary = {}
        for name, result in self.results.items():
            if 'classification_report' in result:
                report = result['classification_report']
                if '1' in report:
                    results_summary[name] = {
                        'auc': float(result['auc']),
                        'threshold': float(result.get('threshold', 0.5)),
                        'f1_optimized': float(result.get('f1_optimized', 0)),
                        'precision': float(report['1']['precision']),
                        'recall': float(report['1']['recall']),
                        'f1_score': float(report['1']['f1-score']),
                        'accuracy': float(report['accuracy'])
                    }
                    
                    # 如果有默认阈值结果，也保存对比
                    if 'classification_report_default' in result:
                        default_report = result['classification_report_default']
                        if '1' in default_report:
                            results_summary[name + '_default_threshold'] = {
                                'auc': float(result['auc']),
                                'threshold': 0.5,
                                'precision': float(default_report['1']['precision']),
                                'recall': float(default_report['1']['recall']),
                                'f1_score': float(default_report['1']['f1-score']),
                                'accuracy': float(default_report['accuracy'])
                            }
        
        output_file = f"{output_dir}/detector_optimized_results_{timestamp}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results_summary, f, indent=2, ensure_ascii=False)
        
        print(f"\nOptimized results saved to: {output_file}")
        return results_summary
    
    def save_models(self, output_dir):
        """保存训练好的模型"""
        import joblib
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        models_dir = f"{output_dir}/../models"
        os.makedirs(models_dir, exist_ok=True)
        
        # 保存scaler
        scaler_path = f"{models_dir}/scaler_{timestamp}.joblib"
        joblib.dump(self.scaler, scaler_path)
        print(f"Scaler saved to: {scaler_path}")
        
        # 保存所有模型
        model_paths = {}
        for name, model in self.models.items():
            model_path = f"{models_dir}/{name}_{timestamp}.joblib"
            joblib.dump(model, model_path)
            model_paths[name] = model_path
            print(f"{name} saved to: {model_path}")
        
        # 保存特征名称
        feature_names_path = f"{models_dir}/feature_names_{timestamp}.json"
        with open(feature_names_path, 'w', encoding='utf-8') as f:
            json.dump(self.feature_names, f, indent=2, ensure_ascii=False)
        
        # 保存优化阈值
        threshold_path = f"{models_dir}/thresholds_{timestamp}.json"
        thresholds_serializable = {k: float(v) for k, v in self.best_threshold.items()}
        with open(threshold_path, 'w', encoding='utf-8') as f:
            json.dump(thresholds_serializable, f, indent=2, ensure_ascii=False)
        
        # 保存模型元数据
        metadata = {
            'timestamp': timestamp,
            'scaler_path': scaler_path,
            'feature_names_path': feature_names_path,
            'threshold_path': threshold_path,
            'model_paths': model_paths,
            'best_model': max(self.results.keys(), key=lambda x: self.results[x]['auc']),
            'best_auc': max(result['auc'] for result in self.results.values())
        }
        
        metadata_path = f"{models_dir}/models_metadata_{timestamp}.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        print(f"\nModels metadata saved to: {metadata_path}")
        return metadata_path

def main():
    # 文件路径 - 使用llama2-7b训练数据，保存到llama2-7b目录
    train_file = "/mnt/d/experiments/GraphDeEP/experiment_records/inference_results/llama2-7b/colab_train_simple_part1&2.jsonl"
    output_dir = "/mnt/d/experiments/GraphDeEP/detector/llama2-7b/metaqa-1hop/results"
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 初始化优化检测器
    detector = OptimizedHallucinationDetector()
    
    # 加载训练数据
    train_data = []
    with open(train_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip() and not line.startswith('{"config"'):
                train_data.append(json.loads(line))
    
    print(f"Loaded {len(train_data)} training samples")
    
    # 提取增强特征
    X_train, y_train = detector.extract_enhanced_features(train_data)
    
    # 分割训练集
    X_train_split, X_val, y_train_split, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    
    # 标准化特征
    X_train_scaled = detector.scaler.fit_transform(X_train_split)
    X_val_scaled = detector.scaler.transform(X_val)
    
    print(f"\nData split:")
    print(f"Training: {len(X_train_scaled)} samples")
    print(f"Validation: {len(X_val_scaled)} samples")
    
    # 超参数优化
    detector.optimize_hyperparameters(X_train_scaled, y_train_split)
    
    # 应用重采样
    X_train_resampled, y_train_resampled = detector.apply_resampling(X_train_scaled, y_train_split)
    
    # 训练集成模型
    detector.train_ensemble_models(X_train_resampled, y_train_resampled, X_val_scaled, y_val)
    
    # 交叉验证评估
    detector.evaluate_with_cross_validation(X_train_scaled, y_train_split)
    
    # 绘制结果
    detector.plot_enhanced_results(y_val, output_dir)
    
    # 保存结果
    results = detector.save_optimized_results(output_dir)
    
    # 保存模型
    model_metadata_path = detector.save_models(output_dir)
    
    # 打印最终总结
    print("\n" + "="*80)
    print("OPTIMIZED DETECTOR RESULTS SUMMARY")
    print("="*80)
    
    # 按AUC排序
    sorted_results = sorted(results.items(), key=lambda x: x[1]['auc'], reverse=True)
    
    for model_name, metrics in sorted_results:
        print(f"\n{model_name}:")
        for metric, value in metrics.items():
            if isinstance(value, float):
                print(f"  {metric}: {value:.4f}")
            else:
                print(f"  {metric}: {value}")
    
    print(f"\n🏆 Best model: {sorted_results[0][0]} with AUC: {sorted_results[0][1]['auc']:.4f}")

if __name__ == "__main__":
    main()