"""
PRD+SAS+表面特征消融实验，自动测试集评估，结果保存到prd_sas_with_features_results/
"""
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, precision_recall_curve
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
import xgboost as xgb
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
import warnings
warnings.filterwarnings('ignore')

class PrdSasWithFeaturesDetector:
    def __init__(self):
        self.models = {}
        self.best_models = {}
        self.scaler = StandardScaler()
        self.results = {}
        self.feature_names = None
        self.best_threshold = {}
    
    def extract_features(self, data):
        features = []
        labels = []
        for item in data:
            feature_dict = {
                'gass_score': item.get('gass_score', 0),
                'prd_score': item.get('prd_score', 0),
            }
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
            features.append(feature_dict)
            # 标签
            if 'squad_evaluation' in item:
                is_hallucination = item['squad_evaluation'].get('squad_is_hallucination', False)
            else:
                is_hallucination = not item.get('metrics', {}).get('hit@1', False)
            labels.append(int(is_hallucination))
        df = pd.DataFrame(features)
        self.feature_names = list(df.columns)
        print(f"PRD+SAS+features: {len(self.feature_names)}")
        print(f"Label distribution: {np.bincount(labels)}")
        print(f"Hallucination rate: {np.mean(labels):.3f}")
        return df, np.array(labels)
    def calculate_repetition(self, text):
        words = text.lower().split()
        if len(words) <= 1:
            return 0
        unique_words = len(set(words))
        return 1 - (unique_words / len(words))
    def optimize_hyperparameters(self, X_train, y_train, cv=5):
        print("\nOptimizing hyperparameters...")
        xgb_params = {
            'n_estimators': [100, 200],
            'max_depth': [3, 5],
            'learning_rate': [0.1, 0.2],
            'subsample': [0.8, 1.0],
        }
        rf_params = {
            'n_estimators': [100, 200],
            'max_depth': [10, 20],
            'min_samples_split': [2, 5],
            'max_features': ['sqrt', 'log2'],
        }
        lgb_params = {
            'n_estimators': [100, 200],
            'max_depth': [3, 5],
            'learning_rate': [0.1, 0.2],
            'num_leaves': [31, 50],
        }
        lr_params = {
            'C': [0.01, 0.1, 1.0, 10.0],
            'penalty': ['l1', 'l2'],
            'solver': ['liblinear', 'saga'],
            'max_iter': [1000, 2000],
        }
        pos_weight = len(y_train) / (2 * np.sum(y_train))
        models_to_optimize = {
            'XGBoost': (xgb.XGBClassifier(random_state=42, scale_pos_weight=pos_weight, eval_metric='logloss'), xgb_params),
            'RandomForest': (RandomForestClassifier(random_state=42, class_weight='balanced'), rf_params),
            'LightGBM': (lgb.LGBMClassifier(random_state=42, class_weight='balanced', verbose=-1), lgb_params),
            'LogisticRegression': (LogisticRegression(random_state=42, class_weight='balanced'), lr_params),
        }
        for name, (model, params) in models_to_optimize.items():
            print(f"\nOptimizing {name}...")
            search = RandomizedSearchCV(
                model, params, 
                n_iter=10,
                cv=3,
                scoring='roc_auc',
                n_jobs=-1,
                random_state=42
            )
            search.fit(X_train, y_train)
            self.best_models[name] = search.best_estimator_
            print(f"Best {name} params: {search.best_params_}")
            print(f"Best {name} CV score: {search.best_score_:.4f}")
    def train_ensemble_models(self, X_train, y_train, X_val, y_val):
        print("\nTraining ensemble models...")
        base_models = [
            ('xgb', self.best_models.get('XGBoost', xgb.XGBClassifier(random_state=42))),
            ('rf', self.best_models.get('RandomForest', RandomForestClassifier(random_state=42))),
            ('lgb', self.best_models.get('LightGBM', lgb.LGBMClassifier(random_state=42, verbose=-1))),
            ('lr', self.best_models.get('LogisticRegression', LogisticRegression(random_state=42))),
        ]
        voting_clf = VotingClassifier(
            estimators=base_models,
            voting='soft'
        )
        voting_clf.fit(X_train, y_train)
        threshold, f1_opt = self.optimize_threshold(voting_clf, X_val, y_val)
        self.best_threshold['VotingEnsemble'] = threshold
        y_proba = voting_clf.predict_proba(X_val)[:, 1]
        y_pred_default = voting_clf.predict(X_val)
        y_pred_optimal = (y_proba >= threshold).astype(int)
        auc = roc_auc_score(y_val, y_proba)
        self.models['VotingEnsemble'] = voting_clf
        self.results['VotingEnsemble'] = {
            'predictions': y_pred_optimal,
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
        for name, model in self.best_models.items():
            print(f"\nTraining optimized {name}...")
            model.fit(X_train, y_train)
            threshold, f1_opt = self.optimize_threshold(model, X_val, y_val)
            self.best_threshold[f'Optimized_{name}'] = threshold
            y_proba = model.predict_proba(X_val)[:, 1]
            y_pred_default = model.predict(X_val)
            y_pred_optimal = (y_proba >= threshold).astype(int)
            auc = roc_auc_score(y_val, y_proba)
            self.models[f'Optimized_{name}'] = model
            self.results[f'Optimized_{name}'] = {
                'predictions': y_pred_optimal,
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
        print("\nApplying resampling techniques...")
        smote = SMOTE(random_state=42)
        undersampler = RandomUnderSampler(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
        X_resampled, y_resampled = undersampler.fit_resample(X_resampled, y_resampled)
        print(f"Original distribution: {np.bincount(y_train)}")
        print(f"Resampled distribution: {np.bincount(y_resampled)}")
        return X_resampled, y_resampled
    def evaluate_with_cross_validation(self, X, y, cv=5):
        print("\nCross-validation evaluation...")
        for name, model in self.best_models.items():
            cv_scores = cross_val_score(model, X, y, cv=cv, scoring='roc_auc')
            print(f"{name} CV AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    def optimize_threshold(self, model, X_val, y_val):
        y_proba = model.predict_proba(X_val)[:, 1]
        precisions, recalls, thresholds = precision_recall_curve(y_val, y_proba)
        f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
        best_idx = np.argmax(f1_scores)
        best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
        return best_threshold, f1_scores[best_idx]
    def save_optimized_results(self, output_dir, results_dict=None, prefix="results"):
        import numpy as np
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if results_dict is None:
            results_dict = self.results
        # 递归地把所有numpy类型转为Python类型
        def make_serializable(val):
            if isinstance(val, np.ndarray):
                return val.tolist()
            elif isinstance(val, (np.float32, np.float64)):
                return float(val)
            elif isinstance(val, (np.int32, np.int64)):
                return int(val)
            elif isinstance(val, dict):
                return {k: make_serializable(v) for k, v in val.items()}
            elif isinstance(val, list):
                return [make_serializable(v) for v in val]
            else:
                return val
        serializable_results = {}
        for name, result in results_dict.items():
            serializable_results[name] = {
                k: make_serializable(v)
                for k, v in result.items()
                if k not in ["predictions", "probabilities"]  # 跳过大数组
            }
        output_file = os.path.join(output_dir, f"prd_sas_with_features_{prefix}_{timestamp}.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)
        print(f"\n{prefix.capitalize()} results saved to: {output_file}")
        return output_file
    def save_models(self, output_dir):
        import joblib
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # 强制保存到prd_sas_with_features_results/models/
        models_dir = os.path.join(
            "/mnt/d/experiments/GraphDeEP/detector/qwen2.5-7b/metaqa-1hop/results/prd_sas_with_features_results", "models"
        )
        os.makedirs(models_dir, exist_ok=True)
        scaler_path = os.path.join(models_dir, f"scaler_prd_sas_with_features_{timestamp}.joblib")
        joblib.dump(self.scaler, scaler_path)
        print(f"Scaler saved to: {scaler_path}")
        model_paths = {}
        for name, model in self.models.items():
            model_path = os.path.join(models_dir, f"{name}_prd_sas_with_features_{timestamp}.joblib")
            joblib.dump(model, model_path)
            model_paths[name] = model_path
            print(f"{name} saved to: {model_path}")
        feature_names_path = os.path.join(models_dir, f"feature_names_prd_sas_with_features_{timestamp}.json")
        with open(feature_names_path, 'w', encoding='utf-8') as f:
            json.dump(self.feature_names, f, indent=2, ensure_ascii=False)
        threshold_path = os.path.join(models_dir, f"thresholds_prd_sas_with_features_{timestamp}.json")
        thresholds_serializable = {k: float(v) for k, v in self.best_threshold.items()}
        with open(threshold_path, 'w', encoding='utf-8') as f:
            json.dump(thresholds_serializable, f, indent=2, ensure_ascii=False)
        metadata = {
            'timestamp': timestamp,
            'scaler_path': scaler_path,
            'feature_names_path': feature_names_path,
            'threshold_path': threshold_path,
            'model_paths': model_paths,
            'best_model': max(self.results.keys(), key=lambda x: self.results[x]['auc']),
            'best_auc': max(result['auc'] for result in self.results.values())
        }
        metadata_path = os.path.join(models_dir, f"models_metadata_prd_sas_with_features_{timestamp}.json")
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        print(f"\nModels metadata saved to: {metadata_path}")
        return metadata_path

def evaluate_on_test_set(detector, test_file, output_dir):
    print("\n=== TEST SET EVALUATION (PRD+SAS+features, ALL MODELS) ===")
    test_data = []
    with open(test_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip() and not line.startswith('{"config"'):
                test_data.append(json.loads(line))
    print(f"Loaded {len(test_data)} test samples")
    X_test, y_test = detector.extract_features(test_data)
    X_test_scaled = detector.scaler.transform(X_test)
    test_results = {}
    for model_name, model in detector.models.items():
        threshold = detector.best_threshold.get(model_name, 0.5)
        y_proba = model.predict_proba(X_test_scaled)[:, 1]
        y_pred = (y_proba >= threshold).astype(int)
        auc = roc_auc_score(y_test, y_proba)
        report = classification_report(y_test, y_pred, output_dict=True)
        print(f"\n[{model_name}] Test Set AUC: {auc:.4f}")
        print(f"[{model_name}] Test Set Precision: {report['1']['precision']:.4f}")
        print(f"[{model_name}] Test Set Recall: {report['1']['recall']:.4f}")
        print(f"[{model_name}] Test Set F1: {report['1']['f1-score']:.4f}")
        print(f"[{model_name}] Test Set Accuracy: {report['accuracy']:.4f}")
        test_results[model_name] = {
            'auc': auc,
            'threshold': float(threshold),
            'precision': report['1']['precision'],
            'recall': report['1']['recall'],
            'f1_score': report['1']['f1-score'],
            'accuracy': report['accuracy'],
            'classification_report': report
        }
    # 保存测试集结果
    detector.save_optimized_results(output_dir, test_results, prefix="test")
    return test_results

def main():
    train_file = "/mnt/d/experiments/GraphDeEP/experiment_records/inference_results/qwen2.5-7b/colab_train_simple_part1&2.jsonl"
    test_file = "/mnt/d/experiments/GraphDeEP/experiment_records/inference_results/qwen2.5-7b/colab_test_simple.jsonl"
    output_dir = "/mnt/d/experiments/GraphDeEP/detector/qwen2.5-7b/metaqa-1hop/results/prd_sas_with_features_results"
    os.makedirs(output_dir, exist_ok=True)
    detector = PrdSasWithFeaturesDetector()
    train_data = []
    with open(train_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip() and not line.startswith('{"config"'):
                train_data.append(json.loads(line))
    print(f"Loaded {len(train_data)} training samples")
    X_train, y_train = detector.extract_features(train_data)
    X_train_split, X_val, y_train_split, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    X_train_scaled = detector.scaler.fit_transform(X_train_split)
    X_val_scaled = detector.scaler.transform(X_val)
    print(f"\nData split:")
    print(f"Training: {len(X_train_scaled)} samples")
    print(f"Validation: {len(X_val_scaled)} samples")
    detector.optimize_hyperparameters(X_train_scaled, y_train_split)
    X_train_resampled, y_train_resampled = detector.apply_resampling(X_train_scaled, y_train_split)
    detector.train_ensemble_models(X_train_resampled, y_train_resampled, X_val_scaled, y_val)
    detector.evaluate_with_cross_validation(X_train_scaled, y_train_split)
    # 保存训练集结果
    detector.save_optimized_results(output_dir, detector.results, prefix="train")
    model_metadata_path = detector.save_models(output_dir)
    print("\n" + "="*80)
    print("PRD+SAS+FEATURES DETECTOR RESULTS SUMMARY")
    print("="*80)
    sorted_results = sorted(detector.results.items(), key=lambda x: x[1]['auc'], reverse=True)
    for model_name, metrics in sorted_results:
        print(f"\n{model_name}:")
        for metric, value in metrics.items():
            if isinstance(value, float):
                print(f"  {metric}: {value:.4f}")
            else:
                print(f"  {metric}: {value}")
    print(f"\n🏆 Best model: {sorted_results[0][0]} with AUC: {sorted_results[0][1]['auc']:.4f}")
    # 自动评估测试集
    evaluate_on_test_set(detector, test_file, output_dir)

if __name__ == "__main__":
    main() 