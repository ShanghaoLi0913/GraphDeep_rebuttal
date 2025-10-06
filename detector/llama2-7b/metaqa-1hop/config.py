"""
Detector配置文件
包含所有训练参数和路径设置
"""

# 数据文件路径
DATA_CONFIG = {
    'train_file': "/mnt/d/experiments/GraphDeEP/experiment_records/inference_results/llama2-7b/colab_train_simple_part1&2.jsonl",
    'test_file': "/mnt/d/experiments/GraphDeEP/experiment_records/inference_results/llama2-7b/colab_test_simple.jsonl",
    'output_dir': "/mnt/d/experiments/GraphDeEP/detector/llama2-7b/metaqa-1hop/results",
    'models_dir': "/mnt/d/experiments/GraphDeEP/detector/llama2-7b/metaqa-1hop/models"
}

# 特征配置
FEATURE_CONFIG = {
    'primary_features': ['gass_score', 'prd_score'],
    'baseline_features': ['output_length', 'answer_length', 'repetition_score', 'squad_f1', 'squad_exact_match'],
    'use_squad_label': True,  # 使用SQuAD评估作为标签，否则使用hit@1
}

# 模型配置
MODEL_CONFIG = {
    'random_state': 42,
    'test_size': 0.2,  # 验证集比例
    'cross_validation_folds': 5,
    
    # 各模型超参数
    'logistic_regression': {
        'C': 1.0,
        'class_weight': 'balanced',
        'max_iter': 1000
    },
    
    'random_forest': {
        'n_estimators': 100,
        'class_weight': 'balanced',
        'max_depth': None,
        'min_samples_split': 2
    },
    
    'xgboost': {
        'n_estimators': 100,
        'learning_rate': 0.1,
        'max_depth': 6,
        'subsample': 0.8
    }
}

# 评估配置
EVAL_CONFIG = {
    'metrics': ['precision', 'recall', 'f1-score', 'auc'],
    'plot_roc': True,
    'plot_feature_importance': True,
    'save_confusion_matrix': True
}

# 输出配置
OUTPUT_CONFIG = {
    'save_models': True,
    'save_predictions': True,
    'save_plots': True,
    'verbose': True
}