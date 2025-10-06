#!/usr/bin/env python3
"""
诊断baseline特征问题
"""
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, classification_report
from sentence_transformers import SentenceTransformer

def load_and_analyze_features():
    """加载数据并分析特征"""
    print("🔍 诊断baseline特征问题...")
    
    # 加载测试数据
    test_data = []
    test_file = "/mnt/d/experiments/GraphDeEP/experiment_records/inference_results/llama2-7b/colab_test_simple.jsonl"
    
    with open(test_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip() and not line.startswith('{"config"'):
                test_data.append(json.loads(line))
                if len(test_data) >= 100:  # 只分析前100个样本
                    break
    
    print(f"✅ 加载了 {len(test_data)} 个样本")
    
    # 提取标签
    labels = []
    for item in test_data:
        if 'squad_evaluation' in item:
            is_hallucination = item['squad_evaluation'].get('squad_is_hallucination', False)
        else:
            is_hallucination = not item.get('metrics', {}).get('hit@1', False)
        labels.append(int(is_hallucination))
    
    labels = np.array(labels)
    print(f"📊 标签分布: {np.bincount(labels)} (幻觉率: {np.mean(labels):.3f})")
    
    # 加载sentence transformer（轻量级版本）
    print("📥 加载SentenceTransformer...")
    sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # 提取一个简单特征：BERTScore vs Question
    print("🔧 提取BERTScore特征...")
    bertscore_features = []
    
    for item in test_data:
        question = item.get('question', '')
        output = item.get('model_output', '')
        
        if question and output:
            # 计算问题和回答的相似度
            question_emb = sentence_model.encode([question])
            answer_emb = sentence_model.encode([output])
            similarity = float(np.dot(question_emb[0], answer_emb[0]) / 
                             (np.linalg.norm(question_emb[0]) * np.linalg.norm(answer_emb[0])))
        else:
            similarity = 0.0
        
        bertscore_features.append(similarity)
    
    bertscore_features = np.array(bertscore_features).reshape(-1, 1)
    
    # 分析特征分布
    print("\n📈 BERTScore特征分析:")
    print(f"  范围: [{bertscore_features.min():.3f}, {bertscore_features.max():.3f}]")
    print(f"  均值±标准差: {bertscore_features.mean():.3f}±{bertscore_features.std():.3f}")
    
    # 分组分析
    hallucination_mask = labels == 1
    correct_mask = labels == 0
    
    hall_scores = bertscore_features[hallucination_mask].flatten()
    correct_scores = bertscore_features[correct_mask].flatten()
    
    print(f"  幻觉样本: {hall_scores.mean():.3f}±{hall_scores.std():.3f}")
    print(f"  正确样本: {correct_scores.mean():.3f}±{correct_scores.std():.3f}")
    print(f"  差异: {correct_scores.mean() - hall_scores.mean():.3f}")
    
    # 快速训练测试
    print("\n🚀 快速分类器测试:")
    
    # 标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(bertscore_features)
    
    # 训练逻辑回归
    clf = LogisticRegression(random_state=42, class_weight='balanced')
    clf.fit(X_scaled, labels)
    
    # 预测
    y_proba = clf.predict_proba(X_scaled)[:, 1]
    y_pred = clf.predict(X_scaled)
    
    # 评估
    auc = roc_auc_score(labels, y_proba)
    report = classification_report(labels, y_pred, output_dict=True, zero_division=0)
    
    print(f"  AUC: {auc:.4f}")
    print(f"  F1: {report['1']['f1-score']:.4f}")
    print(f"  Precision: {report['1']['precision']:.4f}")
    print(f"  Recall: {report['1']['recall']:.4f}")
    
    # 检查预测分布
    print(f"  预测分布: {np.bincount(y_pred)}")
    print(f"  概率范围: [{y_proba.min():.3f}, {y_proba.max():.3f}]")
    
    # 阈值分析
    from sklearn.metrics import precision_recall_curve
    precisions, recalls, thresholds = precision_recall_curve(labels, y_proba)
    f1_scores = 2 * (precisions[:-1] * recalls[:-1]) / (precisions[:-1] + recalls[:-1] + 1e-8)
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx]
    best_f1 = f1_scores[best_idx]
    
    print(f"\n🎯 阈值优化:")
    print(f"  最佳阈值: {best_threshold:.3f}")
    print(f"  最佳F1: {best_f1:.4f}")
    
    # 用最佳阈值重新预测
    y_pred_opt = (y_proba >= best_threshold).astype(int)
    print(f"  优化后预测分布: {np.bincount(y_pred_opt)}")
    
    return auc, best_f1

if __name__ == "__main__":
    load_and_analyze_features()