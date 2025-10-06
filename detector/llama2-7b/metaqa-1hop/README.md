# Hallucination Detector

这个目录包含用于训练和评估GraphDeEP幻觉检测器的代码和结果。

## 🏆 最佳性能

| 模型 | Validation AUC | **Test Set AUC** | 精确度 | 召回率 | F1 |
|------|----------------|------------------|--------|--------|-----|
| **VotingEnsemble** | 0.9217 | **0.8849** | 0.488 | 0.577 | 0.528 |
| Optimized RandomForest | 0.9210 | **0.8829** | 0.613 | 0.433 | 0.507 |

## 📁 文件结构

```
detector/
├── README.md                        # 本文件
├── train_detector_fixed.py          # 基础版训练脚本 (AUC 0.87)
├── train_detector_optimized.py      # 🏆 优化版训练脚本 (AUC 0.88)
├── evaluate_on_test_set.py          # Test Set真实性能评估
├── config.py                        # 配置文件
├── results/                         # 实验结果
│   ├── detector_optimized_results_20250708_143038.json  # Validation结果
│   ├── test_set_results_20250708_195010.json           # 🎯 Test Set真实结果
│   └── test_set_evaluation_20250708_195005.png         # 可视化图表
└── models/                          # 保存的模型文件
```

## 🧪 实验版本对比

### 基础版 (`train_detector_fixed.py`)
- **特征**: 6个 (GASS + PRD + 表面特征)
- **模型**: 单一模型 (XGBoost, RandomForest, LogisticRegression)
- **性能**: AUC 0.8740
- **用途**: Baseline对比

### 优化版 (`train_detector_optimized.py`) ⭐
- **特征**: 19个增强特征 + 平衡校准GASS
- **模型**: 集成学习 (VotingEnsemble)
- **优化**: 超参数调优 + SMOTE平衡采样
- **性能**: Test Set AUC **0.8849**
- **用途**: 主力检测器

## 🎯 特征工程

### 核心特征
- **GASS Score**: Graph-based Alignment Score (需平衡校准)
- **PRD Score**: Path Reliance Degree 
- **Balanced Calibrated GASS**: 校准后的GASS分数
- **TUS Score**: Triple Utilization Score
- **GASS-JSD Score**: GASS Jensen-Shannon Divergence

### 组合特征
- **GASS/PRD Ratio**: GASS与PRD的比值
- **GASS-TUS Diff**: GASS与TUS的差值
- **Balanced-Original GASS Diff**: 校准前后GASS差值

### 表面特征
- 输出长度、重复度、单词统计
- 问题类型特征 (what/which/who)
- 格式特征 (ans:前缀、逗号数量等)

## 🚀 使用方法

### 1. 训练最优检测器
```bash
cd /mnt/d/experiments/GraphDeEP/detector
python train_detector_optimized.py
```

### 2. 真实性能评估
```bash
python evaluate_on_test_set.py
```

### 3. 基础版本对比
```bash
python train_detector_fixed.py
```

## 📊 数据要求

### 训练数据
- `experiment_records/inference_results/llama2-7b/colab_train_simple_part1&2.jsonl`
- 合并的part1&2训练数据

### 测试数据  
- `experiment_records/inference_results/llama2-7b/colab_test_simple.jsonl`
- 测试集数据

### 数据格式
```json
{
    "gass_score": 0.1776,
    "prd_score": 0.7301,
    "tus_score": 0.8732,
    "gass_jsd_score": 1.0,
    "squad_evaluation": {
        "squad_is_hallucination": false,
        "squad_f1_score": 0.667
    },
    "model_output": "ans: answer text",
    "question": "what movies did X act in"
}
```

## 📈 评估指标

- **AUC-ROC**: 主要评估指标 (0.8849)
- **Precision**: 幻觉检测精确度 (48.8%)
- **Recall**: 幻觉检测召回率 (57.7%)
- **F1-Score**: 综合性能指标 (52.8%)

## ⚠️ 重要发现

1. **Validation vs Test Set**: Validation性能被高估3.7%
2. **GASS需要校准**: 原始GASS方向错误，需要平衡校准
3. **PRD表现稳定**: 无需校准，跨模型一致性好
4. **集成学习有效**: VotingEnsemble优于单一模型

## 🎯 实际应用

- **检测能力**: 能识别57.7%的幻觉
- **误报率**: 51.2%的幻觉预测是错误的
- **适用场景**: 辅助人工审核，降低幻觉风险

## 🔬 研究贡献

1. **首个基于GASS+PRD的幻觉检测器**
2. **发现并解决GASS方向性问题**
3. **证明平衡校准的有效性**
4. **提供真实、无偏的性能评估**