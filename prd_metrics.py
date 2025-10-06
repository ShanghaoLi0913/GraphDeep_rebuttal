"""
PRD (Path Reliance Degree) 指标计算模块

Path Reliance Degree衡量模型对预定义推理路径的依赖程度。
高PRD表示过度依赖结构化路径，可能导致推理僵化和幻觉风险。
低PRD表示灵活的语义推理，准确性更高。

核心公式：
PRD = avg_path_attention - avg_non_path_attention

作者: AI Assistant  
日期: 2025年7月4日
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Set, Optional
from tus_metrics import find_triple_positions

def calculate_prd(
    attention_weights: torch.Tensor,
    gold_triples: List[List[str]],
    answer_start_idx: int,
    answer_end_idx: int,
    input_ids: torch.Tensor,
    tokenizer,
    debug: bool = False
) -> float:
    """
    计算Path Reliance Degree (PRD) - 路径依赖程度
    
    PRD衡量模型对预定义推理路径的依赖程度：
    - 高PRD: 过度依赖路径，可能导致幻觉
    - 低PRD: 灵活推理，准确性更高
    
    参数:
        attention_weights: 注意力权重 shape=[num_layers, num_heads, seq_len, seq_len]
        gold_triples: 数据集中的gold三元组列表（推理路径）
        answer_start_idx: 答案开始位置
        answer_end_idx: 答案结束位置
        input_ids: 输入序列的token ids
        tokenizer: 分词器
        debug: 是否打印调试信息
        
    返回:
        PRD分数 (float)
    """
    if not gold_triples:
        if debug:
            print("DEBUG PRD: No gold triples found")
        return 0.0
        
    # 获取注意力矩阵的维度
    num_layers, num_heads, seq_len, _ = attention_weights.shape
    if debug:
        print(f"DEBUG PRD: Attention matrix shape: {attention_weights.shape}")
    
    # 确保答案位置在序列长度范围内
    answer_start_idx = min(answer_start_idx, seq_len - 1)
    answer_end_idx = min(answer_end_idx, seq_len - 1)
    answer_positions = list(range(answer_start_idx, answer_end_idx + 1))
    
    if not answer_positions:
        if debug:
            print("DEBUG PRD: No valid answer positions found")
        return 0.0
        
    if debug:
        print(f"DEBUG PRD: Answer positions: {answer_positions}")
        answer_text = tokenizer.decode(input_ids[answer_start_idx:answer_end_idx+1])
        print(f"DEBUG PRD: Answer text: '{answer_text}'")
    
    # 获取路径位置（gold triples对应的token位置）
    path_positions = find_triple_positions(input_ids, tokenizer, gold_triples, debug=debug)
    
    if not path_positions:
        if debug:
            print("DEBUG PRD: No path positions found")
        return 0.0
    
    # 创建non-path位置集合
    all_positions = set(range(seq_len))
    non_path_positions = all_positions - path_positions
    
    if debug:
        print(f"DEBUG PRD: Path positions count: {len(path_positions)}")
        print(f"DEBUG PRD: Non-path positions count: {len(non_path_positions)}")
    
    # 计算平均注意力权重
    path_attention_sum = 0.0
    non_path_attention_sum = 0.0
    valid_count = 0
    
    for layer in range(num_layers):
        for head in range(num_heads):
            layer_attention = attention_weights[layer, head]
            
            for ans_pos in answer_positions:
                if ans_pos >= seq_len:
                    continue
                
                attention_scores = layer_attention[ans_pos]
                attention_probs = F.softmax(attention_scores, dim=-1)
                
                # 计算路径位置注意力
                valid_path_positions = [pos for pos in path_positions if pos < seq_len]
                if valid_path_positions:
                    path_attention = attention_probs[valid_path_positions].sum().item()
                    path_attention_sum += path_attention
                
                # 计算非路径位置注意力
                valid_non_path_positions = [pos for pos in non_path_positions if pos < seq_len]
                if valid_non_path_positions:
                    non_path_attention = attention_probs[valid_non_path_positions].sum().item()
                    non_path_attention_sum += non_path_attention
                
                valid_count += 1
    
    if valid_count == 0:
        if debug:
            print("DEBUG PRD: No valid calculations performed")
        return 0.0
    
    avg_path_attention = path_attention_sum / valid_count
    avg_non_path_attention = non_path_attention_sum / valid_count
    
    # 计算PRD分数：路径注意力 - 非路径注意力
    prd_score = avg_path_attention - avg_non_path_attention
    
    if debug:
        print(f"DEBUG PRD: Path attention={avg_path_attention:.4f}, Non-path attention={avg_non_path_attention:.4f}")
        print(f"DEBUG PRD: PRD score={prd_score:.4f}")
        
        # 解释PRD分数
        if prd_score > 0.01:
            print("DEBUG PRD: High path reliance - potential hallucination risk")
        elif prd_score < -0.01:
            print("DEBUG PRD: Low path reliance - flexible reasoning")
        else:
            print("DEBUG PRD: Moderate path reliance")
    
    return float(prd_score)

def interpret_prd_score(prd_score: float) -> Dict[str, str]:
    """
    解释PRD分数的含义
    
    参数:
        prd_score: PRD分数
        
    返回:
        包含解释信息的字典
    """
    if prd_score > 0.01:
        return {
            'level': 'high',
            'interpretation': '高路径依赖 - 过度依赖预定义推理路径',
            'risk': 'high',
            'recommendation': '可能存在幻觉风险，建议验证语义一致性'
        }
    elif prd_score < -0.01:
        return {
            'level': 'low', 
            'interpretation': '低路径依赖 - 灵活的语义推理',
            'risk': 'low',
            'recommendation': '推理模式健康，准确性较高'
        }
    else:
        return {
            'level': 'moderate',
            'interpretation': '适度路径依赖 - 平衡的推理模式', 
            'risk': 'medium',
            'recommendation': '需结合其他指标综合判断'
        }