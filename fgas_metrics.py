"""
FFN-Gold Alignment Score (FGAS) 计算模块

本模块实现了基于FFN层语义表达的Gold三元组对齐分数计算。
FGAS用于量化模型生成答案过程中，FFN层是否表达了与golden triples对齐的知识内容。

与TUS的区别：
- TUS: 关注注意力机制是否"看到"了正确的信息 (attention-based)
- FGAS: 关注FFN层是否"理解"和"表达"了正确的语义内容 (representation-based)
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from transformers import AutoModel, AutoTokenizer
import logging

def extract_ffn_representations(
    model,
    input_ids: torch.Tensor,
    answer_start_idx: int,
    answer_end_idx: int,
    layer_indices: Optional[List[int]] = None
) -> torch.Tensor:
    """
    提取指定层FFN的输出表示
    
    参数:
        model: 预训练的语言模型
        input_ids: 输入序列的token ids
        answer_start_idx: 答案开始位置
        answer_end_idx: 答案结束位置
        layer_indices: 要提取的层索引，默认为最后4层
        
    返回:
        FFN输出表示 shape=[num_layers, seq_len, hidden_size]
    """
    if layer_indices is None:
        # 默认使用最后4层
        layer_indices = list(range(model.config.num_hidden_layers - 4, model.config.num_hidden_layers))
    
    # 确保输入在正确的设备上
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)  # 添加batch维度
    input_ids = input_ids.to(model.device)
    
    ffn_outputs = []
    
    with torch.no_grad():
        # 获取模型的隐藏状态
        outputs = model(input_ids, output_hidden_states=True)
        hidden_states = outputs.hidden_states  # tuple of (batch_size, seq_len, hidden_size)
        
        for layer_idx in layer_indices:
            # 获取指定层的隐藏状态
            layer_hidden = hidden_states[layer_idx]  # 修复：去掉+1，直接使用layer_idx
            
            # 提取答案部分的表示
            answer_representations = layer_hidden[0, answer_start_idx:answer_end_idx + 1, :]
            ffn_outputs.append(answer_representations)
    
    # 返回形状为 [num_layers, answer_length, hidden_size] 的张量
    return torch.stack(ffn_outputs, dim=0)

def encode_gold_triples(
    gold_triples: List[List[str]],
    tokenizer,
    model,
    device: torch.device
) -> torch.Tensor:
    """
    将gold三元组编码为语义表示
    
    参数:
        gold_triples: 金三元组列表
        tokenizer: 分词器
        model: 预训练模型
        device: 计算设备
        
    返回:
        金三元组的语义表示 shape=[num_triples, hidden_size]
    """
    if not gold_triples:
        return torch.empty(0, model.config.hidden_size, device=device)
    
    # 将三元组转换为文本格式
    triple_texts = []
    for h, r, t in gold_triples:
        # 格式化三元组文本，使用更自然的格式
        triple_text = f"{h} {r} {t}"
        triple_texts.append(triple_text)
    
    # 批量编码三元组
    encoded = tokenizer(
        triple_texts,
        padding=True,
        truncation=True,
        max_length=64,  # 减小长度，因为三元组通常较短
        return_tensors="pt",
        add_special_tokens=True
    )
    
    # 移动到正确的设备
    encoded = {k: v.to(device) for k, v in encoded.items()}
    
    with torch.no_grad():
        # 获取三元组的表示
        outputs = model(**encoded, output_hidden_states=True)
        # 使用最后一层的平均池化表示
        hidden_states = outputs.hidden_states[-1]  # 使用最后一层
        attention_mask = encoded['attention_mask']
        
        # 计算平均池化
        masked_hidden = hidden_states * attention_mask.unsqueeze(-1)
        sum_hidden = masked_hidden.sum(1)
        sum_mask = attention_mask.sum(1, keepdim=True)
        triple_representations = sum_hidden / sum_mask
    
    return triple_representations

def calculate_semantic_similarity(
    answer_representations: torch.Tensor,
    gold_representations: torch.Tensor,
    similarity_metric: str = "cosine"
) -> torch.Tensor:
    """
    计算答案表示与金三元组表示之间的语义相似度
    
    参数:
        answer_representations: 答案的FFN表示 shape=[num_layers, answer_length, hidden_size]
        gold_representations: 金三元组表示 shape=[num_triples, hidden_size]
        similarity_metric: 相似度计算方法，支持 "cosine", "dot_product"
        
    返回:
        相似度矩阵 shape=[num_layers, answer_length, num_triples]
    """
    if gold_representations.size(0) == 0:
        return torch.zeros(answer_representations.size(0), answer_representations.size(1), 0)
    
    num_layers, answer_length, hidden_size = answer_representations.shape
    num_triples = gold_representations.size(0)
    
    similarities = []
    
    for layer_idx in range(num_layers):
        layer_similarities = []
        
        for answer_pos in range(answer_length):
            answer_vec = answer_representations[layer_idx, answer_pos, :]  # [hidden_size]
            
            if similarity_metric == "cosine":
                # 计算余弦相似度
                sim = F.cosine_similarity(
                    answer_vec.unsqueeze(0),  # [1, hidden_size]
                    gold_representations,     # [num_triples, hidden_size]
                    dim=1
                )
            elif similarity_metric == "dot_product":
                # 计算点积相似度
                sim = torch.matmul(gold_representations, answer_vec)  # [num_triples]
            else:
                raise ValueError(f"Unsupported similarity metric: {similarity_metric}")
            
            layer_similarities.append(sim)
        
        # 堆叠当前层的所有位置相似度
        similarities.append(torch.stack(layer_similarities, dim=0))  # [answer_length, num_triples]
    
    return torch.stack(similarities, dim=0)  # [num_layers, answer_length, num_triples]

def calculate_fgas(
    model,
    tokenizer,
    input_ids: torch.Tensor,
    gold_triples: List[List[str]],
    answer_start_idx: int,
    answer_end_idx: int,
    layer_indices: Optional[List[int]] = None,
    similarity_metric: str = "cosine",
    aggregation_method: str = "max",
    debug: bool = False
) -> float:
    """
    计算FFN-Gold Alignment Score (FGAS)
    
    参数:
        model: 预训练的语言模型
        tokenizer: 分词器
        input_ids: 输入序列的token ids
        gold_triples: 金三元组列表
        answer_start_idx: 答案开始位置
        answer_end_idx: 答案结束位置
        layer_indices: 要分析的层索引
        similarity_metric: 相似度计算方法
        aggregation_method: 聚合方法 ("max", "mean", "weighted_mean")
        debug: 是否输出调试信息
        
    返回:
        FGAS分数 (0-1之间的浮点数)
    """
    logger = logging.getLogger(__name__)
    
    # 参数验证
    if not gold_triples:
        if debug:
            print("DEBUG: No gold triples provided, returning FGAS = 0.0")
        return 0.0
    
    try:
        if debug:
            print(f"DEBUG: Starting FGAS calculation")
            print(f"DEBUG: Gold triples: {gold_triples}")
            print(f"DEBUG: Answer positions: {answer_start_idx}-{answer_end_idx}")
            print(f"DEBUG: Input IDs shape: {input_ids.shape}")
        
        # 1. 提取FFN表示
        answer_representations = extract_ffn_representations(
            model, input_ids, answer_start_idx, answer_end_idx, layer_indices
        )
        
        if debug:
            print(f"DEBUG: Answer representations shape: {answer_representations.shape}")
        
        # 2. 编码金三元组
        gold_representations = encode_gold_triples(
            gold_triples, tokenizer, model, model.device
        )
        
        if debug:
            print(f"DEBUG: Gold representations shape: {gold_representations.shape}")
        
        # 3. 计算语义相似度
        similarities = calculate_semantic_similarity(
            answer_representations, gold_representations, similarity_metric
        )
        
        if debug:
            print(f"DEBUG: Similarities shape: {similarities.shape}")
            print(f"DEBUG: Max similarity: {similarities.max().item():.4f}")
            print(f"DEBUG: Min similarity: {similarities.min().item():.4f}")
            print(f"DEBUG: Mean similarity: {similarities.mean().item():.4f}")
        
        # 4. 聚合相似度分数
        if aggregation_method == "max":
            # 对每个答案位置，取与金三元组的最大相似度
            max_similarities = similarities.max(dim=2)[0]  # [num_layers, answer_length]
            # 对所有层和位置取平均
            fgas_score = max_similarities.mean().item()
            
        elif aggregation_method == "mean":
            # 计算所有相似度的平均值
            fgas_score = similarities.mean().item()
            
        elif aggregation_method == "weighted_mean":
            # 使用注意力权重作为权重（这里简化为均匀权重）
            weights = torch.ones_like(similarities)
            weighted_similarities = (similarities * weights).sum() / weights.sum()
            fgas_score = weighted_similarities.item()
            
        else:
            raise ValueError(f"Unsupported aggregation method: {aggregation_method}")
        
        if debug:
            print(f"DEBUG: Final FGAS score before clipping: {fgas_score:.6f}")
        
        # 确保分数在合理范围内，但不强制限制在[0,1]
        final_score = max(0.0, fgas_score)
        
        if debug:
            print(f"DEBUG: Final FGAS score: {final_score:.6f}")
        
        return final_score
        
    except Exception as e:
        if debug:
            print(f"DEBUG: Error calculating FGAS: {str(e)}")
            import traceback
            traceback.print_exc()
        logger.error(f"Error calculating FGAS: {str(e)}")
        return 0.0

def batch_calculate_fgas(
    model,
    tokenizer,
    batch_input_ids: torch.Tensor,
    batch_gold_triples: List[List[List[str]]],
    batch_answer_positions: List[Tuple[int, int]],
    **kwargs
) -> List[float]:
    """
    批量计算FGAS分数
    
    参数:
        model: 预训练的语言模型
        tokenizer: 分词器
        batch_input_ids: 批次输入序列
        batch_gold_triples: 批次金三元组
        batch_answer_positions: 批次答案位置
        **kwargs: 传递给calculate_fgas的其他参数
        
    返回:
        FGAS分数列表
    """
    fgas_scores = []
    
    for i in range(len(batch_input_ids)):
        input_ids = batch_input_ids[i]
        gold_triples = batch_gold_triples[i]
        answer_start, answer_end = batch_answer_positions[i]
        
        fgas_score = calculate_fgas(
            model, tokenizer, input_ids, gold_triples,
            answer_start, answer_end, **kwargs
        )
        fgas_scores.append(fgas_score)
    
    return fgas_scores 