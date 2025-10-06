"""
GASS (Graph Answer Set Score) 指标计算

这个模块实现基于语义对齐的GASS指标，用于衡量生成token的FFN表示与金标准知识的相似度。

作者: AI Assistant
日期: 2025年6月26日
"""

import torch
import torch.nn.functional as F
import logging
from typing import List, Optional
import numpy as np


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
        layer_indices: 要提取的层索引，默认使用倒数第二层 [-2]
        
    返回:
        FFN输出表示 shape=[num_layers, answer_length, hidden_size]
    """
    if layer_indices is None:
        layer_indices = [-2]  # 默认使用倒数第二层
    
    # 确保输入在正确的设备上
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)  # 添加batch维度
    input_ids = input_ids.to(model.device)
    
    ffn_outputs = []
    
    with torch.no_grad():
        # 获取模型的隐藏状态 - 使用FP32精度避免数值溢出
        with torch.autocast(device_type="cuda", dtype=torch.float32, enabled=True):
            outputs = model(input_ids, output_hidden_states=True, use_cache=False)
        hidden_states = outputs.hidden_states  # tuple of (batch_size, seq_len, hidden_size)
        
        for layer_idx in layer_indices:
            # 获取指定层的隐藏状态
            layer_hidden = hidden_states[layer_idx]
            
            # 提取答案部分的表示
            answer_representations = layer_hidden[0, answer_start_idx:answer_end_idx, :]
            ffn_outputs.append(answer_representations)
    
    # 返回形状为 [num_layers, answer_length, hidden_size] 的张量
    return torch.stack(ffn_outputs, dim=0)


def encode_gold_knowledge_set(
    gold_knowledge: List[List[str]],
    tokenizer,
    model,
    device: torch.device,
    debug: bool = False
) -> torch.Tensor:
    """
    将Gold Knowledge Set编码为语义表示
    
    参数:
        gold_knowledge: Gold Knowledge Set三元组列表
        tokenizer: 分词器
        model: 预训练模型
        device: 计算设备
        debug: 是否输出调试信息
        
    返回:
        Gold Knowledge的语义表示 shape=[num_triples, hidden_size]
    """
    if not gold_knowledge:
        return torch.empty(0, model.config.hidden_size, device=device)
    
    if debug:
        print(f"DEBUG: Encoding {len(gold_knowledge)} gold knowledge triples")
    
    # 将三元组转换为文本格式
    triple_texts = []
    for h, r, t in gold_knowledge:
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
    
    if debug:
        print(f"DEBUG: Encoded input_ids shape: {encoded['input_ids'].shape}")
        print(f"DEBUG: Sequence lengths: {encoded['attention_mask'].sum(dim=1)}")
        for i, (text, ids) in enumerate(zip(triple_texts, encoded['input_ids'])):
            seq_len = encoded['attention_mask'][i].sum().item()
            print(f"DEBUG: Triple {i}: '{text}' -> length={seq_len}")
    
    # 移动到正确的设备
    encoded = {k: v.to(device) for k, v in encoded.items()}
    
    with torch.no_grad():
        # 获取三元组的表示 - 使用FP32精度避免数值溢出
        with torch.autocast(device_type="cuda", dtype=torch.float32, enabled=True):
            outputs = model(**encoded, output_hidden_states=True)
        # 使用倒数第二层的平均池化表示
        hidden_states = outputs.hidden_states[-2]  # 使用倒数第二层
        attention_mask = encoded['attention_mask']
        
        if debug:
            print(f"DEBUG: Hidden states shape: {hidden_states.shape}")
            print(f"DEBUG: Attention mask shape: {attention_mask.shape}")
            print(f"DEBUG: Hidden states stats - mean: {hidden_states.mean():.6f}, std: {hidden_states.std():.6f}")
            print(f"DEBUG: Has NaN in hidden states: {torch.isnan(hidden_states).any()}")
            print(f"DEBUG: Has Inf in hidden states: {torch.isinf(hidden_states).any()}")
        
        # 计算平均池化 (CLS-style pooling 或 averaging over textual linearization)
        masked_hidden = hidden_states * attention_mask.unsqueeze(-1)
        sum_hidden = masked_hidden.sum(1)
        sum_mask = attention_mask.sum(1, keepdim=True)
        
        # 防止除零：确保sum_mask不为0
        sum_mask = torch.clamp(sum_mask, min=1e-8)
        triple_representations = sum_hidden / sum_mask
        
        # 检查并处理NaN值
        if torch.isnan(triple_representations).any():
            if debug:
                print(f"WARNING: NaN detected in triple representations, replacing with zeros")
            triple_representations = torch.nan_to_num(triple_representations, nan=0.0, posinf=0.0, neginf=0.0)
        
        if debug:
            print(f"DEBUG: Triple representations shape: {triple_representations.shape}")
            print(f"DEBUG: Triple representations stats - mean: {triple_representations.mean():.6f}, std: {triple_representations.std():.6f}")
            print(f"DEBUG: Has NaN in triple representations: {torch.isnan(triple_representations).any()}")
            print(f"DEBUG: Has Inf in triple representations: {torch.isinf(triple_representations).any()}")
            # 检查每个triple的norm
            norms = torch.norm(triple_representations, dim=1)
            print(f"DEBUG: Triple representation norms: {norms}")
            print(f"DEBUG: Any zero norms in triples: {(norms == 0).any()}")
    
    return triple_representations


def calculate_gass(
    model,
    tokenizer,
    input_ids,
    retrieved_subgraph: List[List[str]],
    gold_subgraph: List[List[str]],
    answer_start_idx: int,
    answer_end_idx: int,
    debug: bool = False,
    layer_indices: Optional[List[int]] = None
) -> float:
    """
    计算基于语义对齐的Graph Answer Set Score (GASS)
    
    GASS公式定义：
    对每个生成token ti，计算其对知识集合G的最大语义对齐分数：
        align(ti) = max_{j∈[1,m]} cos(hi, gj)
    最终GASS得分为所有生成token的对齐分数平均：
        GASS = (1/n) * Σ_{i=1}^n align(ti)
    
    参数:
        model: 预训练的语言模型
        tokenizer: 分词器
        input_ids: 输入序列的token ids
        retrieved_subgraph: 检索到的三元组列表 (暂时保留以保持接口兼容)
        gold_subgraph: 金标准三元组列表 (用作Gold Knowledge Set G)
        answer_start_idx: 答案开始位置
        answer_end_idx: 答案结束位置
        debug: 是否输出调试信息
        
    返回:
        GASS分数 (0-1之间的浮点数)
        
    公式:
        T = {t1, t2, ..., tn}: 模型生成的输出token序列
        hi ∈ R^d: 第i个token ti的FFN表示 (transformer block的倒数第二层FFN输出)
        G = {g1, g2, ..., gm}: Gold Knowledge Set (embedding表示)
        gj ∈ R^d: 第j个triple的表示
        
        align(ti) = max_{j∈[1,m]} cos(hi, gj)
        GASS = (1/n) * Σ_{i=1}^n align(ti)
    """
    logger = logging.getLogger(__name__)
    
    # 参数验证
    if not gold_subgraph:
        if debug:
            print("DEBUG: Empty gold knowledge set provided, returning GASS = 0.0")
        return 0.0
    
    if answer_start_idx >= answer_end_idx:
        if debug:
            print(f"DEBUG: Invalid answer positions: start={answer_start_idx}, end={answer_end_idx}")
        return 0.0
    
    try:
        if debug:
            print(f"DEBUG: Starting GASS calculation (semantic alignment)")
            print(f"DEBUG: Gold knowledge set size: {len(gold_subgraph)}")
            print(f"DEBUG: Answer positions: {answer_start_idx}-{answer_end_idx}")
        
        # 1. 提取生成token的FFN表示
        answer_representations = extract_ffn_representations(
            model, input_ids, answer_start_idx, answer_end_idx, 
            layer_indices=layer_indices
        )
        
        if debug:
            print(f"DEBUG: Answer representations shape: {answer_representations.shape}")
        
        # 检查是否有有效的表示
        if answer_representations.numel() == 0:
            if debug:
                print("DEBUG: Empty answer representations")
            return 0.0
        
        # 2. 编码Gold Knowledge Set
        gold_representations = encode_gold_knowledge_set(
            gold_subgraph, tokenizer, model, model.device, debug
        )
        
        if debug:
            print(f"DEBUG: Gold representations shape: {gold_representations.shape}")
        
        if gold_representations.size(0) == 0:
            if debug:
                print("DEBUG: Empty gold representations")
            return 0.0
        
        # 3. 计算每个token的对齐分数
        alignment_scores = []
        
        # answer_representations shape: [num_layers, answer_length, hidden_size]
        # 我们只使用第一层（倒数第二层）
        token_representations = answer_representations[0]  # [answer_length, hidden_size]
        
        for i, token_repr in enumerate(token_representations):
            # token_repr shape: [hidden_size]
            # gold_representations shape: [num_triples, hidden_size]
            
            if debug and i < 3:  # 详细debug前几个token
                print(f"DEBUG: Token {i} representation stats - mean: {token_repr.mean():.6f}, std: {token_repr.std():.6f}")
                print(f"DEBUG: Token {i} norm: {torch.norm(token_repr):.6f}")
                print(f"DEBUG: Token {i} has NaN: {torch.isnan(token_repr).any()}")
                print(f"DEBUG: Token {i} has Inf: {torch.isinf(token_repr).any()}")
            
            # 计算余弦相似度: align(ti) = max_{j∈[1,m]} cos(hi, gj)
            similarities = F.cosine_similarity(
                token_repr.unsqueeze(0),  # [1, hidden_size]
                gold_representations,     # [num_triples, hidden_size]
                dim=1
            )
            
            if debug and i < 3:  # 详细debug余弦相似度计算
                print(f"DEBUG: Token {i} similarities before nan_to_num: {similarities}")
                print(f"DEBUG: Token {i} similarities has NaN: {torch.isnan(similarities).any()}")
                print(f"DEBUG: Token {i} similarities has Inf: {torch.isinf(similarities).any()}")
            
            # 将NaN替换为0
            similarities = torch.nan_to_num(similarities, nan=0.0)
            
            if debug and i < 3:
                print(f"DEBUG: Token {i} similarities after nan_to_num: {similarities}")
            
            # 取最大相似度作为该token的对齐分数
            max_similarity = similarities.max().item()
            alignment_scores.append(max_similarity)
            
            if debug and i < 3:  # 只打印前几个token的详细信息
                print(f"DEBUG: Token {i}: max_similarity = {max_similarity:.4f}")
        
        # 4. 计算最终GASS分数: GASS = (1/n) * Σ_{i=1}^n align(ti)
        final_gass_score = float(np.mean(alignment_scores))
        
        # 确保分数在[0,1]范围内
        final_gass_score = max(0.0, min(1.0, final_gass_score))
        
        if debug:
            print(f"DEBUG: Number of tokens: {len(alignment_scores)}")
            print(f"DEBUG: Alignment scores range: {min(alignment_scores):.4f} - {max(alignment_scores):.4f}")
            print(f"DEBUG: Final GASS score: {final_gass_score:.4f}")
        
        return final_gass_score
        
    except Exception as e:
        logger.error(f"Error in GASS calculation: {str(e)}")
        if debug:
            print(f"DEBUG: Error in GASS calculation: {str(e)}")
            import traceback
            traceback.print_exc()
        return 0.0


def batch_calculate_gass(
    model,
    tokenizer,
    batch_input_ids,
    batch_retrieved_subgraphs: List[List[List[str]]],
    batch_gold_subgraphs: List[List[List[str]]],
    batch_answer_positions: List[tuple],
    debug: bool = False
) -> List[float]:
    """
    批量计算GASS分数
    
    参数:
        model: 预训练的语言模型
        tokenizer: 分词器
        batch_input_ids: 批量输入序列
        batch_retrieved_subgraphs: 批量检索到的子图 (保留以保持接口兼容)
        batch_gold_subgraphs: 批量金标准子图 (用作Gold Knowledge Set)
        batch_answer_positions: 批量答案位置 [(start, end), ...]
        debug: 是否输出调试信息
        
    返回:
        GASS分数列表
    """
    scores = []
    
    for i in range(len(batch_answer_positions)):
        start_idx, end_idx = batch_answer_positions[i]
        
        score = calculate_gass(
            model=model,
            tokenizer=tokenizer,
            input_ids=batch_input_ids[i] if batch_input_ids is not None else None,
            retrieved_subgraph=batch_retrieved_subgraphs[i],
            gold_subgraph=batch_gold_subgraphs[i],
            answer_start_idx=start_idx,
            answer_end_idx=end_idx,
            debug=debug
        )
        
        scores.append(score)
    
    return scores 