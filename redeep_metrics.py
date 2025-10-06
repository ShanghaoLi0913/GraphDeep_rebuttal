"""
指标计算模块

实现了基于Llama-2的Triple Utilization Score (TUS)的计算。
TUS用于衡量模型在生成答案时对输入三元组的利用程度。
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from dataset_processor import build_external_context

def find_triple_positions(
    input_ids: torch.Tensor,
    tokenizer,
    gold_triples: List[List[str]],
    max_span_length: int = 20,
    debug: bool = False
) -> Set[int]:
    """
    在输入序列中找出gold三元组对应的token位置
    使用宽松匹配策略：
    - 双向包含匹配 (span_text in h or h in span_text)
    - 更高的召回率，但可能有一些噪音
    
    当前TUS/nTUS计算使用此函数
    
    参数:
        input_ids: 输入序列的token ids
        tokenizer: 分词器
        gold_triples: 数据集中的gold三元组列表
        max_span_length: 最大span长度
        debug: 是否打印调试信息
        
    返回:
        包含所有gold三元组token位置的集合
    """
    seq_len = len(input_ids)
    gold_positions = set()
    
    if debug:
        print(f"\nDEBUG: Gold triples: {gold_triples}")
    
    # 遍历输入序列中的每个可能的span
    for start_pos in range(seq_len - 1):
        for span_length in range(1, min(max_span_length, seq_len - start_pos)):
            end_pos = start_pos + span_length
            span_text = tokenizer.decode(input_ids[start_pos:end_pos]).lower()
            
            # 检查这个span是否匹配任何gold三元组的组成部分
            for h, r, t in gold_triples:
                h, r, t = h.lower(), r.lower(), t.lower()
                if (span_text in h or h in span_text or
                    span_text in r or r in span_text or
                    span_text in t or t in span_text):
                    gold_positions.update(range(start_pos, end_pos))
                    break
    
    if debug:
        print(f"\nDEBUG: Found {len(gold_positions)} gold positions: {sorted(list(gold_positions))}")
    
    return gold_positions

def calculate_attention_scores(
    attention_weights: torch.Tensor,
    answer_positions: List[int],
    gold_positions: Set[int],
    debug: bool = False
) -> Tuple[float, int]:
    """
    计算注意力分数
    
    参数:
        attention_weights: 注意力权重
        answer_positions: 答案token的位置列表
        gold_positions: gold三元组token的位置集合
        debug: 是否打印调试信息
        
    返回:
        (total_attention_score, valid_attention_count)元组
    """
    num_layers, num_heads, seq_len, _ = attention_weights.shape
    total_attention_score = 0.0
    valid_attention_count = 0
    
    if debug:
        print(f"\nDEBUG: Checking attention scores for answer positions {answer_positions} against gold positions {sorted(list(gold_positions))}")
    
    for layer in range(num_layers):
        for head in range(num_heads):
            layer_attention = attention_weights[layer, head]
            
            for ans_pos in answer_positions:
                if ans_pos >= seq_len:
                    continue
                
                attention_scores = layer_attention[ans_pos]
                attention_probs = F.softmax(attention_scores, dim=-1)
                
                valid_gold_positions = [pos for pos in gold_positions if pos < seq_len]
                if valid_gold_positions:
                    gold_attention = attention_probs[valid_gold_positions].sum().item()
                    total_attention_score += gold_attention
                    valid_attention_count += 1
    
    return total_attention_score, valid_attention_count

def calculate_tus(
    attention_weights: torch.Tensor,
    external_context: Dict[str, List[int]],
    gold_triples: List[List[str]],
    answer_start_idx: int,
    answer_end_idx: int,
    input_ids: Optional[torch.Tensor] = None,
    tokenizer = None,
    debug: bool = False
) -> float:
    """
    计算Triple Utilization Score (TUS)
    
    参数:
        attention_weights: 注意力权重 shape=[num_layers, num_heads, seq_len, seq_len]
        external_context: 包含实体和关系位置信息的字典
        gold_triples: 数据集中的gold三元组列表
        answer_start_idx: 答案开始位置
        answer_end_idx: 答案结束位置
        input_ids: 输入序列的token ids
        tokenizer: 分词器
        debug: 是否打印调试信息
    """
    # 参数验证
    if not gold_triples or input_ids is None or tokenizer is None:
        return 0.0
        
    # 获取注意力矩阵的维度
    num_layers, num_heads, seq_len, _ = attention_weights.shape
    if debug:
        print(f"DEBUG: Attention matrix shape: {attention_weights.shape}")
    
    # 确保答案位置在序列长度范围内
    answer_start_idx = min(answer_start_idx, seq_len - 1)
    answer_end_idx = min(answer_end_idx, seq_len - 1)
    answer_positions = list(range(answer_start_idx, answer_end_idx + 1))
    
    if not answer_positions:
        if debug:
            print("DEBUG: No valid answer positions found")
        return 0.0
        
    if debug:
        print(f"DEBUG: Answer positions: {answer_positions}")
        answer_text = tokenizer.decode(input_ids[answer_start_idx:answer_end_idx+1])
        print(f"DEBUG: Answer text: '{answer_text}'")
    
    # 找出gold三元组对应的token位置 (使用宽松匹配)
    gold_positions = find_triple_positions(input_ids, tokenizer, gold_triples, debug=debug)
    if not gold_positions:
        return 0.0
    
    # 计算注意力分数
    total_attention_score, valid_attention_count = calculate_attention_scores(
        attention_weights, answer_positions, gold_positions, debug=debug
    )
    
    if valid_attention_count == 0:
        if debug:
            print("DEBUG: No valid attention scores found")
        return 0.0
    
    # 计算平均注意力分数
    average_attention = total_attention_score / valid_attention_count
    if debug:
        print(f"DEBUG: Average attention score: {average_attention:.4f}")
    
    return average_attention

def find_exact_triple_positions(
    input_ids: torch.Tensor,
    tokenizer,
    triples: List[List[str]],
    debug: bool = False
) -> Set[int]:
    """
    精确找到三元组在输入序列中的token位置
    使用严格匹配策略：
    - 检查单词边界
    - 精确的字符到token对齐
    - 较高的精确度，但可能遗漏一些匹配
    
    注意：当前TUS/nTUS计算使用宽松匹配(find_triple_positions)
    此函数保留用于对比实验
    
    参数:
        input_ids: 输入序列的token ids
        tokenizer: 分词器
        triples: 三元组列表
        debug: 是否打印调试信息
        
    返回:
        包含三元组token位置的集合
    """
    seq_len = len(input_ids)
    triple_positions = set()
    
    # 将整个输入序列解码为文本
    full_text = tokenizer.decode(input_ids).lower()
    
    if debug:
        print(f"\nDEBUG EXACT: Processing {len(triples)} triples")
        print(f"DEBUG EXACT: Full text snippet: {full_text[:200]}...")
    
    for triple_idx, (h, r, t) in enumerate(triples):
        h, r, t = h.lower().strip(), r.lower().strip(), t.lower().strip()
        
        # 为每个三元组的实体和关系创建可能的表示形式
        entities_and_relations = [h, r, t]
        
        for item in entities_and_relations:
            if len(item) < 3:  # 跳过太短的词，避免误匹配
                continue
                
            # 检查item是否作为完整词出现在文本中
            if item in full_text:
                # 找到所有出现位置
                start_idx = 0
                while True:
                    pos = full_text.find(item, start_idx)
                    if pos == -1:
                        break
                    
                    # 确保是完整的词（前后是空格或标点）
                    is_complete_word = True
                    if pos > 0 and full_text[pos-1].isalnum():
                        is_complete_word = False
                    if pos + len(item) < len(full_text) and full_text[pos + len(item)].isalnum():
                        is_complete_word = False
                    
                    if is_complete_word:
                        # 将字符位置转换为token位置（近似）
                        # 这是一个简化的方法，实际应用中可能需要更精确的对齐
                        char_start = pos
                        char_end = pos + len(item)
            
                        # 估算token位置
                        token_start = int(char_start * seq_len / len(full_text))
                        token_end = int(char_end * seq_len / len(full_text))
                        
                        # 添加一些缓冲区域
                        token_start = max(0, token_start - 2)
                        token_end = min(seq_len, token_end + 2)
                        
                        # 精确验证：检查这个token范围是否真的包含目标词
                        for t_start in range(max(0, token_start-5), min(seq_len, token_end+5)):
                            for t_end in range(t_start+1, min(seq_len, token_end+10)):
                                try:
                                    decoded_span = tokenizer.decode(input_ids[t_start:t_end]).lower().strip()
                                    if item in decoded_span:
                                        triple_positions.update(range(t_start, t_end))
                                        if debug and triple_idx < 3:  # 只打印前几个的debug信息
                                            print(f"DEBUG EXACT: Found '{item}' at tokens {t_start}-{t_end}: '{decoded_span}'")
                                        break
                                except:
                                    continue
                            else:
                                continue
                            break
                    
                    start_idx = pos + 1
    
    if debug:
        print(f"DEBUG EXACT: Found {len(triple_positions)} total positions: {sorted(list(triple_positions))[:20]}...")
    
    return triple_positions

def calculate_ntus(
    attention_weights: torch.Tensor,
    external_context: Dict[str, List[int]],
    gold_triples: List[List[str]],
    trimmed_triples: List[List[str]],
    answer_start_idx: int,
    answer_end_idx: int,
    input_ids: Optional[torch.Tensor] = None,
    tokenizer = None,
    debug: bool = False
) -> float:
    """
    计算归一化的 Triple Utilization Score (NTUS)
    
    NTUS(yt) = A(yt, Gold) / (A(yt, All) + ε)
    其中:
    - A(yt, Gold): 生成token yt 对 gold triples 的注意力量
    - A(yt, All): 生成token yt 对整个子图中所有 triples 的注意力量
    - ε: 防止除零的小常数
    
    参数:
        attention_weights: 注意力权重 shape=[num_layers, num_heads, seq_len, seq_len]
        external_context: 包含实体和关系位置信息的字典
        gold_triples: 数据集中的gold三元组列表
        trimmed_triples: 输入context中的所有三元组列表
        answer_start_idx: 答案开始位置
        answer_end_idx: 答案结束位置
        input_ids: 输入序列的token ids
        tokenizer: 分词器
        debug: 是否打印调试信息
        
    返回:
        归一化的TUS分数
    """
    # 参数验证
    if not gold_triples or not trimmed_triples or input_ids is None or tokenizer is None:
        return 0.0
    
    # 获取注意力矩阵的维度
    num_layers, num_heads, seq_len, _ = attention_weights.shape
    if debug:
        print(f"DEBUG NTUS: Attention matrix shape: {attention_weights.shape}")
    
    # 确保答案位置在序列长度范围内
    answer_start_idx = min(answer_start_idx, seq_len - 1)
    answer_end_idx = min(answer_end_idx, seq_len - 1)
    answer_positions = list(range(answer_start_idx, answer_end_idx + 1))
    
    if not answer_positions:
        if debug:
            print("DEBUG NTUS: No valid answer positions found")
        return 0.0
    
    if debug:
        print(f"DEBUG NTUS: Answer positions: {answer_positions}")
    
    # 找出gold三元组对应的token位置 (使用宽松匹配)
    gold_positions = find_triple_positions(input_ids, tokenizer, gold_triples, debug=debug)
    
    # 找出所有三元组对应的token位置 (使用宽松匹配)
    all_positions = find_triple_positions(input_ids, tokenizer, trimmed_triples, debug=debug)
    
    if not gold_positions or not all_positions:
        if debug:
            print("DEBUG NTUS: No gold or all positions found")
        return 0.0
    
    if debug:
        print(f"DEBUG NTUS: Gold positions count: {len(gold_positions)}")
        print(f"DEBUG NTUS: All positions count: {len(all_positions)}")
    
    # 防止除零的小常数
    epsilon = 1e-8
    
    # 对每个答案token计算NTUS
    ntus_scores = []
    
    for layer in range(num_layers):
        for head in range(num_heads):
            layer_attention = attention_weights[layer, head]
            
            for ans_pos in answer_positions:
                if ans_pos >= seq_len:
                    continue
                    
                attention_scores = layer_attention[ans_pos]
                attention_probs = F.softmax(attention_scores, dim=-1)
                
                # 计算A(yt, Gold): 对gold triples的注意力
                valid_gold_positions = [pos for pos in gold_positions if pos < seq_len]
                if valid_gold_positions:
                    a_gold = attention_probs[valid_gold_positions].sum().item()
                else:
                    a_gold = 0.0
                
                # 计算A(yt, All): 对所有triples的注意力
                valid_all_positions = [pos for pos in all_positions if pos < seq_len]
                if valid_all_positions:
                    a_all = attention_probs[valid_all_positions].sum().item()
                else:
                    a_all = 0.0
                
                # 计算NTUS(yt) = A(yt, Gold) / (A(yt, All) + ε)
                ntus_token = a_gold / (a_all + epsilon)
                ntus_scores.append(ntus_token)
                
                if debug and len(ntus_scores) <= 3:  # 只打印前几个debug信息
                    print(f"DEBUG NTUS: Token {ans_pos}, A(Gold)={a_gold:.4f}, A(All)={a_all:.4f}, NTUS={ntus_token:.4f}")
    
    if not ntus_scores:
        if debug:
            print("DEBUG NTUS: No NTUS scores calculated")
        return 0.0
        
    # 计算所有token的平均NTUS
    average_ntus = sum(ntus_scores) / len(ntus_scores)
    
    if debug:
        print(f"DEBUG NTUS: Average NTUS score: {average_ntus:.4f} (from {len(ntus_scores)} calculations)")
    
    return average_ntus 