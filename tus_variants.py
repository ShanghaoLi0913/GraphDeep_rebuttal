"""
TUS变体指标计算模块

实现了多种Triple Utilization Score (TUS)的变体计算方法。
这些变体用于测试不同的注意力利用度量方式，以找到更好的幻觉检测指标。

变体包括：
1. TUS-Strict: 严格匹配的TUS
2. TUS-Contrast: 对比TUS (gold vs non-gold)
3. TUS-Relative: 相对TUS (gold注意力占比)
4. TUS-Max: 最大注意力TUS
5. TUS-Weighted: 加权TUS (不同层权重)
6. TUS-Entropy: 基于熵的TUS
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from dataset_processor import build_external_context
from tus_metrics import find_triple_positions, find_exact_triple_positions

def calculate_tus_variants(
    attention_weights: torch.Tensor,
    external_context: Dict[str, List[int]],
    gold_triples: List[List[str]],
    answer_start_idx: int,
    answer_end_idx: int,
    input_ids: Optional[torch.Tensor] = None,
    tokenizer = None,
    question: Optional[str] = None,
    attention_sequence: Optional[List[torch.Tensor]] = None,
    debug: bool = False
) -> Dict[str, float]:
    """
    计算多种TUS变体来测试不同的计算方法
    
    参数:
        attention_weights: 注意力权重 shape=[num_layers, num_heads, seq_len, seq_len]
        external_context: 包含实体和关系位置信息的字典
        gold_triples: 数据集中的gold三元组列表
        answer_start_idx: 答案开始位置
        answer_end_idx: 答案结束位置
        input_ids: 输入序列的token ids
        tokenizer: 分词器
        question: 问题文本
        attention_sequence: 生成过程中每一步的注意力权重列表
        debug: 是否打印调试信息
        
    返回:
        包含各种TUS变体的字典：
        - tus_strict: 严格匹配TUS
        - tus_contrast: 对比TUS（gold vs non-gold）
        - tus_contrast_ratio: 对比TUS（gold注意力占比）
        - tus_relative: 相对TUS（gold注意力占比）
        - tus_relative_context: 相对TUS（gold注意力 / context注意力）
        - tus_precise: 精准TUS
        - tus_dynamic: 动态TUS
        - tus_max: 最大注意力TUS
        - tus_weighted: 加权TUS（不同层权重）
        - tus_entropy: 基于熵的TUS
    """
    # 参数验证
    if not gold_triples or input_ids is None or tokenizer is None:
        return {variant: 0.0 for variant in ['tus_strict', 'tus_contrast', 'tus_contrast_ratio', 'tus_relative', 'tus_relative_context', 'tus_precise', 'tus_dynamic', 'tus_max', 'tus_weighted', 'tus_entropy']}
        
    # 获取注意力矩阵的维度
    num_layers, num_heads, seq_len, _ = attention_weights.shape
    if debug:
        print(f"DEBUG VARIANTS: Attention matrix shape: {attention_weights.shape}")
    
    # 确保答案位置在序列长度范围内
    answer_start_idx = min(answer_start_idx, seq_len - 1)
    answer_end_idx = min(answer_end_idx, seq_len - 1)
    answer_positions = list(range(answer_start_idx, answer_end_idx + 1))
    
    if not answer_positions:
        if debug:
            print("DEBUG VARIANTS: No valid answer positions found")
        return {variant: 0.0 for variant in ['tus_strict', 'tus_contrast', 'tus_contrast_ratio', 'tus_relative', 'tus_relative_context', 'tus_precise', 'tus_dynamic', 'tus_max', 'tus_weighted', 'tus_entropy']}
        
    if debug:
        print(f"DEBUG VARIANTS: Answer positions: {answer_positions}")
        answer_text = tokenizer.decode(input_ids[answer_start_idx:answer_end_idx+1])
        print(f"DEBUG VARIANTS: Answer text: '{answer_text}'")
    
    # 获取gold位置（使用宽松匹配作为基础）
    gold_positions_loose = find_triple_positions(input_ids, tokenizer, gold_triples, debug=debug)
    gold_positions_strict = find_exact_triple_positions(input_ids, tokenizer, gold_triples, debug=debug)
    
    # 1. 严格匹配TUS
    tus_strict = calculate_tus_strict(attention_weights, answer_positions, gold_positions_strict, debug=debug)
    
    # 2. 对比TUS（gold vs non-gold）
    tus_contrast = calculate_tus_contrast(attention_weights, answer_positions, gold_positions_loose, seq_len, debug=debug)
    
    # 3. 对比TUS（gold注意力占比）
    tus_contrast_ratio = calculate_tus_contrast_ratio(
        attention_weights, 
        answer_positions, 
        gold_positions_loose,
        seq_len,
        debug=debug
    )
    
    # 4. 相对TUS（gold注意力占比）
    tus_relative = calculate_tus_relative(attention_weights, answer_positions, gold_positions_loose, debug=debug)
    
    # 5. 相对TUS（gold注意力 / context注意力）
    tus_relative_context = calculate_tus_relative_context(
        attention_weights, 
        answer_positions, 
        gold_positions_loose, 
        debug=debug
    )
    
    # 6. 最大注意力TUS
    tus_max = calculate_tus_max(attention_weights, answer_positions, gold_positions_loose, debug=debug)
    
    # 7. 加权TUS（不同层权重）
    tus_weighted = calculate_tus_weighted(attention_weights, answer_positions, gold_positions_loose, debug=debug)
    
    # 8. 基于熵的TUS
    tus_entropy = calculate_tus_entropy(attention_weights, answer_positions, gold_positions_loose, debug=debug)
    
    # 获取答案文本（用于精准TUS）
    answer_text = tokenizer.decode(input_ids[answer_start_idx:answer_end_idx+1])
    
    # 添加精准TUS
    tus_precise = 0.0
    if question is not None:  # 只在提供问题时计算精准TUS
        tus_precise = calculate_tus_precise(
            attention_weights,
            question,
            answer_text,
            answer_positions,
            gold_triples,
            input_ids,
            tokenizer,
            debug=debug
        )
    
    # 添加动态TUS
    tus_dynamic = 0.0
    if attention_sequence is not None:  # 只在提供attention_sequence时计算动态TUS
        tus_dynamic = calculate_tus_dynamic(
            attention_sequence,
            answer_positions,
            gold_positions_loose,
            debug=debug
        )
    
    results = {
        'tus_strict': tus_strict,
        'tus_contrast': tus_contrast,
        'tus_contrast_ratio': tus_contrast_ratio,
        'tus_relative': tus_relative,
        'tus_relative_context': tus_relative_context,
        'tus_precise': tus_precise,
        'tus_dynamic': tus_dynamic,
        'tus_max': tus_max,
        'tus_weighted': tus_weighted,
        'tus_entropy': tus_entropy
    }
    
    if debug:
        print(f"DEBUG VARIANTS: TUS Variants Results:")
        for variant, score in results.items():
            print(f"  {variant}: {score:.4f}")
    
    return results

def calculate_tus_strict(attention_weights, answer_positions, gold_positions, debug=False):
    """
    计算严格匹配TUS：只使用精确匹配的gold位置
    """
    if not gold_positions:
        if debug:
            print("DEBUG TUS-Strict: No gold positions found")
        return 0.0
    
    num_layers, num_heads, seq_len, _ = attention_weights.shape
    total_attention_score = 0.0
    valid_count = 0
    
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
                    valid_count += 1
    
    if valid_count == 0:
        return 0.0
    
    strict_score = total_attention_score / valid_count
    
    if debug:
        print(f"DEBUG TUS-Strict: Score={strict_score:.4f}, Valid count={valid_count}")
    
    return strict_score

def calculate_tus_contrast(attention_weights, answer_positions, gold_positions, seq_len, debug=False):
    """
    计算对比TUS：gold注意力 - non-gold注意力
    这个指标应该在正确回答时为正，幻觉时为负或接近0
    """
    if not gold_positions:
        return 0.0
    
    num_layers, num_heads, _, _ = attention_weights.shape
    gold_attention_sum = 0.0
    non_gold_attention_sum = 0.0
    valid_count = 0
    
    # 创建non-gold位置集合
    all_positions = set(range(seq_len))
    non_gold_positions = all_positions - gold_positions
    
    for layer in range(num_layers):
        for head in range(num_heads):
            layer_attention = attention_weights[layer, head]
            
            for ans_pos in answer_positions:
                if ans_pos >= seq_len:
                    continue
                
                attention_scores = layer_attention[ans_pos]
                attention_probs = F.softmax(attention_scores, dim=-1)
                
                # 计算gold位置注意力
                valid_gold_positions = [pos for pos in gold_positions if pos < seq_len]
                if valid_gold_positions:
                    gold_attention = attention_probs[valid_gold_positions].sum().item()
                    gold_attention_sum += gold_attention
                
                # 计算non-gold位置注意力
                valid_non_gold_positions = [pos for pos in non_gold_positions if pos < seq_len]
                if valid_non_gold_positions:
                    non_gold_attention = attention_probs[valid_non_gold_positions].sum().item()
                    non_gold_attention_sum += non_gold_attention
                
                valid_count += 1
    
    if valid_count == 0:
        return 0.0
    
    avg_gold_attention = gold_attention_sum / valid_count
    avg_non_gold_attention = non_gold_attention_sum / valid_count
    
    # 对比分数：正值表示更多注意力在gold位置
    contrast_score = avg_gold_attention - avg_non_gold_attention
    
    if debug:
        print(f"DEBUG TUS-Contrast: Gold={avg_gold_attention:.4f}, Non-gold={avg_non_gold_attention:.4f}, Contrast={contrast_score:.4f}")
    
    return contrast_score

def calculate_tus_contrast_ratio(attention_weights, answer_positions, gold_positions, seq_len, debug=False):
    """
    计算对比TUS：gold注意力占比
    理论依据：重要的不是看了多少gold信息，而是相比irrelevant信息更偏向gold
    返回值范围：[0,1]，0.5为中性值，>0.5表示更偏向gold，<0.5表示更偏向distractor
    """
    if not gold_positions:
        return 0.5  # 中性值
    
    num_layers, num_heads, _, _ = attention_weights.shape
    gold_attention_sum = 0.0
    distractor_attention_sum = 0.0
    valid_count = 0
    
    # 创建distractor位置集合（除了gold位置和special tokens的其他位置）
    context_start = 1  # 跳过[CLS]
    context_end = seq_len - 1  # 跳过[SEP]
    context_positions = set(range(context_start, context_end))
    distractor_positions = context_positions - set(gold_positions)
    
    for layer in range(num_layers):
        for head in range(num_heads):
            layer_attention = attention_weights[layer, head]
            
            for ans_pos in answer_positions:
                if ans_pos >= seq_len:
                    continue
                
                attention_scores = layer_attention[ans_pos]
                attention_probs = F.softmax(attention_scores, dim=-1)
                
                # 计算gold位置注意力
                valid_gold_positions = [pos for pos in gold_positions if pos < seq_len]
                if valid_gold_positions:
                    gold_attention = attention_probs[valid_gold_positions].sum().item()
                    gold_attention_sum += gold_attention
                
                # 计算distractor位置注意力
                valid_distractor_positions = [pos for pos in distractor_positions if pos < seq_len]
                if valid_distractor_positions:
                    distractor_attention = attention_probs[valid_distractor_positions].sum().item()
                    distractor_attention_sum += distractor_attention
                
                valid_count += 1
    
    if valid_count == 0:
        return 0.5  # 中性值
    
    avg_gold_attention = gold_attention_sum / valid_count
    avg_distractor_attention = distractor_attention_sum / valid_count
    total_attention = avg_gold_attention + avg_distractor_attention
    
    # 避免除零，返回中性值
    if total_attention == 0:
        return 0.5
    
    # 计算gold注意力占比
    contrast_ratio = avg_gold_attention / total_attention
    
    if debug:
        print(f"DEBUG TUS-Contrast-Ratio: Gold={avg_gold_attention:.4f}, Distractor={avg_distractor_attention:.4f}, Ratio={contrast_ratio:.4f}")
    
    return contrast_ratio

def calculate_tus_relative(attention_weights, answer_positions, gold_positions, debug=False):
    """
    计算相对TUS：gold注意力 / 总注意力
    返回gold位置获得的注意力比例
    """
    if not gold_positions:
        return 0.0
    
    num_layers, num_heads, seq_len, _ = attention_weights.shape
    gold_attention_sum = 0.0
    valid_count = 0
    
    for layer in range(num_layers):
        for head in range(num_heads):
            layer_attention = attention_weights[layer, head]
            
            for ans_pos in answer_positions:
                if ans_pos >= seq_len:
                    continue
                
                attention_scores = layer_attention[ans_pos]
                attention_probs = F.softmax(attention_scores, dim=-1)
                
                # 计算gold位置注意力（总注意力为1，所以这就是比例）
                valid_gold_positions = [pos for pos in gold_positions if pos < seq_len]
                if valid_gold_positions:
                    gold_attention = attention_probs[valid_gold_positions].sum().item()
                    gold_attention_sum += gold_attention
                    valid_count += 1
    
    if valid_count == 0:
        return 0.0
    
    relative_score = gold_attention_sum / valid_count
    
    if debug:
        print(f"DEBUG TUS-Relative: Gold attention proportion={relative_score:.4f}")
    
    return relative_score

def calculate_tus_max(attention_weights, answer_positions, gold_positions, debug=False):
    """
    计算最大TUS：使用最大注意力而不是总和
    关注最相关的gold位置
    """
    if not gold_positions:
        return 0.0
    
    num_layers, num_heads, seq_len, _ = attention_weights.shape
    max_attention_scores = []
    
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
                    # 使用最大注意力而不是总和
                    max_gold_attention = attention_probs[valid_gold_positions].max().item()
                    max_attention_scores.append(max_gold_attention)
    
    if not max_attention_scores:
        return 0.0
    
    max_score = np.mean(max_attention_scores)
    
    if debug:
        print(f"DEBUG TUS-Max: Max attention range: {min(max_attention_scores):.4f} - {max(max_attention_scores):.4f}, Mean: {max_score:.4f}")
    
    return max_score

def calculate_tus_weighted(attention_weights, answer_positions, gold_positions, debug=False):
    """
    计算加权TUS：不同层给予不同权重
    后面的层（更接近输出）权重更高
    """
    if not gold_positions:
        return 0.0
    
    num_layers, num_heads, seq_len, _ = attention_weights.shape
    
    # 层权重：线性增长，后面的层权重更高
    layer_weights = torch.linspace(0.5, 2.0, num_layers)
    layer_weights = layer_weights / layer_weights.sum()  # 归一化
    
    weighted_attention_sum = 0.0
    total_weight = 0.0
    
    for layer in range(num_layers):
        layer_weight = layer_weights[layer].item()
        
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
                    weighted_attention_sum += gold_attention * layer_weight
                    total_weight += layer_weight
    
    if total_weight == 0:
        return 0.0
    
    weighted_score = weighted_attention_sum / total_weight
    
    if debug:
        print(f"DEBUG TUS-Weighted: Weighted score={weighted_score:.4f}")
    
    return weighted_score

def calculate_tus_entropy(attention_weights, answer_positions, gold_positions, debug=False):
    """
    计算基于熵的TUS：考虑注意力分布的集中度
    较低的熵表示注意力更集中在特定位置
    """
    if not gold_positions:
        return 0.0
    
    num_layers, num_heads, seq_len, _ = attention_weights.shape
    entropy_weighted_scores = []
    
    for layer in range(num_layers):
        for head in range(num_heads):
            layer_attention = attention_weights[layer, head]
            
            for ans_pos in answer_positions:
                if ans_pos >= seq_len:
                    continue
                
                attention_scores = layer_attention[ans_pos]
                attention_probs = F.softmax(attention_scores, dim=-1)
                
                # 计算注意力分布的熵
                entropy = -torch.sum(attention_probs * torch.log(attention_probs + 1e-10)).item()
                
                # 计算gold位置注意力
                valid_gold_positions = [pos for pos in gold_positions if pos < seq_len]
                if valid_gold_positions:
                    gold_attention = attention_probs[valid_gold_positions].sum().item()
                    
                    # 熵加权：较低的熵（更集中的注意力）权重更高
                    max_entropy = np.log(seq_len)  # 最大可能熵
                    normalized_entropy = entropy / max_entropy
                    concentration_weight = 1.0 - normalized_entropy  # 越集中权重越高
                    
                    entropy_weighted_score = gold_attention * concentration_weight
                    entropy_weighted_scores.append(entropy_weighted_score)
    
    if not entropy_weighted_scores:
        return 0.0
    
    entropy_score = np.mean(entropy_weighted_scores)
    
    if debug:
        print(f"DEBUG TUS-Entropy: Entropy-weighted score={entropy_score:.4f}")
    
    return entropy_score

def sum_attention_to_positions(attention_weights, positions, answer_positions):
    """计算特定位置获得的注意力总和"""
    if not positions:
        return 0.0
        
    num_layers, num_heads, seq_len, _ = attention_weights.shape
    attention_sum = 0.0
    valid_count = 0
    
    for layer in range(num_layers):
        for head in range(num_heads):
            layer_attention = attention_weights[layer, head]
            
            for ans_pos in answer_positions:
                if ans_pos >= seq_len:
                    continue
                    
                attention_scores = layer_attention[ans_pos]
                attention_probs = F.softmax(attention_scores, dim=-1)
                
                valid_positions = [pos for pos in positions if pos < seq_len]
                if valid_positions:
                    pos_attention = attention_probs[valid_positions].sum().item()
                    attention_sum += pos_attention
                    valid_count += 1
    
    if valid_count == 0:
        return 0.0
        
    return attention_sum / valid_count

def calculate_tus_relative_context(attention_weights, answer_positions, gold_positions, debug=False):
    """
    计算相对TUS：gold注意力 / context注意力
    这个版本明确使用context位置的注意力作为分母
    """
    if not gold_positions:
        return 0.0
    
    num_layers, num_heads, seq_len, _ = attention_weights.shape
    
    # 获取所有context位置(除了special tokens和padding)
    # 这里假设special tokens在序列开始,padding在序列结尾
    # 可以根据实际情况调整
    context_start = 1  # 跳过第一个token(通常是[CLS]或类似的special token)
    context_end = seq_len - 1  # 跳过最后一个token(通常是[SEP]或padding)
    all_context_positions = set(range(context_start, context_end))
    
    # 计算gold位置的注意力
    gold_attention = sum_attention_to_positions(attention_weights, gold_positions, answer_positions)
    
    # 计算所有context位置的注意力
    total_context_attention = sum_attention_to_positions(attention_weights, all_context_positions, answer_positions)
    
    # 避免除零
    if total_context_attention == 0:
        return 0.0
    
    relative_score = gold_attention / total_context_attention
    
    if debug:
        print(f"DEBUG TUS-Relative-Context: Gold attention={gold_attention:.4f}, Context attention={total_context_attention:.4f}, Ratio={relative_score:.4f}")
    
    return relative_score

def identify_answer_relevant_triples(question: str, answer_text: str, gold_triples: List[List[str]], debug: bool = False) -> Set[Tuple[str, str, str]]:
    """
    基于问题和答案识别最相关的gold triples
    
    策略：
    1. 如果问题是关于演员的电影，那么包含该演员的triple更相关
    2. 如果问题是关于电影的演员，那么包含该电影的triple更相关
    3. 如果问题包含特定关系（如"导演"），那么包含该关系的triple更相关
    """
    # 将问题和答案转换为小写以进行匹配
    question = question.lower()
    answer_text = answer_text.lower()
    
    # 获取问题中的关键词
    question_keywords = set(question.split())
    answer_keywords = set(answer_text.split())
    
    relevant_positions = set()
    for triple in gold_triples:
        # 将triple转换为小写
        triple = tuple(t.lower() for t in triple)
        
        # 检查triple是否包含答案中的任何词
        if any(keyword in ' '.join(triple) for keyword in answer_keywords):
            if debug:
                print(f"DEBUG Precise-TUS: Triple {triple} contains answer keywords")
            relevant_positions.add(triple)
            continue
            
        # 检查triple是否包含问题中的关键词
        if any(keyword in ' '.join(triple) for keyword in question_keywords):
            if debug:
                print(f"DEBUG Precise-TUS: Triple {triple} contains question keywords")
            relevant_positions.add(triple)
            continue
    
    if debug:
        print(f"DEBUG Precise-TUS: Found {len(relevant_positions)} relevant triples out of {len(gold_triples)}")
        
    return relevant_positions

def calculate_tus_precise(
    attention_weights: torch.Tensor,
    question: str,
    answer_text: str,
    answer_positions: List[int],
    gold_triples: List[List[str]],
    input_ids: torch.Tensor,
    tokenizer,
    debug: bool = False
) -> float:
    """
    计算精准TUS：只关注与答案直接相关的gold triples
    
    理论依据：
    1. 不是所有gold triples对回答问题都同等重要
    2. 与问题和答案直接相关的triple更重要
    3. 模型应该更关注这些相关的triple
    """
    if not gold_triples:
        return 0.0
        
    # 识别相关的gold triples
    relevant_triples = identify_answer_relevant_triples(question, answer_text, gold_triples, debug=debug)
    if not relevant_triples:
        if debug:
            print("DEBUG Precise-TUS: No relevant triples found")
        return 0.0
    
    # 获取相关triple的位置
    relevant_positions = set()
    for triple in relevant_triples:
        # 将元组转换回列表
        triple_list = list(triple)
        positions = find_triple_positions(input_ids, tokenizer, [triple_list], debug=debug)
        relevant_positions.update(positions)
    
    if not relevant_positions:
        if debug:
            print("DEBUG Precise-TUS: No valid positions found for relevant triples")
        return 0.0
    
    # 获取所有context位置
    num_layers, num_heads, seq_len, _ = attention_weights.shape
    context_start = 1  # 跳过[CLS]
    context_end = seq_len - 1  # 跳过[SEP]
    all_positions = set(range(context_start, context_end))
    
    # 计算注意力分数
    relevant_attention = sum_attention_to_positions(attention_weights, relevant_positions, answer_positions)
    total_attention = sum_attention_to_positions(attention_weights, all_positions, answer_positions)
    
    # 避免除零
    if total_attention == 0:
        return 0.0
        
    precise_score = relevant_attention / total_attention
    
    if debug:
        print(f"DEBUG Precise-TUS: Relevant attention={relevant_attention:.4f}, Total attention={total_attention:.4f}, Score={precise_score:.4f}")
        
    return precise_score

def calculate_trend_slope(attention_scores: List[float], debug: bool = False) -> float:
    """
    使用线性回归计算注意力分数的趋势斜率
    
    参数:
        attention_scores: 每个生成步骤的注意力分数列表
        debug: 是否打印调试信息
        
    返回:
        斜率值：正值表示注意力增加，负值表示注意力减少
    """
    if len(attention_scores) < 2:
        return 0.0
        
    # 使用numpy的polyfit进行线性回归
    x = np.arange(len(attention_scores))
    y = np.array(attention_scores)
    slope, _ = np.polyfit(x, y, 1)
    
    if debug:
        print(f"DEBUG Trend-Slope: Attention scores={attention_scores}")
        print(f"DEBUG Trend-Slope: Slope={slope:.4f}")
        
    return float(slope)

def calculate_tus_dynamic(
    attention_sequence: List[torch.Tensor],
    answer_positions: List[int],
    gold_positions: Set[int],
    debug: bool = False
) -> float:
    """
    计算动态TUS：跟踪生成过程中注意力的变化趋势
    
    理论依据：
    1. 生成过程中注意力应该越来越聚焦在相关的gold信息上
    2. 注意力趋势的斜率反映了模型对gold信息的利用程度
    3. 正斜率表示模型越来越关注gold信息，负斜率表示模型逐渐忽视gold信息
    
    参数:
        attention_sequence: 生成过程中每一步的注意力权重列表
        answer_positions: 答案token的位置列表
        gold_positions: gold triple的位置集合
        debug: 是否打印调试信息
        
    返回:
        注意力趋势的斜率
    """
    if not gold_positions or not attention_sequence:
        return 0.0
        
    attention_scores = []
    for step, attention_weights in enumerate(attention_sequence):
        # 计算当前步骤的注意力分数
        step_score = sum_attention_to_positions(attention_weights, gold_positions, answer_positions)
        attention_scores.append(step_score)
        
        if debug:
            print(f"DEBUG Dynamic-TUS: Step {step}, Score={step_score:.4f}")
    
    # 计算趋势斜率
    slope = calculate_trend_slope(attention_scores, debug=debug)
    
    if debug:
        print(f"DEBUG Dynamic-TUS: Final slope={slope:.4f}")
        if slope > 0:
            print("DEBUG Dynamic-TUS: Positive trend - model increasingly focuses on gold information")
        else:
            print("DEBUG Dynamic-TUS: Negative trend - model gradually ignores gold information")
    
    return slope 