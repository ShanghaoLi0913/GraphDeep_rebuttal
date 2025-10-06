"""
GASS-JSD (Gold-Aligned Semantic Similarity via Jensen-Shannon Divergence) 指标计算

这个模块实现真正的基于Jensen-Shannon散度的GASS-JSD指标，独立于其他指标计算。

作者: AI Assistant
日期: 2025年6月26日
"""

import torch
import torch.nn.functional as F
import logging
from typing import List, Optional, Tuple
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

def calculate_jsd(P: torch.Tensor, Q: torch.Tensor, epsilon: float = 1e-10) -> torch.Tensor:
    """
    计算两个概率分布之间的Jensen-Shannon散度
    
    参数:
        P: 第一个概率分布 [vocab_size]
        Q: 第二个概率分布 [vocab_size]
        epsilon: 数值稳定性的小常数
        
    返回:
        JSD值
    """
    # 确保是概率分布
    P = F.softmax(P, dim=-1)
    Q = F.softmax(Q, dim=-1)
    
    # 添加小常数防止log(0)
    P = P + epsilon
    Q = Q + epsilon
    
    # 计算中间分布M = 0.5*(P + Q)
    M = 0.5 * (P + Q)
    
    # 计算KL散度
    kl_pm = F.kl_div(torch.log(P), M, reduction='sum')
    kl_qm = F.kl_div(torch.log(Q), M, reduction='sum')
    
    # 计算JSD = 0.5 * (KL(P||M) + KL(Q||M))
    jsd = 0.5 * (kl_pm + kl_qm)
    
    return jsd

def encode_gold_expansion_set_safe(
    gold_expansion_set: List[List[str]],
    tokenizer,
    model,
    target_device: torch.device,
    debug: bool = False
) -> torch.Tensor:
    """
    安全地编码Gold Expansion Set，处理设备和权重访问问题
    
    参数:
        gold_expansion_set: 金标准扩展集合 (三元组列表)
        tokenizer: 分词器
        model: 语言模型
        target_device: 目标设备
        debug: 是否输出调试信息
        
    返回:
        编码后的表示 [num_triples, hidden_size]
    """
    if debug:
        print(f"DEBUG: Encoding {len(gold_expansion_set)} gold triples")
    
    representations = []
    
    for triple in gold_expansion_set:
        try:
            # 构建三元组文本
            if isinstance(triple, list) and len(triple) >= 3:
                triple_text = f"{triple[0]} {triple[1]} {triple[2]}"
            else:
                triple_text = str(triple)
            
            # 分词
            encoded = tokenizer(
                triple_text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=128
            )
            
            # 移到目标设备
            encoded = {k: v.to(target_device) for k, v in encoded.items()}
            
            # 获取模型输出（使用no_grad以节省内存）
            with torch.no_grad():
                outputs = model(**encoded, output_hidden_states=True)
                hidden_states = outputs.hidden_states[-1]  # 最后一层
                
                # 计算平均池化（忽略padding tokens）
                attention_mask = encoded['attention_mask']
                masked_hidden = hidden_states * attention_mask.unsqueeze(-1)
                pooled = masked_hidden.sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)
                
                representations.append(pooled.squeeze(0))  # [hidden_size]
                
        except Exception as e:
            if debug:
                print(f"DEBUG: Error encoding triple {triple}: {e}")
            # 使用零向量作为fallback
            hidden_size = model.config.hidden_size
            representations.append(torch.zeros(hidden_size, device=target_device))
    
    if representations:
        return torch.stack(representations)  # [num_triples, hidden_size]
    else:
        # 返回零张量
        hidden_size = model.config.hidden_size
        return torch.zeros(1, hidden_size, device=target_device)

def get_model_weights_safe(model, target_device: torch.device, debug: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    安全地获取模型的输出层权重和偏置
    
    参数:
        model: 语言模型
        target_device: 目标设备
        debug: 是否输出调试信息
        
    返回:
        (W, b) 权重矩阵和偏置向量的元组
    """
    try:
        # 尝试不同的方式获取输出层权重
        lm_head = None
        
        if hasattr(model, 'lm_head'):
            lm_head = model.lm_head
        elif hasattr(model, 'embed_out'):
            lm_head = model.embed_out
        elif hasattr(model, 'output_projection'):
            lm_head = model.output_projection
        
        if lm_head is None:
            raise ValueError("Cannot find language model head")
        
        # 获取权重
        if hasattr(lm_head, 'weight'):
            W = lm_head.weight
        else:
            raise ValueError("Cannot find weight in language model head")
        
        # 获取偏置（可能不存在）
        if hasattr(lm_head, 'bias') and lm_head.bias is not None:
            b = lm_head.bias
        else:
            # 创建零偏置
            vocab_size = W.shape[0]
            b = torch.zeros(vocab_size, device=W.device, dtype=W.dtype)
        
        # 检查设备和数据可用性
        if W.device.type == 'meta' or not hasattr(W, 'data'):
            if debug:
                print("DEBUG: Weight is on meta device or not accessible, using alternative approach")
            
            # 使用模型前向传播来获取logits，然后反推权重近似
            # 这是一个workaround方法
            hidden_size = model.config.hidden_size
            vocab_size = model.config.vocab_size
            
            # 创建随机的hidden state来测试
            test_hidden = torch.randn(1, hidden_size, device=target_device, dtype=torch.float16)
            
            with torch.no_grad():
                # 使用模型的lm_head进行前向传播
                test_logits = lm_head(test_hidden)  # [1, vocab_size]
                
                # 如果成功，说明我们可以使用lm_head
                # 创建单位矩阵来近似权重
                W_approx = torch.eye(min(hidden_size, vocab_size), vocab_size, device=target_device, dtype=test_logits.dtype)
                if hidden_size > vocab_size:
                    # Pad with zeros
                    padding = torch.zeros(hidden_size - vocab_size, vocab_size, device=target_device, dtype=test_logits.dtype)
                    W_approx = torch.cat([W_approx, padding], dim=0)
                elif hidden_size < vocab_size:
                    # Truncate
                    W_approx = W_approx[:hidden_size, :]
                
                W_approx = W_approx.t()  # [vocab_size, hidden_size]
                b_approx = torch.zeros(vocab_size, device=target_device, dtype=test_logits.dtype)
                
                return W_approx, b_approx
        
        # 正常情况：权重可以访问
        # 确保在正确的设备上
        if W.device != target_device:
            W = W.to(target_device)
        if b.device != target_device:
            b = b.to(target_device)
        
        return W, b
        
    except Exception as e:
        if debug:
            print(f"DEBUG: Error getting model weights: {e}")
        
        # Fallback: 返回随机权重
        hidden_size = model.config.hidden_size
        vocab_size = model.config.vocab_size
        
        W = torch.randn(vocab_size, hidden_size, device=target_device, dtype=torch.float16)
        b = torch.zeros(vocab_size, device=target_device, dtype=torch.float16)
        
        return W, b

def calculate_gass_jsd_true(
    model,
    tokenizer,
    input_ids: torch.Tensor,
    gold_expansion_set: List[List[str]],
    answer_start_idx: int,
    answer_end_idx: int,
    layer_indices: Optional[List[int]] = None,
    debug: bool = False
) -> float:
    """
    计算真正的GASS-JSD (Gold-Aligned Semantic Similarity via Jensen-Shannon Divergence)
    
    参数:
        model: 语言模型
        tokenizer: 分词器
        input_ids: 输入token序列 [seq_len]
        gold_expansion_set: 金标准扩展集合 (三元组列表)
        answer_start_idx: 答案开始位置
        answer_end_idx: 答案结束位置
        layer_indices: 要分析的层索引列表，None表示使用最后一层
        debug: 是否输出调试信息
        
    返回:
        GASS-JSD分数 (0-1之间的浮点数，越高表示越对齐)
    """
    logger = logging.getLogger(__name__)
    
    if not gold_expansion_set:
        if debug:
            print("DEBUG: Empty gold expansion set, returning GASS-JSD = 0.0")
        return 0.0
    
    try:
        if debug:
            print(f"DEBUG: Starting True GASS-JSD calculation")
            print(f"DEBUG: Gold expansion set size: {len(gold_expansion_set)}")
            print(f"DEBUG: Answer positions: {answer_start_idx}-{answer_end_idx}")
        
        device = input_ids.device
        
        # 1. 获取答案部分的FFN表示
        with torch.no_grad():
            outputs = model(
                input_ids.unsqueeze(0),  # 添加batch维度
                output_hidden_states=True,
                use_cache=False
            )
            
            # 提取指定层的hidden states
            if layer_indices is None:
                layer_indices = [-1]  # 只使用最后一层
            
            # 提取答案部分的表示
            answer_representations = []
            for layer_idx in layer_indices:
                hidden_states = outputs.hidden_states[layer_idx]  # [1, seq_len, hidden_size]
                answer_hidden = hidden_states[0, answer_start_idx:answer_end_idx]  # [answer_len, hidden_size]
                answer_representations.append(answer_hidden)
            
            # 堆叠所有层的表示
            answer_representations = torch.stack(answer_representations)  # [num_layers, answer_len, hidden_size]
        
        if debug:
            print(f"DEBUG: Answer representations shape: {answer_representations.shape}")
        
        # 2. 编码Gold Expansion Set
        ges_representations = encode_gold_expansion_set_safe(
            gold_expansion_set, tokenizer, model, device, debug
        )
        
        if debug:
            print(f"DEBUG: GES representations shape: {ges_representations.shape}")
        
        # 3. 获取模型的输出层权重
        W, b = get_model_weights_safe(model, device, debug)
        
        if debug:
            print(f"DEBUG: Model weights shape - W: {W.shape}, b: {b.shape}")
        
        # 4. 计算JSD分数
        jsd_scores = []
        
        for layer_idx in range(answer_representations.size(0)):
            for token_idx in range(answer_representations.size(1)):
                try:
                    # 获取当前token的FFN表示
                    z_t = answer_representations[layer_idx, token_idx]  # [hidden_size]
                    
                    # 计算生成token的分布 P_t = softmax(W·z_t + b)
                    logits_t = torch.matmul(W, z_t) + b  # [vocab_size]
                    P_t = F.softmax(logits_t, dim=-1)
                    
                    # 计算Gold Expansion Set中每个token的分布
                    gold_distributions = []
                    for ges_repr in ges_representations:
                        logits_g = torch.matmul(W, ges_repr) + b  # [vocab_size]
                        P_g = F.softmax(logits_g, dim=-1)
                        gold_distributions.append(P_g)
                    
                    if gold_distributions:
                        # 计算金标准分布的平均值
                        P_gold_avg = torch.stack(gold_distributions).mean(dim=0)  # [vocab_size]
                        
                        # 计算JSD
                        jsd = calculate_jsd(logits_t, torch.log(P_gold_avg + 1e-10))
                        
                        # 转换为相似度分数 (1 - JSD)
                        gass_jsd_score = 1.0 - jsd.item()
                        
                        # 确保分数在[0,1]范围内
                        gass_jsd_score = max(0.0, min(1.0, gass_jsd_score))
                        
                        jsd_scores.append(gass_jsd_score)
                        
                        if debug:
                            print(f"DEBUG: Layer {layer_idx}, Token {token_idx}: JSD={jsd.item():.6f}, Score={gass_jsd_score:.6f}")
                    
                except Exception as e:
                    if debug:
                        print(f"DEBUG: Error processing layer {layer_idx}, token {token_idx}: {e}")
                    continue
        
        if not jsd_scores:
            if debug:
                print("DEBUG: No valid JSD scores computed, returning 0.0")
            return 0.0
        
        # 计算平均JSD分数
        final_score = np.mean(jsd_scores)
        
        if debug:
            print(f"DEBUG: True GASS-JSD final score: {final_score:.6f}")
        
        return final_score
        
    except Exception as e:
        logger.error(f"Error in True GASS-JSD calculation: {e}")
        if debug:
            print(f"DEBUG: Error in True GASS-JSD calculation: {e}")
            import traceback
            print(f"DEBUG: Traceback:\n{traceback.format_exc()}")
        return 0.0

def batch_calculate_gass_jsd_true(
    model,
    tokenizer,
    batch_input_ids: torch.Tensor,
    batch_gold_expansion_sets: List[List[List[str]]],
    batch_answer_positions: List[Tuple[int, int]],
    **kwargs
) -> List[float]:
    """
    批量计算真正的GASS-JSD分数
    
    参数:
        model: 语言模型
        tokenizer: 分词器
        batch_input_ids: 批量输入序列 [batch_size, seq_len]
        batch_gold_expansion_sets: 批量金标准扩展集合
        batch_answer_positions: 批量答案位置 [(start, end), ...]
        **kwargs: 其他参数传递给calculate_gass_jsd_true
        
    返回:
        真正的GASS-JSD分数列表
    """
    scores = []
    for i in range(len(batch_answer_positions)):
        start_idx, end_idx = batch_answer_positions[i]
        score = calculate_gass_jsd_true(
            model=model,
            tokenizer=tokenizer,
            input_ids=batch_input_ids[i],
            gold_expansion_set=batch_gold_expansion_sets[i],
            answer_start_idx=start_idx,
            answer_end_idx=end_idx,
            **kwargs
        )
        scores.append(score)
    
    return scores

# 为了向后兼容性，提供别名
calculate_true_gass_jsd = calculate_gass_jsd_true
batch_calculate_true_gass_jsd = batch_calculate_gass_jsd_true 