"""
模型工具模块 (Model Utilities)

本模块实现了模型相关的工具函数。主要功能包括：
1. 模型加载和初始化
2. 批量生成答案
3. 答案评估

主要函数:
- load_model: 加载模型
- generate_answer_batch: 批量生成答案
- is_answer_covered: 检查答案覆盖率

作者: [Your Name]
创建日期: 2024-03-19
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from typing import List, Dict, Set

def load_model(model_name: str = "meta-llama/Llama-2-7b-chat-hf") -> tuple:
    """
    加载语言模型
    
    Args:
        model_name: 模型名称
    
    Returns:
        (tokenizer, model) 元组
    """
    quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=True)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        quantization_config=quantization_config,
        use_auth_token=True
    )
    
    return tokenizer, model

def generate_answer_batch(prompts: List[str], 
                        tokenizer: AutoTokenizer, 
                        model: AutoModelForCausalLM,
                        max_new_tokens: int = 32) -> List[str]:
    """
    批量生成答案
    
    Args:
        prompts: prompt列表
        tokenizer: 分词器
        model: 语言模型
        max_new_tokens: 最大生成token数
    
    Returns:
        生成的答案列表
    """
    inputs = tokenizer(prompts, padding=True, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)
    return tokenizer.batch_decode(outputs, skip_special_tokens=True)

def is_answer_covered(trimmed_triples: List[List], 
                     entity_list: List[str], 
                     golden_texts: List[str]) -> bool:
    """
    检查裁剪后的子图是否包含答案
    
    Args:
        trimmed_triples: 裁剪后的三元组列表
        entity_list: 实体列表
        golden_texts: 正确答案文本列表
    
    Returns:
        如果子图包含答案返回True，否则返回False
    """
    for h, _, t in trimmed_triples:
        h_text = entity_list[h].lower()
        t_text = entity_list[t].lower()
        if any(gold in h_text or gold in t_text for gold in golden_texts):
            return True
    return False 