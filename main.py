"""
主程序入口

用于计算Triple Utilization Score (TUS) 和 FFN-Gold Alignment Score (FGAS)，基于Llama-2模型的attention patterns和hidden states

使用方法:
    1. 处理所有样本:
       python main.py
    
    2. 处理指定数量的样本(如100个):
       python main.py -n 100
    
    3. 开启调试模式处理样本:
       python main.py -n 10 -d
    
输出文件格式(inference_results_YYYYMMDD_HHMMSS.jsonl):
    - 第一行: 配置信息(时间戳、模型名称、样本数量等)
    - 中间行: 每个样本的处理结果(包含问题、答案、TUS分数、FGAS分数等)
    - 最后行: 统计信息(总样本数、有效样本数、平均TUS分数、平均FGAS分数等)
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import os
import re  # 添加re模块导入
from datetime import datetime
from tqdm import tqdm
from dataset_processor import build_external_context, load_entities_and_relations
from redeep_metrics import calculate_tus, calculate_ntus
from fgas_metrics import calculate_fgas  # 添加FGAS导入
from typing import List, Dict
import logging

# 设置日志级别
def setup_logging(debug_mode):
    if debug_mode:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)
    return logging.getLogger(__name__)

def clean_answer(answer):
    """清理单个答案"""
    # 移除多余的句子结构
    answer = re.sub(r'.*(stars in|acts in|appears in)\s*', '', answer, flags=re.IGNORECASE)
    
    # 移除年份
    answer = re.sub(r'\s*\(\d{4}\)', '', answer)
    
    # 移除标点符号
    answer = answer.strip('.,!?')
    
    # 转换为小写并去除首尾空格
    return answer.lower().strip()

def extract_answer(text):
    """修复的答案提取函数"""
    answers = []
    
    # 先按行分割
    lines = text.strip().split('\n')
    
    for line in lines:
        line = line.strip()
        
        if line.startswith('ans:'):
            content = line[4:].strip()  # 去掉 "ans:" 前缀
            
            # 如果内容包含逗号，说明是多个答案
            if ',' in content:
                # 按逗号分割并清理每个答案
                parts = [part.strip() for part in content.split(',')]
                for part in parts:
                    if part:  # 确保不是空字符串
                        cleaned = clean_answer(part)
                        if cleaned:
                            answers.append(cleaned)
            else:
                # 单个答案
                cleaned = clean_answer(content)
                if cleaned:
                    answers.append(cleaned)
    
    return answers

def is_answer_correct(predicted_answers: list, golden_answers: list) -> bool:
    """
    判断预测答案是否正确（Hit@1）
    
    参数:
        predicted_answers: 预测的答案列表
        golden_answers: 正确答案列表
    
    返回:
        bool: 是否正确（只要有一个答案正确就返回True）
    """
    # 预处理所有答案
    processed_pred = [ans.lower().strip() for ans in predicted_answers]
    processed_gold = [ans.lower().strip() for ans in golden_answers]
    
    # 规范化空格
    pred_norm = [' '.join(ans.split()) for ans in processed_pred]
    gold_norm = [' '.join(ans.split()) for ans in processed_gold]
    
    # 只要有一个预测答案在标准答案中就算对
    return any(pred in gold_norm for pred in pred_norm)

def calculate_metrics(predicted_answers: list, golden_answers: list) -> dict:
    """
    计算评估指标
    
    参数:
        predicted_answers: 预测的答案列表
        golden_answers: 正确答案列表
    
    返回:
        dict: 包含Hit@1指标的字典
    """
    if not predicted_answers or not golden_answers:
        return {'hit@1': False}
        
    # 预处理所有答案（转换为小写并标准化空格）
    # 对于预测答案，我们只取第一个答案（如果包含多个答案，取第一个）
    first_pred = predicted_answers[0].lower().strip()
    if ',' in first_pred:
        first_pred = first_pred.split(',')[0].strip()
    first_pred = ' '.join(first_pred.split())
    
    gold_norm = [' '.join(ans.lower().strip().split()) for ans in golden_answers]
    
    # 计算Hit@1：检查第一个预测是否在正确答案列表中
    hit_at_1 = first_pred in gold_norm
    
    # 将预测答案转换为列表形式
    pred_list = []
    for pred in predicted_answers:
        if ',' in pred:
            pred_list.extend([p.strip() for p in pred.split(',')])
        else:
            pred_list.append(pred.strip())
    
    print(f"\n=== DEBUG: Hit@1 Calculation ===")
    print(f"First prediction: {first_pred}")
    print(f"All predictions: {pred_list}")
    print(f"Gold answers: {gold_norm}")
    print(f"Hit@1: {hit_at_1}")
    print("=== End of Debug ===\n")
    
    return {
        'hit@1': hit_at_1
    }

def find_answer_position(output_text: str, answer_text: str, tokenizer) -> dict:
    """
    在输出文本中找到答案的位置
    
    参数:
        output_text: 模型生成的完整输出文本
        answer_text: 提取出的答案文本
        tokenizer: 分词器
    
    返回:
        包含答案起始和结束位置的字典
    """
    # 如果答案为空，返回默认位置
    if not answer_text:
        return {'start_idx': 0, 'end_idx': 0}
    
    # 编码完整输出文本
    output_tokens = tokenizer.encode(output_text, add_special_tokens=False)
    
    # 编码答案文本
    answer_tokens = tokenizer.encode(answer_text, add_special_tokens=False)
    
    # 在输出tokens中查找答案tokens
    for i in range(len(output_tokens) - len(answer_tokens) + 1):
        if output_tokens[i:i+len(answer_tokens)] == answer_tokens:
            return {
                'start_idx': i,
                'end_idx': i + len(answer_tokens) - 1
            }
    
    # 如果找不到完全匹配，返回默认位置
    return {'start_idx': 0, 'end_idx': len(output_tokens) - 1}

def process_sample(sample, model, tokenizer, entity_list, relation_list, debug=False):
    """处理单个样本"""
    logger = logging.getLogger(__name__)
    try:
        # 构建系统提示
        system_prompt = """Based on the triples retrieved from a knowledge graph, please answer the question. Please return ONLY the answer entities as a list, each prefixed with "ans:". Do not include explanations or reasoning."""

        # 构建三元组文本
        if not sample.get('trimmed_triples'):
            return {
                'error': 'No trimmed triples found',
                'tus_score': 0.0,
                'fgas_score': 0.0,
                'metrics': {'hit@1': False}
            }
            
        triples_text = "Knowledge:\n"
        for h, r, t in sample['trimmed_triples']:
            triples_text += f"{h} {r} {t}\n"
        
        # 构建用户输入（包含三元组和问题）
        user_content = f"{triples_text}\nQuestion: {sample['question']}\nAnswer:"
        
        # 构建完整的提示
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ]
        
        # 准备输入
        inputs = tokenizer.apply_chat_template(messages, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        # 生成答案
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                num_return_sequences=1,
                output_attentions=True,
                temperature=0.7,
                top_p=0.9
            )
        
        # 找到生成文本的开始位置（在[/INST]之后）
        input_length = len(inputs['input_ids'][0])
        generated_ids = outputs[0][input_length:]
        
        # 解码生成的文本
        print("\n=== DEBUG: Tokenizer Decoding ===")
        output_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        print(f"Raw decoded text: {repr(output_text)}")
        print("=== End of Decoding Debug ===\n")
        
        # 提取答案
        predicted_answers = extract_answer(output_text)
        
        # 计算评估指标
        metrics = calculate_metrics(predicted_answers, sample['golden_texts'])
        
        # 获取注意力权重
        attention_weights = torch.stack([output.attentions[-1] for output in outputs])
        
        # 找到答案的起始和结束位置
        answer_start_idx = len(inputs['input_ids'][0]) - 1  # 最后一个输入token的位置
        answer_end_idx = len(outputs[0]) - 1  # 最后一个生成token的位置
        
        # 计算TUS分数
        tus_score = calculate_tus(
            attention_weights=attention_weights,
            external_context=build_external_context(sample, tokenizer, entity_list, relation_list),
            gold_triples=sample['gold_triples'],
            answer_start_idx=answer_start_idx,
            answer_end_idx=answer_end_idx,
            input_ids=outputs[0],
            tokenizer=tokenizer,
            debug=debug
        )
        
        # 计算nTUS分数（归一化的TUS）
        ntus_score = calculate_ntus(
            attention_weights=attention_weights,
            external_context=build_external_context(sample, tokenizer, entity_list, relation_list),
            gold_triples=sample['gold_triples'],
            trimmed_triples=sample['trimmed_triples'],
            answer_start_idx=answer_start_idx,
            answer_end_idx=answer_end_idx,
            input_ids=outputs[0],
            tokenizer=tokenizer,
            debug=debug
        )
        
        # 计算FGAS分数 - 使用正确的答案位置和Golden Expansion Set
        fgas_answer_start = len(inputs['input_ids'][0])  # 生成文本的真正开始位置
        fgas_answer_end = len(outputs[0]) - 1  # 生成文本的结束位置
        
        # 🔥 FGAS使用扩展集合（语义丰富度）
        fgas_golden_triples = sample.get('golden_expansion_set', sample['gold_triples'])
        
        fgas_score = calculate_fgas(
            model=model,
            tokenizer=tokenizer,
            input_ids=outputs[0],
            gold_triples=fgas_golden_triples,
            answer_start_idx=fgas_answer_start,
            answer_end_idx=fgas_answer_end,
            debug=debug
        )
        
        return {
            'question': sample['question'],
            'model_input': user_content,
            'model_output': output_text,
            'answer': predicted_answers,
            'golden_answers': sample['golden_texts'],
            'metrics': metrics,
            'tus_score': tus_score,
            'ntus_score': ntus_score,
            'fgas_score': fgas_score,
            'extracted_answers': predicted_answers,
            'raw_model_output': output_text
        }
            
    except Exception as e:
        logger.error(f"处理样本时出错: {str(e)}")
        return {
            'error': str(e),
            'tus_score': 0.0,
            'ntus_score': 0.0,
            'fgas_score': 0.0,
            'metrics': {'hit@1': False}
        }

def main(num_samples=None, debug=False):
    """
    主函数
    
    参数:
        num_samples: 要处理的样本数量，None表示处理所有样本
        debug: 是否打印调试信息
    """
    # 1. 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 设置批处理大小
    batch_size = 2  # 减小批处理大小以适应显存
    
    # 2. 加载模型和tokenizer
    model_name = "meta-llama/Llama-2-7b-chat-hf"
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,  # 在GPU上使用半精度
        device_map="auto",  # 自动处理设备映射
        use_cache=True  # 启用KV缓存
    )  # 移除.to(device)，因为device_map="auto"会自动处理设备分配
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # 设置tokenizer配置
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'  # 对于decoder-only模型，使用左侧padding
    
    # 3. 加载实体和关系列表
    data_dir = "/mnt/d/datasets/GraphTruth/metaqa-1hop/metaqa-1hop"
    try:
        entity_list, relation_list = load_entities_and_relations(data_dir)
        print(f"加载了 {len(entity_list)} 个实体和 {len(relation_list)} 个关系")
    except Exception as e:
        print(f"加载实体和关系时出错: {str(e)}")
        return
    
    # 4. 加载数据
    samples = []
    try:
        # 🔥 使用最新的TUS一致性策略数据文件（修复了FGAS扩展问题）
        data_file = "experiment_records/trimming_results_tus_consistent_20250623_203829.jsonl"
        with open(data_file, 'r') as f:
            # 跳过配置行
            next(f)
            # 读取指定数量的样本
            for line in f:
                data = json.loads(line)
                if 'sample_id' in data:  # 确保是样本数据而不是统计信息
                    samples.append(data)
                    if num_samples and len(samples) >= num_samples:
                        break
    except Exception as e:
        print(f"加载数据集时出错: {str(e)}")
        return
    
    print(f"\n处理 {len(samples)} 个样本...")
    print(f"批处理大小: {batch_size}")
    if device.type == "cuda":
        print(f"当前CUDA设备: {torch.cuda.current_device()}")
        print(f"可用显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    # 5. 创建结果目录
    os.makedirs('experiment_records', exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = f'experiment_records/inference_results_{timestamp}.jsonl'
    
    # 6. 处理样本并保存结果
    results = []
    total_tus = 0
    total_fgas = 0
    valid_samples = 0
    failed_samples = []
    
    with open(result_file, 'w', encoding='utf-8') as f:
        # 写入配置信息
        config = {
        'timestamp': timestamp,
        'model': model_name,
            'num_samples': num_samples,
            'data_source': data_file,
        'device': str(device),
            'batch_size': batch_size
        }
        f.write(json.dumps({'config': config}, ensure_ascii=False) + '\n')
        
        # 按批次处理样本
        for i in tqdm(range(0, len(samples), batch_size), desc="处理批次"):
            batch_samples = samples[i:i+batch_size]
            batch_results = []
            
            # 批量处理输入
            batch_inputs = []
            for sample in batch_samples:
                try:
                    # 构建系统提示
                    system_prompt = """Based on the triples retrieved from a knowledge graph, please answer the question. Please return ONLY the answer entities as a list, each prefixed with "ans:". Do not include explanations or reasoning."""
                    
                    # 构建三元组文本
                    if not sample.get('trimmed_triples'):
                        failed_samples.append({
                            'sample_id': sample.get('sample_id', -1),
                            'question': sample.get('question', ''),
                            'answer': sample.get('golden_texts', [''])[0],
                            'reason': 'No trimmed triples found'
                        })
                        continue
                        
                    triples_text = "Knowledge:\n"
                    for h, r, t in sample['trimmed_triples']:
                        triples_text += f"{h} {r} {t}\n"
                    
                    # 构建用户输入（包含三元组和问题）
                    user_content = f"{triples_text}\nQuestion: {sample['question']}\nAnswer:"
                    
                    # 构建完整输入
                    messages = [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_content}
                    ]
                    
                    # 使用chat template
                    full_input = tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True
                    )
                    
                    batch_inputs.append((sample, full_input))
                except Exception as e:
                    failed_samples.append({
                        'sample_id': sample.get('sample_id', -1),
                        'question': sample.get('question', ''),
                        'answer': sample.get('golden_texts', [''])[0],
                        'reason': str(e)
                    })
            
            if not batch_inputs:
                continue
                
            # 批量编码
            input_texts = [x[1] for x in batch_inputs]
            inputs = tokenizer(
                input_texts,
                padding=True,
                truncation=True,
                max_length=512,  # 设置最大长度
                return_tensors="pt"
            )
            # 将所有tensor移动到正确的设备
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            
            # 批量生成
            with torch.no_grad():
                # 首先用forward获取attention weights
                forward_outputs = model(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    output_attentions=True
                )
                
                # 获取最后4层的attention weights并正确处理
                all_attentions = forward_outputs.attentions  # 这是一个元组，每个元素是一层的attention
                last_4_attentions = all_attentions[-4:]  # 取最后4层
                attention_weights = torch.stack(last_4_attentions, dim=0)  # [4, batch, heads, seq_len, seq_len]
                
                # 然后用generate生成答案
                outputs = model.generate(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    max_new_tokens=50,  # 减小生成长度，因为我们只需要实体名
                    do_sample=False,  # 使用贪婪解码
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    use_cache=True  # 启用KV缓存
                )
            
            # 处理每个样本的输出
            for idx, (sample, full_input) in enumerate(batch_inputs):
                try:
                    # 获取当前样本的attention weights
                    sample_attention = attention_weights[:, idx]  # [4, heads, seq_len, seq_len]
                    
                    # 解码输出
                    if isinstance(outputs, tuple):
                        output_ids = outputs[0][idx]
                    else:
                        output_ids = outputs[idx]
                    
                    # 获取答案的起始和结束位置
                    answer_start_idx = len(inputs['input_ids'][idx])
                    answer_end_idx = len(output_ids)
                    
                    # 计算TUS分数
                    tus_score = calculate_tus(
                        attention_weights=sample_attention,
                        external_context=build_external_context(sample, tokenizer, entity_list, relation_list),
                        gold_triples=sample['gold_triples'],
                        answer_start_idx=answer_start_idx,
                        answer_end_idx=answer_end_idx,
                        input_ids=inputs['input_ids'][idx],
                        tokenizer=tokenizer,
                        debug=debug
                    )
                    
                    # 计算nTUS分数（归一化的TUS）
                    ntus_score = calculate_ntus(
                        attention_weights=sample_attention,
                        external_context=build_external_context(sample, tokenizer, entity_list, relation_list),
                        gold_triples=sample['gold_triples'],
                        trimmed_triples=sample['trimmed_triples'],
                        answer_start_idx=answer_start_idx,
                        answer_end_idx=answer_end_idx,
                        input_ids=inputs['input_ids'][idx],
                        tokenizer=tokenizer,
                        debug=debug
                    )
                    
                    # 计算FGAS分数 - 使用正确的答案位置和Golden Expansion Set
                    fgas_answer_start = len(inputs['input_ids'][idx])  # 生成文本的真正开始位置
                    fgas_answer_end = len(output_ids) - 1  # 生成文本的结束位置
                    
                    # 🔥 FGAS使用扩展集合（语义丰富度）
                    fgas_golden_triples = sample.get('golden_expansion_set', sample['gold_triples'])
                    
                    fgas_score = calculate_fgas(
                        model=model,
                        tokenizer=tokenizer,
                        input_ids=output_ids,
                        gold_triples=fgas_golden_triples,
                        answer_start_idx=fgas_answer_start,
                        answer_end_idx=fgas_answer_end,
                        debug=debug
                    )
                    
                    # 解码生成的文本
                    generated_text = tokenizer.decode(output_ids[answer_start_idx:answer_end_idx], skip_special_tokens=True)
                    predicted_answers = [ans.strip() for ans in generated_text.lower().split('ans:') if ans.strip()]
                    if not predicted_answers:
                        predicted_answers = [generated_text.strip()]
                    
                    # 计算指标
                    answer_texts = sample['golden_texts']
                    metrics = calculate_metrics(predicted_answers, answer_texts)
                    
                    result = {
                        'question': sample['question'],
                        'answer': predicted_answers[0],
                        'golden_answers': answer_texts,
                        'metrics': metrics,
                        'tus_score': tus_score,
                        'ntus_score': ntus_score,
                        'fgas_score': fgas_score,
                        'model_input': full_input,
                        'model_output': generated_text,
                        'extracted_answers': predicted_answers,
                        'raw_model_output': generated_text
                    }
                    
                    valid_samples += 1
                    batch_results.append(result)
                    
                    # 保存结果
                    f.write(json.dumps(result, ensure_ascii=False) + '\n')
                except Exception as e:
                    failed_samples.append({
                        'sample_id': sample.get('sample_id', -1),
                        'question': sample['question'],
                        'answer': sample['golden_texts'][0],
                        'reason': str(e)
                    })
            
            results.extend(batch_results)
            
            # 清理显存
            if device.type == "cuda":
                torch.cuda.empty_cache()
    
    print("\n====================================================================================================")
    print("评估结果:")
    print("----------------------------------------------------------------------------------------------------\n")
    
    # 详细信息
    print("详细信息:")
    print("----------------------------------------------------------------------------------------------------")
    for i, result in enumerate(results, 1):
        print(f"\n样本 {i}:")
        print(f"问题: {result['question']}")
        print(f"模型输入:\n{result.get('model_input', '(无模型输入)')}")
        print(f"模型输出:\n{result.get('model_output', '(无原始输出)')}")
        print(f"提取答案: {result.get('answer', [])}")
        print(f"标准答案: {', '.join(result.get('golden_answers', []))}")
        print(f"评估结果: {'✓ 正确' if result['metrics']['hit@1'] else '✗ 错误'} (TUS={result.get('tus_score', 0):.3f}, nTUS={result.get('ntus_score', 0):.3f}, FGAS={result.get('fgas_score', 0):.3f})")
        print("--------------------------------------------------")
    
    # 汇总统计
    total_samples = len(results)
    valid_samples = len([r for r in results if 'error' not in r])
    failed_samples = len(failed_samples)
    hit_at_1 = sum(1 for r in results if r['metrics']['hit@1']) / total_samples * 100 if total_samples > 0 else 0
    avg_tus = sum(r.get('tus_score', 0) for r in results) / total_samples if total_samples > 0 else 0
    avg_ntus = sum(r.get('ntus_score', 0) for r in results) / total_samples if total_samples > 0 else 0
    avg_fgas = sum(r.get('fgas_score', 0) for r in results) / total_samples if total_samples > 0 else 0
    
    print("\n==================================================")
    print("汇总统计:")
    print("--------------------------------------------------")
    print("| 指标              | 值              |")
    print("|------------------|----------------|")
    print(f"| 总样本数          |{total_samples:>15} |")
    print(f"| 有效样本数        |{valid_samples:>15} |")
    print(f"| 失败样本数        |{failed_samples:>15} |")
    print(f"| Hit@1           |{hit_at_1:>14.2f}% |")
    print(f"| 平均TUS分数      |{avg_tus:>14.2f}  |")
    print(f"| 平均nTUS分数     |{avg_ntus:>14.2f}  |")
    print(f"| 平均FGAS分数    |{avg_fgas:>14.2f}  |")
    print("--------------------------------------------------\n")
    
    # 评估结果表格
    print("\n" + "="*140)
    print("评估结果表格:")
    print("-"*140)
    print("| {:^4} | {:^30} | {:^40} | {:^8} | {:^9} | {:^9} | {:^9} |".format(
        "序号", "问题", "提取答案", "正确性", "TUS分数", "nTUS分数", "FGAS分数"
    ))
    print("|" + "-"*6 + "|" + "-"*32 + "|" + "-"*42 + "|" + "-"*10 + "|" + "-"*11 + "|" + "-"*11 + "|" + "-"*11 + "|")
    
    # 遍历每个样本的结果并打印表格行
    for i, result in enumerate(results, 1):
        question = result['question']
        if len(question) > 28:
            question = question[:25] + "..."
            
        # 修改这里：直接显示答案列表
        extracted_answers = str(result.get('answer', []))
        if len(extracted_answers) > 38:
            extracted_answers = extracted_answers[:35] + "..."
            
        is_correct = "✓" if result['metrics']['hit@1'] else "✗"
        tus_score = result.get('tus_score', 0)
        ntus_score = result.get('ntus_score', 0)
        fgas_score = result.get('fgas_score', 0)
        
        print("| {:4d} | {:<30} | {:<40} | {:^8} | {:9.3f} | {:9.3f} | {:9.3f} |".format(
            i, question, extracted_answers, is_correct, tus_score, ntus_score, fgas_score
        ))
    
    print("|" + "-"*6 + "|" + "-"*32 + "|" + "-"*42 + "|" + "-"*10 + "|" + "-"*11 + "|" + "-"*11 + "|" + "-"*11 + "|")
    print("\n")
    
    # 保存结果
    print(f"结果已保存至: {result_file}")
    
    return {
        'total_samples': total_samples,
        'valid_samples': valid_samples,
        'failed_samples': failed_samples,
        'hit_at_1': hit_at_1,
        'avg_tus': avg_tus,
        'avg_ntus': avg_ntus,
        'avg_fgas': avg_fgas
    }

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='计算Triple Utilization Score (TUS)')
    parser.add_argument('-n', '--num_samples', type=int, default=None, help='要处理的样本数量')
    parser.add_argument('-d', '--debug', action='store_true', help='是否打印调试信息')
    args = parser.parse_args()
    
    # 在参数解析后设置日志级别
    logger = setup_logging(args.debug)
    
    main(args.num_samples, args.debug)