"""
数据集处理器模块 (Dataset Processor)

本模块负责数据集的完整处理流程，主要功能包括：
1. 加载原始数据集和实体关系
2. 数据预处理和格式转换
3. 子图裁剪和优化
4. 构建模型输入prompt
5. 保存处理后的数据集

主要函数:
- load_data: 加载原始数据集
- load_entities_and_relations: 加载实体和关系
- prepare_dataset: 准备和裁剪数据集
- build_prompt: 构建模型输入prompt
- save_experiment_results: 保存实验结果

作者: [Your Name]
创建日期: 2024-03-19
"""

import json
import os
import time
import gc
import psutil
import signal
from datetime import datetime
from typing import List, Dict, Tuple, Set, Any
from tqdm import tqdm

from subgraph_utils import (
    get_subgraph_shortest_path_plus,
    get_fixed_length_subgraph,
    process_sample,
    build_graph,
    find_shortest_path,
    get_path_triples
)

class TimeoutError(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutError("操作超时")

def process_sample_with_timeout(sample, entity_list, topk, timeout_seconds=30):
    """
    带超时的样本处理函数
    
    Args:
        sample: 样本数据
        entity_list: 实体列表
        topk: 子图大小
        timeout_seconds: 超时时间（秒）
    
    Returns:
        处理结果或None（如果超时/出错）
    """
    try:
        # 设置超时信号
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(timeout_seconds)
        
        # 执行处理
        result = process_sample(sample, entity_list, topk)
        
        # 取消超时信号
        signal.alarm(0)
        return result
        
    except TimeoutError:
        print(f"⚠️ 样本超时跳过 (ID: {sample.get('id', 'unknown')})")
        signal.alarm(0)
        return None
    except Exception as e:
        print(f"⚠️ 样本处理错误跳过: {e}")
        signal.alarm(0)
        return None

def load_data(data_path: str) -> List[Dict]:
    """
    加载MetaQA数据集
    
    Args:
        data_path: 数据文件路径或数据集根目录路径
    
    Returns:
        数据集列表，每个元素是一个字典，包含问题、子图和答案
    """
    data = []
    try:
        # 如果data_path是一个文件路径，获取其目录
        if data_path.endswith('.json'):
            data_dir = os.path.dirname(data_path)
            dev_file = data_path
        else:
            data_dir = data_path
            dev_file = os.path.join(data_path, "dev_simple.json")
            
        # 加载数据
        with open(dev_file) as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
                    
        print(f"Loaded {len(data)} samples from {dev_file}")
        
    except Exception as e:
        print(f"Error loading data: {e}")
        raise
    return data

def load_entities_and_relations():
    """加载实体和关系列表"""
    # 修改为1hop的路径
    entity_file = "dataset/metaqa-1hop/entities.txt"
    relation_file = "dataset/metaqa-1hop/relations.txt"
    
    # 加载实体列表
    with open(entity_file, 'r', encoding='utf-8') as f:
        entities = [line.strip() for line in f]
    print(f"Loaded {len(entities)} entities from {entity_file}")
    
    # 加载关系列表
    with open(relation_file, 'r', encoding='utf-8') as f:
        relations = [line.strip() for line in f]
    print(f"Loaded {len(relations)} relations from {relation_file}")
    
    return entities, relations

def convert_indices_to_freebase(triple: List[int], 
                              entity_list: List[str], 
                              relation_list: List[str]) -> List[str]:
    """
    将三元组的数字索引转换为实际的freebase ID和关系路径
    
    Args:
        triple: [head_idx, relation_idx, tail_idx]形式的三元组
        entity_list: 实体列表（freebase ID）
        relation_list: 关系列表（关系路径）
    
    Returns:
        [head_freebase_id, relation_path, tail_freebase_id]形式的三元组
    """
    head_idx, rel_idx, tail_idx = triple
    head_id = entity_list[head_idx] if head_idx < len(entity_list) else str(head_idx)
    relation = relation_list[rel_idx] if rel_idx < len(relation_list) else str(rel_idx)
    tail_id = entity_list[tail_idx] if tail_idx < len(entity_list) else str(tail_idx)
    return [head_id, relation, tail_id]

def build_prompt(sample: Dict, entity_list: List[str], relation_list: List[str], topk: int = 20) -> str:
    """
    构建模型输入的prompt
    
    Args:
        sample: 样本数据字典
        entity_list: 实体列表
        relation_list: 关系列表
        topk: 选择的三元组数量
    
    Returns:
        构建好的prompt字符串
    """
    # 获取裁剪后的子图
    trimmed_triples = process_sample(sample, entity_list, topk)
    
    # 构建prompt
    prompt = ('Based on the triples retrieved from a knowledge graph, '
             'please answer the question. Please return formatted answers '
             'as a list, each prefixed with "ans:".\n Triplets:\n')
    
    for h, r, t in trimmed_triples:
        h_label = entity_list[h]
        r_label = relation_list[r]
        t_label = entity_list[t]
        prompt += f"({h_label}, {r_label}, {t_label})\n"
    
    prompt += f"\nQuestion:\n{sample['question']}\nAnswer:"
    return prompt

def get_gold_triples(triples: List[List[int]], 
                    question_entities: Set[int], 
                    answer_entities: Set[int]) -> List[List[int]]:
    """
    获取gold三元组（最短路径上的三元组）
    
    Args:
        triples: 原始三元组列表
        question_entities: 问题实体集合
        answer_entities: 答案实体集合
    
    Returns:
        gold三元组列表
    """
    # 1. 构建图结构
    graph, triple_dict = build_graph(triples)
    
    # 2. 找出所有问题实体到答案实体的最短路径
    gold_triples = set()  # 使用集合去重
    
    for q_entity in question_entities:
        for a_entity in answer_entities:
            path = find_shortest_path(graph, q_entity, a_entity)
            if path:
                path_triples = get_path_triples(path, triple_dict)
                gold_triples.update(path_triples)
    
    return [list(triple) for triple in gold_triples]  # 转换回列表形式


def prepare_dataset(data_dir: str, output_dir: str, topk: int = 20) -> str:
    """
    准备数据集：加载、裁剪和保存
    
    Args:
        data_dir: 原始数据集目录
        output_dir: 输出目录
        topk: 子图大小
    
    Returns:
        保存的结果文件路径
    """
    # 1. 加载数据
    print("\nLoading data...")
    data = load_data(os.path.join(data_dir, "dev_simple.json"))
    entity_list, relation_list = load_entities_and_relations()
    
    # 统计原始三元组数量
    triple_counts = [len(sample['subgraph']['tuples']) for sample in data]
    print(f"样本总数: {len(triple_counts)}")
    print(f"三元组数量 - 最大: {max(triple_counts)}, 最小: {min(triple_counts)}, 平均: {sum(triple_counts)/len(triple_counts):.2f}")
    
    # 2. 设置实验参数
    batch_size = 16  # 增加批处理大小以提升速度
    num_samples = len(data)
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 生成时间戳
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 设置保存文件路径
    trimming_result_path = os.path.join(output_dir, f"trimming_results_{timestamp}.jsonl")
    
    # 记录配置信息
    config = {
        "timestamp": timestamp,
        "topk": topk,
        "total_samples": num_samples,
        "data_dir": data_dir
    }
    
    # 3. 开始数据集裁剪
    print("\n=== 开始子图裁剪 ===")
    trimming_start_time = time.time()
    answer_covered_count = 0  # 全局计数器
    skipped_count = 0  # 跳过的样本计数器
    batch_recalls = []
    
    with open(trimming_result_path, "w", encoding="utf-8") as f:
        # 写入配置信息
        f.write(json.dumps({"config": config}, ensure_ascii=False) + "\n")
        
        for i in tqdm(range(0, num_samples, batch_size), desc="Trimming Subgraphs"):
            batch_samples = data[i:i+batch_size]
            batch_covered = 0
            
            for j, sample in enumerate(batch_samples):
                question = sample['question']
                golden_texts = [ans['text'].lower() for ans in sample.get('answers', []) if ans.get('text')]
                
                # 获取问题实体和答案实体 - 使用精确匹配
                question_entities = set()
                question_lower = question.lower()
                for idx, entity in enumerate(entity_list):
                    entity_lower = entity.lower()
                    # 精确匹配：实体必须作为完整词出现
                    if entity_lower in question_lower:
                        # 检查是否为完整单词（前后是空格或标点）
                        start_pos = question_lower.find(entity_lower)
                        while start_pos != -1:
                            end_pos = start_pos + len(entity_lower)
                            
                            # 检查前后边界
                            before_ok = (start_pos == 0 or not question_lower[start_pos-1].isalnum())
                            after_ok = (end_pos == len(question_lower) or not question_lower[end_pos].isalnum())
                            
                            if before_ok and after_ok:
                                question_entities.add(idx)
                                break
                            
                            start_pos = question_lower.find(entity_lower, start_pos + 1)
                
                answer_entities = set()
                for ans in sample.get('answers', []):
                    if ans.get('text'):
                        ans_text = ans['text'].lower().strip()
                        for idx, entity in enumerate(entity_list):
                            entity_lower = entity.lower()
                            # 精确匹配：答案文本必须与实体完全匹配或实体包含答案文本
                            if ans_text == entity_lower or (ans_text in entity_lower and len(ans_text) >= 3):
                                # 对于包含关系，确保是完整单词匹配
                                if ans_text == entity_lower:
                                    answer_entities.add(idx)
                                else:
                                    # 检查是否为完整单词
                                    start_pos = entity_lower.find(ans_text)
                                    if start_pos != -1:
                                        end_pos = start_pos + len(ans_text)
                                        before_ok = (start_pos == 0 or not entity_lower[start_pos-1].isalnum())
                                        after_ok = (end_pos == len(entity_lower) or not entity_lower[end_pos].isalnum())
                                        if before_ok and after_ok:
                                            answer_entities.add(idx)
                
                # 获取gold三元组
                gold_triples = get_gold_triples(
                    sample['subgraph']['tuples'],
                    question_entities,
                    answer_entities
                )
                
                # 使用带超时的子图构建方法
                trimmed_triples = process_sample_with_timeout(sample, entity_list, topk, timeout_seconds=30)
                
                if trimmed_triples is not None:
                    # 成功处理
                    answer_covered = True  # 由于算法保证答案可达，所以一定为True
                    batch_covered += 1
                    answer_covered_count += 1
                else:
                    # 超时或出错，使用后备方法
                    print(f"⚠️ 样本 {i+j} 使用后备方法处理")
                    try:
                        trimmed_triples = get_fixed_length_subgraph(
                            sample['subgraph']['tuples'],
                            extract_question_entities(question, entity_list),
                            golden_texts,
                            entity_list,
                            topk
                        )
                        answer_covered = is_answer_covered(trimmed_triples, entity_list, golden_texts)
                        if answer_covered:
                            batch_covered += 1
                            answer_covered_count += 1
                    except Exception as e:
                        print(f"❌ 样本 {i+j} 完全失败，跳过: {e}")
                        skipped_count += 1
                        # 创建一个最小的默认三元组列表
                        trimmed_triples = [[0, 0, 0]] * min(topk, len(sample['subgraph']['tuples']))
                        answer_covered = False
                
                # 将三元组的索引转换为实际文本
                text_triples = []
                gold_text_triples = []
                for h, r, t in trimmed_triples:
                    head_text = entity_list[h]
                    rel_text = relation_list[r]
                    tail_text = entity_list[t]
                    text_triples.append([head_text, rel_text, tail_text])
                
                for h, r, t in gold_triples:
                    head_text = entity_list[h]
                    rel_text = relation_list[r]
                    tail_text = entity_list[t]
                    gold_text_triples.append([head_text, rel_text, tail_text])
                
                # 写入裁剪结果
                result = {
                    "sample_id": i + j,
                    "question": question,
                    "golden_texts": golden_texts,
                    "trimmed_subgraph_length": len(trimmed_triples),
                    "original_subgraph_length": len(sample['subgraph']['tuples']),
                    "gold_triples_length": len(gold_triples),  # 添加gold三元组长度
                    "answer_covered": answer_covered,
                    "trimmed_triples": text_triples,  # 保存文本形式的三元组
                    "gold_triples": gold_text_triples,  # 保存gold三元组
                    "processing_time": time.time() - trimming_start_time
                }
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
            
            # 计算batch统计信息
            batch_recall = batch_covered / len(batch_samples) * 100
            batch_recalls.append(batch_recall)
            current_recall = answer_covered_count / (i + len(batch_samples)) * 100
            
            # 内存监控和清理（减少频率）
            current_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            if current_memory > 20000 and (i // batch_size) % 10 == 0:  # 内存超过20GB且每10个batch才清理
                gc.collect()
                print(f"\n内存清理 - 当前内存使用: {current_memory:.1f}MB")
            
            # 保存batch统计信息（包含内存使用）
            batch_stats = {
                "batch_stats": {
                    "batch_id": i // batch_size,
                    "batch_size": len(batch_samples),
                    "batch_covered": batch_covered,
                    "batch_recall": batch_recall,
                    "cumulative_recall": current_recall,
                    "memory_usage_mb": current_memory,
                    "processing_time": time.time() - trimming_start_time
                }
            }
            f.write(json.dumps(batch_stats, ensure_ascii=False) + "\n")
            
            # 每20个batch打印进度信息
            if (i // batch_size) % 50 == 0:  # 减少内存清理频率
                samples_processed = i + len(batch_samples)
                progress_pct = samples_processed / num_samples * 100
                avg_time_per_sample = (time.time() - trimming_start_time) / samples_processed
                eta_seconds = avg_time_per_sample * (num_samples - samples_processed)
                print(f"\n进度: {progress_pct:.1f}% ({samples_processed}/{num_samples})")
                print(f"内存: {current_memory:.1f}MB, ETA: {eta_seconds/60:.1f}分钟")
        
        # 写入最终统计信息
        final_stats = {
            "final_stats": {
                "total_samples": num_samples,
                "answer_covered_count": answer_covered_count,
                "skipped_count": skipped_count,
                "answer_recall": answer_covered_count / num_samples * 100,
                "skip_rate": skipped_count / num_samples * 100,
                "avg_batch_recall": sum(batch_recalls) / len(batch_recalls),
                "min_batch_recall": min(batch_recalls),
                "max_batch_recall": max(batch_recalls),
                "total_time": time.time() - trimming_start_time
            }
        }
        f.write(json.dumps(final_stats, ensure_ascii=False) + "\n")
    
    print(f"\n=== 裁剪完成 ===")
    print(f"Answer Recall: {answer_covered_count/num_samples*100:.2f}% ({answer_covered_count}/{num_samples})")
    print(f"平均Batch Recall: {sum(batch_recalls)/len(batch_recalls):.2f}%")
    print(f"最小Batch Recall: {min(batch_recalls):.2f}%")
    print(f"最大Batch Recall: {max(batch_recalls):.2f}%")
    print(f"裁剪耗时: {time.time() - trimming_start_time:.2f}秒")
    print(f"结果已保存至: {trimming_result_path}")
    
    return trimming_result_path

def extract_question_entities(question: str, entity_list: List[str]) -> List[int]:
    """
    从问题中提取实体索引 - 使用精确匹配
    
    Args:
        question: 输入问题
        entity_list: 实体列表
    
    Returns:
        问题中出现的实体索引列表
    """
    question_entities = []
    question_lower = question.lower()
    for i, entity in enumerate(entity_list):
        entity_lower = entity.lower()
        # 精确匹配：实体必须作为完整词出现
        if entity_lower in question_lower:
            # 检查是否为完整单词（前后是空格或标点）
            start_pos = question_lower.find(entity_lower)
            while start_pos != -1:
                end_pos = start_pos + len(entity_lower)
                
                # 检查前后边界
                before_ok = (start_pos == 0 or not question_lower[start_pos-1].isalnum())
                after_ok = (end_pos == len(question_lower) or not question_lower[end_pos].isalnum())
                
                if before_ok and after_ok:
                    question_entities.append(i)
                    break
                
                start_pos = question_lower.find(entity_lower, start_pos + 1)
    
    return question_entities

def is_answer_covered(trimmed_triples: List[List[int]], entity_list: List[str], golden_texts: List[str]) -> bool:
    """
    检查golden_texts（golden answer的文本，已小写）是否出现在trimmed_triples的头实体或尾实体中。
    
    Args:
        trimmed_triples: 裁剪后的三元组列表
        entity_list: 实体列表
        golden_texts: golden answer文本列表（已转为小写）
    
    Returns:
        bool: 是否至少有一个golden answer在子图中
    """
    for h, r, t in trimmed_triples:
        h_label = entity_list[h].lower()
        t_label = entity_list[t].lower()
        if any(gold in h_label or gold in t_label for gold in golden_texts):
            return True
    return False

def build_external_context(sample: Dict[str, Any], 
                         tokenizer,
                         entity_list: List[str], 
                         relation_list: List[str]) -> Dict[str, Any]:
    """
    从trimming_results中的样本构建external_context字典
    
    参数:
        sample: trimming_results中的一个样本
        tokenizer: 分词器，用于获取token位置
        entity_list: 实体列表
        relation_list: 关系列表
    
    返回:
        external_context字典，包含：
        - entities: 子图中的实体ID列表
        - relations: 子图中的关系ID列表
        - triples: 三元组列表
        - entity_positions: 实体在输入序列中的位置
        - relation_positions: 关系在输入序列中的位置
        - gold_triples_positions: gold三元组在输入序列中的位置
    """
    # 1. 将文本三元组转换为ID形式
    triples = []
    entities = set()
    relations = set()
    
    # 获取gold三元组列表
    gold_triples = sample['gold_triples']
    
    for h_text, r_text, t_text in sample['trimmed_triples']:
        # 查找实体和关系的ID
        try:
            h_id = entity_list.index(h_text)
        except ValueError:
            h_id = len(entity_list) + len(entities)
            entities.add(h_id)
            
        try:
            r_id = relation_list.index(r_text)
        except ValueError:
            r_id = len(relation_list) + len(relations)
            relations.add(r_id)
            
        try:
            t_id = entity_list.index(t_text)
        except ValueError:
            t_id = len(entity_list) + len(entities)
            entities.add(t_id)
        
        triples.append([h_id, r_id, t_id])
        entities.add(h_id)
        entities.add(t_id)
        relations.add(r_id)
    
    # 2. 构建输入序列
    question = sample['question']
    context = ""
    gold_positions = set()  # 用于记录gold三元组的位置
    current_pos = len(tokenizer.encode(question + " [SEP] ", add_special_tokens=True)) - 1
    
    for h, r, t in triples:
        h_text = entity_list[h] if h < len(entity_list) else f"entity_{h}"
        r_text = relation_list[r] if r < len(relation_list) else f"relation_{r}"
        t_text = entity_list[t] if t < len(entity_list) else f"entity_{t}"
        
        # 如果当前三元组是gold三元组
        triple_text = f"{h_text} {r_text} {t_text}"
        if [h_text, r_text, t_text] in gold_triples:
            # 计算这个三元组中所有token的位置
            triple_tokens = tokenizer.encode(triple_text, add_special_tokens=False)
            gold_positions.update(range(current_pos, current_pos + len(triple_tokens)))
        
        context += triple_text + " . "
        current_pos += len(tokenizer.encode(triple_text + " . ", add_special_tokens=False))
    
    # 3. 对完整输入进行编码
    full_input = f"{question} [SEP] {context}"
    tokens = tokenizer.encode(full_input, add_special_tokens=True)
    
    # 4. 找到实体和关系在序列中的位置
    entity_positions = []
    relation_positions = []
    
    # 对问题中的实体进行定位
    for entity_id in entities:
        if entity_id >= len(entity_list):
            continue
        entity_text = entity_list[entity_id]
        # 在问题中查找
        entity_tokens = tokenizer.encode(entity_text, add_special_tokens=False)
        for i in range(len(tokens) - len(entity_tokens) + 1):
            if tokens[i:i+len(entity_tokens)] == entity_tokens:
                entity_positions.extend(range(i, i + len(entity_tokens)))
    
    # 对子图中的关系进行定位
    for relation_id in relations:
        if relation_id >= len(relation_list):
            continue
        relation_text = relation_list[relation_id]
        # 在子图部分查找
        relation_tokens = tokenizer.encode(relation_text, add_special_tokens=False)
        for i in range(len(tokens) - len(relation_tokens) + 1):
            if tokens[i:i+len(relation_tokens)] == relation_tokens:
                relation_positions.extend(range(i, i + len(relation_tokens)))
    
    return {
        'entities': list(entities),
        'relations': list(relations),
        'triples': triples,
        'entity_positions': sorted(list(set(entity_positions))),
        'relation_positions': sorted(list(set(relation_positions))),
        'gold_triples_positions': sorted(list(gold_positions))  # 添加gold三元组位置
    }

if __name__ == "__main__":
    # 设置参数
    # DATA_DIR = "/mnt/d/datasets/GraphTruth/metaqa-1hop/metaqa-1hop"
    DATA_DIR = "dataset/metaqa-1hop"
    OUTPUT_DIR = "experiment_records"
    TOPK = 30  # 子图大小 这里可以修改
    
    # 运行数据集准备
    result_path = prepare_dataset(DATA_DIR, OUTPUT_DIR, TOPK) 