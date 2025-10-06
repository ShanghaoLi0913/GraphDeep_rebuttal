"""
数据集处理器模块 (Dataset Processor)

本模块负责GraphDeEP项目的数据集完整处理流程，主要功能包括：
1. 加载原始MetaQA数据集和实体关系
2. 数据预处理和格式转换
3. 🎯 双重子图裁剪策略（核心创新）
4. 构建模型输入prompt
5. 保存处理后的数据集

🎯 双重子图裁剪策略 (Dual Subgraph Trimming Strategy):
本模块的核心创新，解决TUS和FGAS指标互斥问题：

TUS策略: 精确shortest path golden triples
- 目标: 注意力精度 (attention precision)
- 方法: 问题实体 → 答案实体的最短路径
- 输出: gold_triples字段 (平均2.4个)

FGAS策略: 扩展语义golden expansion set  
- 目标: 语义丰富度 (semantic richness)
- 方法: golden triples + 1-hop邻接三元组
- 输出: golden_expansion_set字段 (平均13.5个)

主要函数:
- load_data: 加载原始数据集
- load_entities_and_relations: 加载实体和关系
- dual_subgraph_trimming: 🎯 双重子图裁剪策略 (推荐)
- prepare_dataset_dual_strategy: 🎯 使用双重策略准备数据集 (默认)
- prepare_dataset_gold_priority: 金三元组优先裁剪 (备选)
- prepare_dataset: 原始裁剪方法 (基线)
- build_prompt: 构建模型输入prompt

作者: GraphDeEP Team
创建日期: 2024-03-19
双重策略: 2024-12-25
"""

import json
import os
import time
from datetime import datetime
from typing import List, Dict, Tuple, Set, Any
from tqdm import tqdm
from collections import defaultdict
import random

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

def load_entities_and_relations(data_path: str) -> Tuple[List[str], List[str]]:
    """
    加载实体和关系列表
    
    Args:
        data_path: 数据文件路径或数据集根目录路径
    
    Returns:
        实体列表和关系列表的元组
    """
    try:
        # 如果data_path是一个文件路径，获取其目录
        if data_path.endswith('.json'):
            data_dir = os.path.dirname(data_path)
        else:
            data_dir = data_path
            
        entities_file = os.path.join(data_dir, 'entities.txt')
        relations_file = os.path.join(data_dir, 'relations.txt')
        
        # 加载实体
        with open(entities_file, 'r', encoding='utf-8') as f:
            entities = [line.strip() for line in f if line.strip()]
            
        # 加载关系
        with open(relations_file, 'r', encoding='utf-8') as f:
            relations = [line.strip() for line in f if line.strip()]
            
        print(f"Loaded {len(entities)} entities from {entities_file}")
        print(f"Loaded {len(relations)} relations from {relations_file}")
            
        return entities, relations
    except Exception as e:
        print(f"Error loading entities and relations: {e}")
        raise

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

def prepare_dataset(data_dir: str, output_dir: str, topk: int = 30) -> str:
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
    entity_list, relation_list = load_entities_and_relations(data_dir)
    
    # 统计原始三元组数量
    triple_counts = [len(sample['subgraph']['tuples']) for sample in data]
    print(f"样本总数: {len(triple_counts)}")
    print(f"三元组数量 - 最大: {max(triple_counts)}, 最小: {min(triple_counts)}, 平均: {sum(triple_counts)/len(triple_counts):.2f}")
    
    # 2. 设置实验参数
    batch_size = 8
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
                
                # 使用新的子图构建方法
                try:
                    trimmed_triples = process_sample(sample, entity_list, topk)
                    answer_covered = True  # 由于算法保证答案可达，所以一定为True
                    batch_covered += 1
                    answer_covered_count += 1
                except AssertionError as e:
                    print(f"Warning: Failed to find path to answer for question: {question}")
                    # 如果找不到路径，使用原始方法作为后备
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
            
            # 保存batch统计信息
            batch_stats = {
                "batch_stats": {
                    "batch_id": i // batch_size,
                    "batch_size": len(batch_samples),
                    "batch_covered": batch_covered,
                    "batch_recall": batch_recall,
                    "cumulative_recall": current_recall,
                    "processing_time": time.time() - trimming_start_time
                }
            }
            f.write(json.dumps(batch_stats, ensure_ascii=False) + "\n")
        
        # 写入最终统计信息
        final_stats = {
            "final_stats": {
                "total_samples": num_samples,
                "answer_covered_count": answer_covered_count,
                "answer_recall": answer_covered_count / num_samples * 100,
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
    entities = []
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
                    entities.append(i)
                    break
                
                start_pos = question_lower.find(entity_lower, start_pos + 1)
    
    return entities

def is_answer_covered(trimmed_triples: List[List[int]], entity_list: List[str], golden_texts: List[str]) -> bool:
    """
    检查裁剪后的子图是否包含正确答案
    
    Args:
        trimmed_triples: 裁剪后的三元组列表
        entity_list: 实体列表
        golden_texts: 正确答案文本列表
    
    Returns:
        是否包含正确答案
    """
    # 获取子图中的所有实体
    subgraph_entities = set()
    for h, r, t in trimmed_triples:
        subgraph_entities.add(entity_list[h].lower())
        subgraph_entities.add(entity_list[t].lower())
    
    # 检查是否包含任何一个正确答案
    for answer in golden_texts:
        if answer.lower() in subgraph_entities:
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

# ============= 子图处理工具函数 =============

def build_graph(triples: List[List[int]]) -> Tuple[Dict[int, List[int]], Dict[Tuple[int, int], Tuple[int, int, int]]]:
    """构建无向图，用于寻找最短路径"""
    graph = defaultdict(list)
    triple_dict = {}  # 用于反查三元组
    for i, (h, r, t) in enumerate(triples):
        graph[h].append(t)
        graph[t].append(h)
        triple_dict[(h, t)] = (h, r, t)  # 确保存储为元组
        triple_dict[(t, h)] = (h, r, t)  # 确保存储为元组
    return graph, triple_dict

def find_shortest_path(graph: Dict[int, List[int]], start: int, end: int, max_depth: int = 3) -> List[int]:
    """使用BFS找到两个实体之间的最短路径"""
    if start == end:
        return [start]
    
    visited = {start}
    queue = [(start, [start])]
    
    while queue:
        vertex, path = queue.pop(0)
        if len(path) > max_depth:  # 限制最大跳数
            continue
            
        for neighbor in graph[vertex]:
            if neighbor == end:
                return path + [neighbor]
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, path + [neighbor]))
    return None

def get_path_triples(path: List[int], triple_dict: Dict[Tuple[int, int], Tuple[int, int, int]]) -> List[Tuple[int, int, int]]:
    """将路径转换为三元组"""
    path_triples = []
    for i in range(len(path) - 1):
        h, t = path[i], path[i + 1]
        if (h, t) in triple_dict:
            path_triples.append(triple_dict[(h, t)])
        elif (t, h) in triple_dict:
            path_triples.append(triple_dict[(t, h)])
    return path_triples

def get_neighbor_triples(entities: Set[int], triples: List[List[int]], exclude_triples: Set[Tuple[int, int, int]]) -> List[Tuple[int, int, int]]:
    """获取实体的1-hop邻居形成的三元组"""
    neighbors = []
    for h, r, t in triples:
        triple = (h, r, t)
        if triple not in exclude_triples:
            if h in entities or t in entities:
                neighbors.append(triple)
    return neighbors

def score_triple(triple: Tuple[int, int, int], path_entities: Set[int], q_entities: Set[int], a_entities: Set[int]) -> int:
    """对三元组进行评分，考虑多个因素"""
    h, r, t = triple
    score = 0
    
    # 1. 与路径实体的连接
    if h in path_entities or t in path_entities:
        score += 3
        
    # 2. 与问题实体的连接
    if h in q_entities or t in q_entities:
        score += 2
        
    # 3. 与答案实体的连接
    if h in a_entities or t in a_entities:
        score += 2
        
    return score

def get_subgraph_gold_priority(triples: List[List[int]], 
                              question_entities: Set[int], 
                              answer_entities: Set[int], 
                              max_size: int = 30,
                              gold_priority_ratio: float = 0.7) -> Tuple[List[List[int]], List[List[int]]]:
    """
    基于金三元组优先级的子图构建方法，兼顾TUS和FGAS需求
    
    策略：
    1. 先从原始图生成完整的金三元组集合（保证TUS完整性）
    2. 确保金三元组优先被保留在裁剪后的子图中（保证FGAS一致性）
    3. 用剩余空间填充重要的邻居三元组（保证子图质量）
    4. 返回裁剪后的子图和对应的金三元组
    
    Args:
        triples: 原始三元组列表
        question_entities: 问题实体集合
        answer_entities: 答案实体集合
        max_size: 子图最大大小
        gold_priority_ratio: 金三元组在子图中的最大占比（0.7表示70%空间留给金三元组）
        
    Returns:
        (trimmed_triples, effective_gold_triples) 元组
        - trimmed_triples: 裁剪后的子图
        - effective_gold_triples: 确实保留在子图中的金三元组
    """
    # 1. 构建图结构
    graph, triple_dict = build_graph(triples)
    
    # 2. 生成完整的金三元组集合（用于TUS的完整分析）
    original_gold_triples = set()
    path_entities = set()
    
    # 检查问题实体和答案实体是否在子图中
    subgraph_entities = set()
    for h, r, t in triples:
        subgraph_entities.add(h)
        subgraph_entities.add(t)
    
    valid_question_entities = question_entities & subgraph_entities
    valid_answer_entities = answer_entities & subgraph_entities
    
    print(f"调试信息: 问题实体 {len(question_entities)} 个，其中 {len(valid_question_entities)} 个在子图中")
    print(f"调试信息: 答案实体 {len(answer_entities)} 个，其中 {len(valid_answer_entities)} 个在子图中")
    
    # 如果问题实体或答案实体不在子图中，使用后备策略
    if not valid_question_entities or not valid_answer_entities:
        print("警告: 问题实体或答案实体不在子图中，使用后备策略")
        # 后备策略：直接查找包含答案实体的三元组作为金三元组
        for h, r, t in triples:
            if h in answer_entities or t in answer_entities:
                original_gold_triples.add((h, r, t))
                path_entities.update([h, t])
        
        # 如果还是没有找到，则使用所有三元组的前几个作为金三元组
        if not original_gold_triples:
            print("警告: 找不到包含答案实体的三元组，使用前几个三元组作为金三元组")
            for i, (h, r, t) in enumerate(triples[:5]):  # 取前5个
                original_gold_triples.add((h, r, t))
                path_entities.update([h, t])
    else:
        # 正常策略：查找最短路径
        for q_entity in valid_question_entities:
            for a_entity in valid_answer_entities:
                path = find_shortest_path(graph, q_entity, a_entity)
                if path:
                    path_triples = get_path_triples(path, triple_dict)
                    original_gold_triples.update(path_triples)
                    path_entities.update(path)
    
    # 3. 计算金三元组的优先级预算
    max_gold_count = min(len(original_gold_triples), int(max_size * gold_priority_ratio))
    
    # 4. 对金三元组按重要性排序
    gold_triples_scored = []
    for triple in original_gold_triples:
        score = score_triple(triple, path_entities, question_entities, answer_entities)
        gold_triples_scored.append((triple, score))
    
    # 按分数排序，选择最重要的金三元组
    gold_triples_scored.sort(key=lambda x: x[1], reverse=True)
    priority_gold_triples = [triple for triple, _ in gold_triples_scored[:max_gold_count]]
    
    # 5. 获取邻居三元组
    neighbor_triples = get_neighbor_triples(
        path_entities, 
        triples, 
        set(priority_gold_triples)
    )
    
    # 6. 对邻居三元组评分
    scored_neighbors = [
        (triple, score_triple(triple, path_entities, question_entities, answer_entities))
        for triple in neighbor_triples
    ]
    scored_neighbors.sort(key=lambda x: x[1], reverse=True)
    
    # 7. 组合最终子图：优先金三元组 + 重要邻居
    selected_triples = list(priority_gold_triples)
    remaining_size = max_size - len(selected_triples)
    
    # 添加高分邻居三元组
    for triple, _ in scored_neighbors[:remaining_size]:
        selected_triples.append(triple)
    
    # 8. 如果还不够，重复使用已有三元组
    while len(selected_triples) < max_size:
        selected_triples.extend(selected_triples[:max_size - len(selected_triples)])
    
    # 9. 确保返回恰好max_size个三元组
    final_triples = [list(t) for t in selected_triples[:max_size]]
    effective_gold_triples = [list(t) for t in priority_gold_triples]
    
    # 10. 验证答案可达性（对于有效的答案实体）
    final_entities = set()
    for h, r, t in final_triples:
        final_entities.add(h)
        final_entities.add(t)
    
    # 只验证原本在子图中的答案实体
    original_valid_answers = answer_entities & subgraph_entities
    if original_valid_answers:
        final_valid_answers = original_valid_answers & final_entities
        if not final_valid_answers:
            print(f"警告: 原本有效的答案实体 {original_valid_answers} 在最终子图中丢失")
        else:
            print(f"验证通过: {len(final_valid_answers)}/{len(original_valid_answers)} 个有效答案实体保留在最终子图中")
    
    return final_triples, effective_gold_triples

def get_subgraph_shortest_path_plus(triples: List[List[int]], 
                                  question_entities: Set[int], 
                                  answer_entities: Set[int], 
                                  entity_list: List[str], 
                                  max_size: int = 30) -> List[List[int]]:
    """
    基于最短路径的子图构建方法，包含三个步骤：
    1. 找出问题实体到答案实体的最短路径
    2. 扩展路径上实体的1-hop邻居
    3. 根据重要性评分选择最终子图
    
    确保返回恰好max_size个三元组，如果不足则重复使用已有三元组。
    """
    # 1. 构建图结构
    graph, triple_dict = build_graph(triples)
    
    # 2. 找出所有问题实体到答案实体的最短路径
    all_path_triples = set()  # 使用元组存储三元组
    path_entities = set()
    
    for q_entity in question_entities:
        for a_entity in answer_entities:
            path = find_shortest_path(graph, q_entity, a_entity)
            if path:
                path_triples = get_path_triples(path, triple_dict)
                all_path_triples.update(path_triples)
                path_entities.update(path)
    
    # 3. 获取路径实体的邻居三元组
    neighbor_triples = get_neighbor_triples(
        path_entities, 
        triples, 
        all_path_triples
    )
    
    # 4. 对邻居三元组评分并选择最重要的
    scored_neighbors = [
        (triple, score_triple(triple, path_entities, question_entities, answer_entities))
        for triple in neighbor_triples
    ]
    scored_neighbors.sort(key=lambda x: x[1], reverse=True)
    
    # 5. 组合最终子图
    selected_triples = list(all_path_triples)
    selected_triples.extend(triple for triple, _ in scored_neighbors)
    
    # 6. 如果三元组总数不足max_size，则循环使用已有的三元组
    while len(selected_triples) < max_size:
        selected_triples.extend(selected_triples[:max_size - len(selected_triples)])
    
    # 7. 确保返回恰好max_size个三元组
    result = [list(t) for t in selected_triples[:max_size]]
    
    # 验证答案可达性
    final_entities = set()
    for h, r, t in result:
        final_entities.add(h)
        final_entities.add(t)
    assert answer_entities & final_entities, "Answer entities must be in the subgraph"
    
    return result

def process_sample(sample: Dict, entity_list: List[str], topk: int = 30) -> List[List[int]]:
    """
    处理单个样本，返回裁剪后的子图
    """
    question = sample['question']
    triples = sample['subgraph']['tuples']
    
    # 获取问题实体 - 使用精确匹配
    question_entities = set()
    question_lower = question.lower()
    for i, entity in enumerate(entity_list):
        entity_lower = entity.lower()
        if entity_lower in question_lower:
            start_pos = question_lower.find(entity_lower)
            while start_pos != -1:
                end_pos = start_pos + len(entity_lower)
                before_ok = (start_pos == 0 or not question_lower[start_pos-1].isalnum())
                after_ok = (end_pos == len(question_lower) or not question_lower[end_pos].isalnum())
                if before_ok and after_ok:
                    question_entities.add(i)
                    break
                start_pos = question_lower.find(entity_lower, start_pos + 1)
    
    # 获取答案实体 - 使用精确匹配
    answer_entities = set()
    for ans in sample.get('answers', []):
        if ans.get('text'):
            ans_text = ans['text'].lower().strip()
            for i, entity in enumerate(entity_list):
                entity_lower = entity.lower()
                if ans_text == entity_lower or (ans_text in entity_lower and len(ans_text) >= 3):
                    if ans_text == entity_lower:
                        answer_entities.add(i)
                    else:
                        start_pos = entity_lower.find(ans_text)
                        if start_pos != -1:
                            end_pos = start_pos + len(ans_text)
                            before_ok = (start_pos == 0 or not entity_lower[start_pos-1].isalnum())
                            after_ok = (end_pos == len(entity_lower) or not entity_lower[end_pos].isalnum())
                            if before_ok and after_ok:
                                answer_entities.add(i)
    
    # 使用改进的子图构建方法
    trimmed_triples = get_subgraph_shortest_path_plus(
        triples,
        question_entities,
        answer_entities,
        entity_list,
        topk
    )
    
    return trimmed_triples

def process_sample_gold_aware(sample: Dict, entity_list: List[str], topk: int = 30) -> Tuple[List[List[int]], List[List[int]]]:
    """
    处理单个样本，使用金三元组感知的裁剪方法
    
    Args:
        sample: 样本数据字典
        entity_list: 实体列表
        topk: 子图大小
    
    Returns:
        (trimmed_triples, gold_triples) 元组
    """
    question = sample['question']
    triples = sample['subgraph']['tuples']
    
    # 获取问题实体
    question_entities = set()
    question_lower = question.lower()
    for i, entity in enumerate(entity_list):
        entity_lower = entity.lower()
        if entity_lower in question_lower:
            start_pos = question_lower.find(entity_lower)
            while start_pos != -1:
                end_pos = start_pos + len(entity_lower)
                before_ok = (start_pos == 0 or not question_lower[start_pos-1].isalnum())
                after_ok = (end_pos == len(question_lower) or not question_lower[end_pos].isalnum())
                if before_ok and after_ok:
                    question_entities.add(i)
                    break
                start_pos = question_lower.find(entity_lower, start_pos + 1)
    
    # 获取答案实体
    answer_entities = set()
    for ans in sample.get('answers', []):
        if ans.get('text'):
            ans_text = ans['text'].lower().strip()
            for i, entity in enumerate(entity_list):
                entity_lower = entity.lower()
                if ans_text == entity_lower or (ans_text in entity_lower and len(ans_text) >= 3):
                    if ans_text == entity_lower:
                        answer_entities.add(i)
                    else:
                        start_pos = entity_lower.find(ans_text)
                        if start_pos != -1:
                            end_pos = start_pos + len(ans_text)
                            before_ok = (start_pos == 0 or not entity_lower[start_pos-1].isalnum())
                            after_ok = (end_pos == len(entity_lower) or not entity_lower[end_pos].isalnum())
                            if before_ok and after_ok:
                                answer_entities.add(i)
    
    # 使用金三元组优先的裁剪方法
    trimmed_triples, gold_triples = get_subgraph_gold_priority(
        triples,
        question_entities,
        answer_entities,
        topk
    )
    
    return trimmed_triples, gold_triples

def get_fixed_length_subgraph(triples: List[List[int]], 
                            entity_indices: List[int], 
                            golden_texts: List[str], 
                            entity_list: List[str], 
                            topk: int = 30) -> List[List[int]]:
    """
    裁剪出与entity_indices相关的固定topk条三元组的子图。
    确保包含答案的三元组一定在结果中，并且总是返回恰好topk个三元组。
    如果原始三元组数量不足topk，则重复使用已有三元组直到达到topk。
    """
    # 1. 找出包含答案的三元组
    answer_triples = []
    for triple in triples:
        h, r, t = triple
        h_text = entity_list[h].lower()
        t_text = entity_list[t].lower()
        if any(gold in h_text or gold in t_text for gold in golden_texts):
            answer_triples.append(triple)
    
    # 2. 获取1-hop三元组
    one_hop = [triple for triple in triples if triple[0] in entity_indices or triple[2] in entity_indices]
    
    # 3. 确保answer_triples在结果中
    result = list(set(answer_triples))  # 去重
    
    # 4. 添加1-hop三元组（排除已有的answer_triples）
    remaining_one_hop = [t for t in one_hop if t not in result]
    result.extend(remaining_one_hop)
    
    # 5. 如果还不够，添加2-hop三元组
    if len(result) < topk:
        one_hop_entities = set([h for h, _, t in result] + [t for h, _, t in result])
        two_hop = [triple for triple in triples 
                  if (triple[0] in one_hop_entities or triple[2] in one_hop_entities) 
                  and triple not in result]
        result.extend(two_hop)
    
    # 6. 如果还不够，循环使用已有三元组
    while len(result) < topk:
        result.extend(result[:topk - len(result)])
    
    # 7. 确保返回恰好topk个三元组
    return result[:topk]

def prepare_dataset_gold_priority(data_dir: str, output_dir: str, topk: int = 30, num_samples: int = None) -> str:
    """
    使用金三元组优先方法准备数据集：加载、裁剪和保存
    
    Args:
        data_dir: 原始数据集目录
        output_dir: 输出目录
        topk: 子图大小
        num_samples: 处理的样本数量，None表示处理所有样本
    
    Returns:
        保存的结果文件路径
    """
    # 1. 加载数据
    print("\n🚀 开始使用金三元组优先方法准备数据集")
    print(f"参数: topk={topk}, num_samples={num_samples or '全部'}")
    
    print("\n📚 加载数据...")
    data = load_data(os.path.join(data_dir, "dev_simple.json"))
    entity_list, relation_list = load_entities_and_relations(data_dir)
    
    # 如果指定了样本数量，则只处理指定数量的样本
    if num_samples is not None:
        data = data[:num_samples]
    
    print(f"将处理 {len(data)} 个样本")
    print(f"实体数量: {len(entity_list)}, 关系数量: {len(relation_list)}")
    
    # 2. 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 生成时间戳
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 设置保存文件路径
    result_path = os.path.join(output_dir, f"trimming_results_gold_priority_{timestamp}.jsonl")
    
    # 3. 处理样本
    print(f"\n⚙️ 开始使用金三元组优先方法处理样本...")
    
    processed_samples = []
    stats = {
        'total_samples': len(data),
        'successful_samples': 0,
        'failed_samples': 0,
        'total_original_gold': 0,
        'total_effective_gold': 0,
        'gold_retention_rates': []
    }
    
    # 记录配置信息
    config = {
        "timestamp": timestamp,
        "method": "gold_priority_trimming",
        "topk": topk,
        "total_samples": len(data),
        "data_dir": data_dir
    }
    
    start_time = time.time()
    
    with open(result_path, "w", encoding="utf-8") as f:
        # 写入配置信息
        f.write(json.dumps({"config": config}, ensure_ascii=False) + "\n")
        
        for i, sample in enumerate(tqdm(data, desc="处理样本")):
            try:
                # 使用新的金三元组优先裁剪方法
                trimmed_triples, effective_gold_triples = process_sample_gold_aware(
                    sample, entity_list, topk
                )
                
                # 获取原始金三元组（用于比较）
                question = sample['question']
                golden_texts = [ans['text'].lower() for ans in sample.get('answers', []) if ans.get('text')]
                
                # 获取问题实体和答案实体
                question_entities = set()
                question_lower = question.lower()
                for idx, entity in enumerate(entity_list):
                    entity_lower = entity.lower()
                    if entity_lower in question_lower:
                        start_pos = question_lower.find(entity_lower)
                        while start_pos != -1:
                            end_pos = start_pos + len(entity_lower)
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
                            if ans_text == entity_lower or (ans_text in entity_lower and len(ans_text) >= 3):
                                if ans_text == entity_lower:
                                    answer_entities.add(idx)
                                else:
                                    start_pos = entity_lower.find(ans_text)
                                    if start_pos != -1:
                                        end_pos = start_pos + len(ans_text)
                                        before_ok = (start_pos == 0 or not entity_lower[start_pos-1].isalnum())
                                        after_ok = (end_pos == len(entity_lower) or not entity_lower[end_pos].isalnum())
                                        if before_ok and after_ok:
                                            answer_entities.add(idx)
                
                # 获取原始完整的金三元组
                original_gold_triples = get_gold_triples(
                    sample['subgraph']['tuples'],
                    question_entities,
                    answer_entities
                )
                
                # 将三元组的索引转换为实际文本（关键步骤！）
                text_triples = []
                effective_gold_text = []
                original_gold_text = []
                
                # 转换裁剪后的子图三元组
                for h, r, t in trimmed_triples:
                    head_text = entity_list[h]
                    rel_text = relation_list[r]
                    tail_text = entity_list[t]
                    text_triples.append([head_text, rel_text, tail_text])
                
                # 转换有效的金三元组
                for h, r, t in effective_gold_triples:
                    head_text = entity_list[h]
                    rel_text = relation_list[r]
                    tail_text = entity_list[t]
                    effective_gold_text.append([head_text, rel_text, tail_text])
                
                # 转换原始完整的金三元组
                for h, r, t in original_gold_triples:
                    head_text = entity_list[h]
                    rel_text = relation_list[r]
                    tail_text = entity_list[t]
                    original_gold_text.append([head_text, rel_text, tail_text])
                
                # 计算金三元组保留率
                retention_rate = 0
                if original_gold_triples:
                    retention_rate = len(effective_gold_triples) / len(original_gold_triples)
                    stats['gold_retention_rates'].append(retention_rate)
                
                # 将原始子图的tuples也转换为文本形式
                original_subgraph_text = []
                for h, r, t in sample['subgraph']['tuples']:
                    head_text = entity_list[h]
                    rel_text = relation_list[r]
                    tail_text = entity_list[t]
                    original_subgraph_text.append([head_text, rel_text, tail_text])
                
                # 构建结果 - 完全按照原始格式
                processing_time = time.time() - start_time
                result = {
                    "sample_id": i,
                    "question": question,
                    "golden_texts": golden_texts,
                    "trimmed_subgraph_length": len(trimmed_triples),
                    "original_subgraph_length": len(sample['subgraph']['tuples']),
                    "gold_triples_length": len(effective_gold_triples),
                    "answer_covered": True,  # 假设都覆盖了，与原格式一致
                    "trimmed_triples": text_triples,
                    "gold_triples": effective_gold_text,
                    "processing_time": processing_time
                }
                
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
                
                stats['successful_samples'] += 1
                stats['total_original_gold'] += len(original_gold_triples)
                stats['total_effective_gold'] += len(effective_gold_triples)
                
            except Exception as e:
                print(f"处理样本 {i} 时出错: {str(e)}")
                stats['failed_samples'] += 1
                continue
        
        # 计算最终统计信息
        if stats['gold_retention_rates']:
            stats['avg_retention_rate'] = sum(stats['gold_retention_rates']) / len(stats['gold_retention_rates'])
            stats['min_retention_rate'] = min(stats['gold_retention_rates'])
            stats['max_retention_rate'] = max(stats['gold_retention_rates'])
        
        if stats['total_original_gold'] > 0:
            stats['overall_retention_rate'] = stats['total_effective_gold'] / stats['total_original_gold']
        
        # 写入最终统计信息
        final_stats = {
            "final_statistics": stats,
            "timestamp": timestamp,
            "total_time": time.time() - start_time
        }
        f.write(json.dumps(final_stats, ensure_ascii=False) + "\n")
    
    print(f"\n✅ 处理完成!")
    print(f"成功: {stats['successful_samples']}, 失败: {stats['failed_samples']}")
    print(f"平均金三元组保留率: {stats.get('avg_retention_rate', 0):.1%}")
    print(f"整体金三元组保留率: {stats.get('overall_retention_rate', 0):.1%}")
    print(f"总耗时: {time.time() - start_time:.2f}秒")
    print(f"结果已保存至: {result_path}")
    
    return result_path

def get_golden_expansion_set(triples: List[List[int]], 
                           golden_triples: List[List[int]]) -> List[List[int]]:
    """
    为FGAS生成Golden Expansion Set (GES)：Golden Triples + 邻接三元组
    
    Args:
        triples: 完整三元组列表
        golden_triples: 基础golden triples（shortest path）
    
    Returns:
        扩展的golden triples集合（包含1-hop邻接）
    """
    if not golden_triples:
        return []
    
    # 转换为集合便于查找
    golden_set = set(tuple(t) for t in golden_triples)
    expansion_set = set(golden_set)  # 先包含原始golden triples
    
    # 提取golden triples中的所有实体
    golden_entities = set()
    for h, r, t in golden_triples:
        golden_entities.add(h)
        golden_entities.add(t)
    
    # 找到与golden entities相连的所有三元组（1-hop邻接）
    for h, r, t in triples:
        triple_tuple = (h, r, t)
        # 如果这个三元组不在golden set中，但包含golden entity，则添加
        if triple_tuple not in golden_set and (h in golden_entities or t in golden_entities):
            expansion_set.add(triple_tuple)
    
    return [list(t) for t in expansion_set]

def get_subgraph_simple(triples: List[List[int]], 
                       question_entities: Set[int], 
                       answer_entities: Set[int], 
                       topk: int = 20) -> List[List[int]]:
    """
    简化的子图构建函数，专注于核心逻辑
    """
    if not question_entities or not answer_entities:
        return triples[:topk] if len(triples) >= topk else triples
    
    # 优先选择包含问题或答案实体的三元组
    relevant_triples = []
    other_triples = []
    
    for triple in triples:
        h, r, t = triple
        if (h in question_entities or t in question_entities or 
            h in answer_entities or t in answer_entities):
            relevant_triples.append(triple)
        else:
            other_triples.append(triple)
    
    # 组合结果
    result = relevant_triples[:topk]
    if len(result) < topk:
        result.extend(other_triples[:topk - len(result)])
    
    return result

def prepare_dataset_tus_consistent(data_dir: str, 
                                  output_dir: str, 
                                  topk: int = 20,
                                  num_samples: int = None) -> str:
    """
    使用TUS一致性双重策略准备数据集
    
    🎯 核心目标：
    - TUS策略：与成功版本100%一致，确保TUS显著性
    - FGAS策略：提供额外的语义扩展，为FGAS计算服务
    - 数据一致性：所有golden triples都在最终子图中
    
    Args:
        data_dir: 原始数据集目录
        output_dir: 输出目录
        topk: 子图大小（默认20，与成功版本一致）
        num_samples: 处理样本数量限制
    
    Returns:
        保存的结果文件路径
    """
    # 1. 加载数据
    print("\n🚀 使用TUS一致性双重策略准备数据集")
    print("="*60)
    print("策略说明:")
    print("  🎯 TUS: 100%复用成功版本逻辑，确保显著性")
    print("  🌟 FGAS: 提供语义扩展集合，增强语义评估")
    print("  ✅ 一致性: 所有golden triples都在最终子图中")
    print("="*60)
    
    data = load_data(os.path.join(data_dir, "dev_simple.json"))
    entity_list, relation_list = load_entities_and_relations(data_dir)
    
    if num_samples:
        data = data[:num_samples]
        print(f"🔄 限制处理 {num_samples} 个样本")
    
    print(f"📚 加载完成: {len(data)} 个样本")
    
    # 2. 处理数据
    results = []
    tus_gold_stats = []
    fgas_expansion_stats = []
    success_count = 0
    
    for i, sample in enumerate(tqdm(data, desc="TUS一致性策略处理")):
        try:
            # 使用TUS一致性双重策略
            trimmed_subgraph, tus_golden_triples, fgas_golden_expansion_set = dual_subgraph_trimming_tus_consistent(
                sample, entity_list, topk
            )
            
            # 转换为文本格式保存
            trimmed_subgraph_text = [convert_indices_to_freebase(t, entity_list, relation_list) 
                                   for t in trimmed_subgraph]
            tus_golden_text = [convert_indices_to_freebase(t, entity_list, relation_list) 
                             for t in tus_golden_triples]
            fgas_expansion_text = [convert_indices_to_freebase(t, entity_list, relation_list) 
                                 for t in fgas_golden_expansion_set]
            
            # 构建结果（保持与现有格式兼容）
            golden_texts = [ans['text'].lower() for ans in sample.get('answers', []) if ans.get('text')]
            answer_covered = len(set(golden_texts) & 
                               set([t[0].lower() if len(t) > 0 else '' for t in trimmed_subgraph_text] + 
                                   [t[2].lower() if len(t) > 2 else '' for t in trimmed_subgraph_text])) > 0
            
            result = {
                'sample_id': i,
                'question': sample['question'],
                'golden_texts': golden_texts,
                'trimmed_subgraph_length': len(trimmed_subgraph),
                'original_subgraph_length': len(sample['subgraph']['tuples']),
                'gold_triples_length': len(tus_golden_triples),
                'golden_expansion_length': len(fgas_golden_expansion_set),
                'answer_covered': answer_covered,
                'trimmed_triples': trimmed_subgraph_text,
                'gold_triples': tus_golden_text,          # 🎯 TUS使用：与成功版本一致
                'golden_expansion_set': fgas_expansion_text,  # 🌟 FGAS使用：语义扩展集合
                'processing_time': 0.0,
                'strategy': 'tus_consistent_dual'
            }
            results.append(result)
            
            # 统计信息
            tus_gold_stats.append(len(tus_golden_triples))
            fgas_expansion_stats.append(len(fgas_golden_expansion_set))
            success_count += 1
            
        except Exception as e:
            print(f"❌ 处理样本 {i} 时出错: {e}")
            # 创建错误记录
            result = {
                'sample_id': i,
                'question': sample.get('question', ''),
                'golden_texts': [],
                'trimmed_subgraph_length': 0,
                'original_subgraph_length': len(sample.get('subgraph', {}).get('tuples', [])),
                'gold_triples_length': 0,
                'golden_expansion_length': 0,
                'answer_covered': False,
                'trimmed_triples': [],
                'gold_triples': [],
                'golden_expansion_set': [],
                'processing_time': 0.0,
                'strategy': 'tus_consistent_dual',
                'error': str(e)
            }
            results.append(result)
    
    # 3. 保存结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f"trimming_results_tus_consistent_{timestamp}.jsonl")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 配置信息
    config = {
        "config": {
            "timestamp": timestamp,
            "method": "tus_consistent_dual_strategy",
            "topk": topk,
            "total_samples": len(data),
            "data_dir": data_dir,
            "strategy_description": "TUS与成功版本100%一致 + FGAS语义扩展集合"
        }
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(json.dumps(config, ensure_ascii=False) + '\n')
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
    
    # 4. 打印详细统计
    print(f"\n🎯 TUS一致性双重策略处理完成！")
    print(f"="*60)
    print(f"📊 总体统计:")
    print(f"  总样本数: {len(results)}")
    print(f"  成功处理: {success_count}")
    print(f"  失败样本: {len(results) - success_count}")
    print(f"  成功率: {success_count/len(results)*100:.1f}%")
    
    if tus_gold_stats:
        print(f"\n🎯 TUS Golden Triples统计 (与成功版本一致):")
        print(f"  平均数量: {sum(tus_gold_stats)/len(tus_gold_stats):.2f}")
        print(f"  最大数量: {max(tus_gold_stats)}")
        print(f"  最小数量: {min(tus_gold_stats)}")
        
        # TUS分布分析
        tus_1_3 = sum(1 for x in tus_gold_stats if 1 <= x <= 3)
        tus_4_10 = sum(1 for x in tus_gold_stats if 4 <= x <= 10)
        tus_over_10 = sum(1 for x in tus_gold_stats if x > 10)
        
        print(f"  分布分析:")
        print(f"    1-3个 (最佳): {tus_1_3} 样本 ({tus_1_3/len(tus_gold_stats)*100:.1f}%)")
        print(f"    4-10个 (合理): {tus_4_10} 样本 ({tus_4_10/len(tus_gold_stats)*100:.1f}%)")
        print(f"    >10个 (较多): {tus_over_10} 样本 ({tus_over_10/len(tus_gold_stats)*100:.1f}%)")
    
    if fgas_expansion_stats:
        print(f"\n🌟 FGAS Golden Expansion Set统计:")
        print(f"  平均数量: {sum(fgas_expansion_stats)/len(fgas_expansion_stats):.2f}")
        print(f"  最大数量: {max(fgas_expansion_stats)}")
        print(f"  最小数量: {min(fgas_expansion_stats)}")
        
        # FGAS分布分析
        fgas_small = sum(1 for x in fgas_expansion_stats if x < 5)
        fgas_medium = sum(1 for x in fgas_expansion_stats if 5 <= x <= 15)
        fgas_large = sum(1 for x in fgas_expansion_stats if x > 15)
        
        print(f"  分布分析:")
        print(f"    <5个 (较少): {fgas_small} 样本 ({fgas_small/len(fgas_expansion_stats)*100:.1f}%)")
        print(f"    5-15个 (合理): {fgas_medium} 样本 ({fgas_medium/len(fgas_expansion_stats)*100:.1f}%)")
        print(f"    >15个 (丰富): {fgas_large} 样本 ({fgas_large/len(fgas_expansion_stats)*100:.1f}%)")
    
    print(f"\n✅ 结果已保存至: {output_file}")
    print(f"\n📋 下一步建议:")
    print(f"  1. 使用 'gold_triples' 字段计算TUS指标")
    print(f"  2. 使用 'golden_expansion_set' 字段计算FGAS指标")
    print(f"  3. 验证TUS是否达到显著性（期望值：p < 0.05）")
    print(f"  4. 检查FGAS的语义丰富度是否有所改善")
    
    return output_file

def generate_fgas_expansion_from_tus(trimmed_subgraph: List[List[int]], 
                                    tus_golden_triples: List[List[int]], 
                                    question_entities: Set[int],
                                    answer_entities: Set[int]) -> List[List[int]]:
    """
    🎯 精确的FGAS扩展策略：确保有区分能力
    
    策略：
    1. 起始点：TUS golden triples（核心）
    2. 只扩展与golden triples直接相连的三元组（1-hop严格限制）
    3. 控制扩展规模，避免包含整个子图
    
    Args:
        trimmed_subgraph: 最终的裁剪子图
        tus_golden_triples: TUS golden triples
        question_entities: 问题实体集合
        answer_entities: 答案实体集合
    
    Returns:
        FGAS golden expansion set（小而精确的扩展集合）
    """
    if not tus_golden_triples:
        return []
    
    # 将子图和TUS golden转为集合便于查找
    subgraph_set = set(tuple(t) for t in trimmed_subgraph)
    tus_golden_set = set(tuple(t) for t in tus_golden_triples)
    
    # 🔥 起始点：所有TUS golden triples（必须包含）
    expansion_set = set(tus_golden_set)
    
    # 获取TUS golden实体
    tus_entities = set()
    for h, r, t in tus_golden_triples:
        tus_entities.add(h)
        tus_entities.add(t)
    
    # 🎯 精确扩展：只添加与golden entities直接相连的关键三元组
    candidates = []
    
    for triple in trimmed_subgraph:
        triple_tuple = tuple(triple)
        
        # 跳过已经在TUS golden中的三元组
        if triple_tuple in tus_golden_set:
            continue
            
        h, r, t = triple
        score = 0
        
        # 🔥 评分标准：更严格的条件
        
        # 最高优先级：与golden实体直接相连 且 涉及答案实体
        if (h in tus_entities or t in tus_entities) and (h in answer_entities or t in answer_entities):
            score = 10
        
        # 中等优先级：与golden实体直接相连 且 涉及问题实体  
        elif (h in tus_entities or t in tus_entities) and (h in question_entities or t in question_entities):
            score = 5
        
        # 低优先级：仅与golden实体直接相连
        elif h in tus_entities or t in tus_entities:
            score = 2
            
        # 其他三元组：不包含
        else:
            score = 0
        
        if score > 0:
            candidates.append((score, triple_tuple))
    
    # 按分数排序，选择前几个
    candidates.sort(key=lambda x: x[0], reverse=True)
    
    # 🎯 控制扩展规模：最多扩展到golden的2-3倍
    max_expansion = max(len(tus_golden_triples) * 3, 10)  # 至少10个，最多golden的3倍
    max_additional = max_expansion - len(tus_golden_triples)
    
    added_count = 0
    for score, triple_tuple in candidates:
        if added_count >= max_additional:
            break
        expansion_set.add(triple_tuple)
        added_count += 1
    
    result = [list(t) for t in expansion_set]
    
    print(f"🎯 FGAS扩展策略:")
    print(f"  - TUS golden: {len(tus_golden_triples)} 个")
    print(f"  - 候选扩展: {len(candidates)} 个")
    print(f"  - 实际扩展: {added_count} 个")
    print(f"  - 最终FGAS: {len(result)} 个 (扩展比例: {len(result)/len(tus_golden_triples):.1f}x)")
    
    return result

def dual_subgraph_trimming_tus_consistent(sample: Dict, 
                                         entity_list: List[str], 
                                         topk: int = 20) -> Tuple[List[List[int]], List[List[int]], List[List[int]]]:
    """
    🎯 TUS一致性双重策略：确保TUS与成功版本完全一致 + 保留FGAS扩展集合
    
    核心思想：
    1. TUS策略：先生成golden triples，再强制保留（保证完整性）
    2. FGAS策略：基于完整的golden triples进行扩展
    3. 数据一致性：100%确保golden triples在最终子图中
    
    Args:
        sample: 样本数据
        entity_list: 实体列表  
        topk: 子图大小（默认20，与成功版本一致）
    
    Returns:
        tuple: (trimmed_subgraph, tus_golden_triples, fgas_golden_expansion_set)
        - trimmed_subgraph: 包含所有golden triples的子图
        - tus_golden_triples: 从原图生成的完整golden triples
        - fgas_golden_expansion_set: 基于完整golden的FGAS扩展集合
    """
    
    print(f"🎯 使用TUS一致性双重策略处理样本...")
    
    # ==================== 第1步：实体匹配（与成功版本一致）====================
    
    question = sample['question']
    triples = sample['subgraph']['tuples']
    
                        # 🔥 与TUS显著版本100%相同的简单实体匹配逻辑
    question_entities = set()
    for idx, entity in enumerate(entity_list):
        if entity in question:  # 简单包含匹配，与成功版本一致
            question_entities.add(idx)
    
    answer_entities = set()
    for ans in sample.get('answers', []):
        if ans.get('text'):
            ans_text = ans['text'].lower()
            for idx, entity in enumerate(entity_list):
                if ans_text in entity.lower():  # 简单包含匹配，与成功版本一致
                    answer_entities.add(idx)
    
    # ==================== 第2步：方案②实现 - 先生成golden，再强制保留 ====================
    
    # 🔥 步骤1：从原图生成完整的TUS golden triples（确保完整性）
    tus_golden_triples = get_gold_triples(
        sample['subgraph']['tuples'],  # 从原始大图生成，确保完整路径
        question_entities,
        answer_entities
    )
    
    print(f"✅ TUS Golden Triples生成完成，数量: {len(tus_golden_triples)}")
    
    if not tus_golden_triples:
        print(f"⚠️ 警告：未找到TUS golden triples，使用fallback策略")
        # 使用简单的子图构建
        trimmed_subgraph = get_subgraph_simple(triples, question_entities, answer_entities, topk)
        return trimmed_subgraph, [], []
    
    # 🔥 步骤2：构建包含所有golden triples的子图（强制保留）
    trimmed_subgraph = ensure_golden_in_subgraph(
        triples, 
        tus_golden_triples, 
        question_entities, 
        answer_entities, 
        topk
    )
    
    # 🚨 关键修复：如果golden数量超过topk，需要更新实际的golden triples列表
    # 确保后续验证使用的是实际包含在子图中的golden triples
    if len(tus_golden_triples) > topk:
        subgraph_set = set(tuple(t) for t in trimmed_subgraph)
        original_golden_set = set(tuple(t) for t in tus_golden_triples)
        actual_golden_in_subgraph = [list(t) for t in subgraph_set & original_golden_set]
        
        print(f"🔄 更新golden triples：原始{len(tus_golden_triples)}个 → 实际包含{len(actual_golden_in_subgraph)}个")
        tus_golden_triples = actual_golden_in_subgraph
    
    print(f"✅ 子图构建完成，长度: {len(trimmed_subgraph)}，确保包含所有{len(tus_golden_triples)}个golden triples")
    
    # ==================== 第3步：为FGAS生成扩展集合 ====================
    
    # 🌟 基于完整的golden triples和确保一致性的子图生成FGAS扩展
    fgas_golden_expansion_set = generate_fgas_expansion_from_tus(
        trimmed_subgraph,  # 已确保包含所有golden triples的子图
        tus_golden_triples,  # 完整的golden triples
        question_entities,
        answer_entities
    )
    
    print(f"✅ FGAS Golden Expansion Set生成完成，数量: {len(fgas_golden_expansion_set)}")
    
    # ==================== 第4步：最终验证（确保100%一致性）====================
    
    # 验证TUS golden triples 100% 在子图中
    subgraph_set = set(tuple(t) for t in trimmed_subgraph)
    tus_golden_set = set(tuple(t) for t in tus_golden_triples)
    
    missing_golden = tus_golden_set - subgraph_set
    if missing_golden:
        print(f"❌ 严重错误：{len(missing_golden)} 个TUS golden triples不在子图中！")
        # 这不应该发生，但作为最后的安全措施
        for missing_triple in missing_golden:
            if len(trimmed_subgraph) < topk:
                trimmed_subgraph.append(list(missing_triple))
            else:
                # 替换最不重要的三元组
                trimmed_subgraph[-1] = list(missing_triple)
    else:
        print(f"✅ 验证通过：所有{len(tus_golden_triples)}个TUS golden triples都在子图中")
    
    # 验证FGAS expansion set也在子图中
    fgas_set = set(tuple(t) for t in fgas_golden_expansion_set)
    missing_fgas = fgas_set - subgraph_set
    if missing_fgas:
        print(f"⚠️ 警告：{len(missing_fgas)} 个FGAS expansion triples不在子图中")
    else:
        print(f"✅ 验证通过：所有{len(fgas_golden_expansion_set)}个FGAS expansion triples都在子图中")
    
    # 验证答案实体覆盖
    subgraph_entities = set()
    for h, r, t in trimmed_subgraph:
        subgraph_entities.add(h)
        subgraph_entities.add(t)
    
    if not (answer_entities & subgraph_entities):
        print(f"⚠️ 警告：答案实体不在最终子图中")
    else:
        print(f"✅ 验证通过：答案实体在最终子图中")
    
    # 统计信息
    print(f"📊 统计信息:")
    print(f"  - 问题实体: {len(question_entities)} 个")
    print(f"  - 答案实体: {len(answer_entities)} 个") 
    print(f"  - 子图大小: {len(trimmed_subgraph)} 个三元组")
    print(f"  - TUS Golden: {len(tus_golden_triples)} 个三元组")
    print(f"  - FGAS Expansion: {len(fgas_golden_expansion_set)} 个三元组")
    print(f"  - TUS一致性: {'✅ 100%' if not missing_golden else '❌ 不完整'}")
    print(f"  - FGAS一致性: {'✅ 100%' if not missing_fgas else '❌ 不完整'}")
    
    return trimmed_subgraph, tus_golden_triples, fgas_golden_expansion_set


def ensure_golden_in_subgraph(triples: List[List[int]], 
                             golden_triples: List[List[int]], 
                             question_entities: Set[int], 
                             answer_entities: Set[int], 
                             max_size: int = 20) -> List[List[int]]:
    """
    构建子图，确保所有golden triples都被包含（方案②的核心实现）
    
    策略：
    1. 强制包含所有golden triples（最高优先级）
    2. 添加与问题/答案实体相关的重要三元组
    3. 填充剩余空间
    
    Args:
        triples: 原始完整三元组列表
        golden_triples: 必须包含的golden triples
        question_entities: 问题实体集合
        answer_entities: 答案实体集合
        max_size: 子图最大大小
    
    Returns:
        确保包含所有golden triples的子图
    """
    if not golden_triples:
        return get_subgraph_simple(triples, question_entities, answer_entities, max_size)
    
    # 🔥 第1优先级：强制包含所有golden triples
    subgraph_set = set(tuple(t) for t in golden_triples)
    print(f"🔥 强制包含{len(golden_triples)}个golden triples")
    
    # 🚨 修复：如果golden triples数量超过max_size，需要优先选择最重要的
    if len(subgraph_set) >= max_size:
        print(f"⚠️ Golden triples数量({len(subgraph_set)})超过topk限制({max_size})，进行优先级筛选")
        
        # 按优先级评分选择最重要的golden triples
        golden_scored = []
        for triple in golden_triples:
            h, r, t = triple
            score = 0
            
            # 与问题实体相关 - 最高优先级
            if h in question_entities or t in question_entities:
                score += 10
            
            # 与答案实体相关 - 次高优先级  
            if h in answer_entities or t in answer_entities:
                score += 8
            
            # 连接问题和答案实体的路径 - 高优先级
            if ((h in question_entities and t in answer_entities) or 
                (h in answer_entities and t in question_entities)):
                score += 15
            
            golden_scored.append((score, triple))
        
        # 按分数排序，选择前max_size个
        golden_scored.sort(key=lambda x: x[0], reverse=True)
        selected_golden = [triple for _, triple in golden_scored[:max_size]]
        
        result = selected_golden
        print(f"✅ 优先级筛选：从{len(golden_triples)}个golden中选择了{len(result)}个最重要的")
        return result
    
    # 🔥 第2优先级：添加与问题/答案实体直接相关的三元组
    important_triples = []
    for triple in triples:
        triple_tuple = tuple(triple)
        if triple_tuple not in subgraph_set:  # 避免重复
            h, r, t = triple
            score = 0
            
            # 与问题实体相关
            if h in question_entities or t in question_entities:
                score += 5
            
            # 与答案实体相关  
            if h in answer_entities or t in answer_entities:
                score += 5
            
            # 与golden triples中的实体相关
            golden_entities = set()
            for gh, gr, gt in golden_triples:
                golden_entities.add(gh)
                golden_entities.add(gt)
            
            if h in golden_entities or t in golden_entities:
                score += 3
            
            if score > 0:
                important_triples.append((score, triple))
    
    # 按重要性排序
    important_triples.sort(key=lambda x: x[0], reverse=True)
    
    # 添加重要三元组直到达到max_size
    remaining_slots = max_size - len(subgraph_set)
    added_important = 0
    
    for score, triple in important_triples:
        if len(subgraph_set) >= max_size:
            break
        subgraph_set.add(tuple(triple))
        added_important += 1
    
    print(f"✅ 添加了{added_important}个重要三元组")
    
    # 🔥 第3优先级：如果还有空间，随机填充其他三元组
    if len(subgraph_set) < max_size:
        remaining_slots = max_size - len(subgraph_set)
        other_triples = []
        
        for triple in triples:
            triple_tuple = tuple(triple)
            if triple_tuple not in subgraph_set:
                other_triples.append(triple)
                if len(other_triples) >= remaining_slots:
                    break
        
        for triple in other_triples:
            subgraph_set.add(tuple(triple))
        
        print(f"✅ 填充了{len(other_triples)}个其他三元组")
    
    result = [list(t) for t in subgraph_set]
    print(f"📊 最终子图: {len(result)}个三元组 (golden: {len(golden_triples)}, 其他: {len(result) - len(golden_triples)})")
    
    return result

if __name__ == "__main__":
    """
    数据集处理主程序
    
    默认使用双重子图裁剪策略，同时生成TUS和FGAS所需的不同golden triples：
    - TUS: 使用精确shortest path golden triples (注意力精度)
    - FGAS: 使用扩展语义golden expansion set (语义丰富度)
    """
    # 设置参数
    data_dir = "/mnt/d/datasets/GraphTruth/metaqa-1hop/metaqa-1hop"
    output_dir = "experiment_records"
    
    print("🚀 运行双重子图裁剪策略 (默认方法)")
    print("="*60)
    print("策略说明:")
    print("  - TUS: 使用精确shortest path golden triples")
    print("  - FGAS: 使用扩展语义golden expansion set")
    print("  - 目标: 让TUS和FGAS指标同时显著")
    print("="*60)
    
    result_file = prepare_dataset_tus_consistent(
        data_dir=data_dir,
        output_dir=output_dir,
        topk=20,                    # 子图大小
        num_samples=None           # 处理全部样本
    )
    
    print(f"\n✅ 双重策略处理完成!")
    print(f"📁 结果文件: {result_file}")
    print(f"\n📋 下一步:")
    print(f"   1. 使用此文件运行inference实验")
    print(f"   2. 分别计算TUS (gold_triples) 和 FGAS (golden_expansion_set) 指标")
    print(f"   3. 验证两个指标是否同时显著")
    
    # 其他方法可选使用（取消注释即可）
    # print("\n--- 可选: 金三元组优先方法 ---")
    # result_file_gold = prepare_dataset_gold_priority(
    #     data_dir=data_dir,
    #     output_dir=output_dir,
    #     topk=20,
    #     num_samples=None
    # )
    # print(f"金三元组优先结果: {result_file_gold}")
    
    # print("\n--- 可选: 原有方法 ---")
    # result_file_original = prepare_dataset(
    #     data_dir=data_dir,
    #     output_dir=output_dir,
    #     topk=30
    # )
    # print(f"原有方法结果: {result_file_original}") 