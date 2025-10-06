"""
子图构建工具模块 (Subgraph Construction Utilities)

本模块负责子图的构建和处理，包括：
1. 最短路径搜索
2. 子图裁剪
3. 邻居扩展

作者: [Your Name]
创建日期: 2024-03-19
"""

from collections import defaultdict
from typing import List, Set, Dict, Tuple, Any
import random

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
    # exclude_triples已经是元组集合，不需要转换
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

def get_subgraph_shortest_path_plus(triples: List[List[int]], 
                                  question_entities: Set[int], 
                                  answer_entities: Set[int], 
                                  entity_list: List[str], 
                                  max_size: int = 20) -> List[List[int]]:
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
                # path_triples中的每个元素已经是元组
                all_path_triples.update(path_triples)
                path_entities.update(path)
    
    # 3. 获取路径实体的邻居三元组
    neighbor_triples = get_neighbor_triples(
        path_entities, 
        triples, 
        all_path_triples  # 直接传递元组集合
    )
    
    # 4. 对邻居三元组评分并选择最重要的
    scored_neighbors = [
        (triple, score_triple(triple, path_entities, question_entities, answer_entities))
        for triple in neighbor_triples
    ]
    scored_neighbors.sort(key=lambda x: x[1], reverse=True)
    
    # 5. 组合最终子图
    selected_triples = list(all_path_triples)  # 转换为列表但保持元组形式
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

def filter_subgraph_by_hop(triples: List[List[int]], entity_indices: List[int]) -> List[List[int]]:
    """
    只保留与entity_indices相关的1-hop三元组。
    """
    selected = set(entity_indices)
    result = []
    for h, r, t in triples:
        if h in selected or t in selected:
            result.append([h, r, t])
    return result

def get_fixed_length_subgraph(triples: List[List[int]], 
                            entity_indices: List[int], 
                            golden_texts: List[str], 
                            entity_list: List[str], 
                            topk: int = 40) -> List[List[int]]:
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
    
    # 6. 如果还不够，添加剩余的三元组
    remaining = [t for t in triples if t not in result]
    result.extend(remaining)
    
    # 7. 如果三元组总数仍然不足topk，则循环使用已有的三元组
    # 确保result非空
    if not result:
        result = [triples[0]] if triples else [[0, 0, 0]]  # 如果triples为空，使用默认三元组
        
    while len(result) < topk:
        needed = topk - len(result)
        result.extend(result[:needed])  # 添加需要的数量
    
    # 8. 确保返回恰好topk个三元组
    return result[:topk]

def process_sample(sample: Dict, entity_list: List[str], topk: int = 20) -> List[List[int]]:
    """
    处理单个样本，返回裁剪后的子图
    
    Args:
        sample: 样本数据字典
        entity_list: 实体列表
        topk: 子图大小
    
    Returns:
        裁剪后的子图三元组列表
    """
    question = sample['question']
    triples = sample['subgraph']['tuples']
    
    # 获取问题实体 - 使用精确匹配
    question_entities = set()
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
                # 精确匹配：答案文本必须与实体完全匹配或实体包含答案文本
                if ans_text == entity_lower or (ans_text in entity_lower and len(ans_text) >= 3):
                    # 对于包含关系，确保是完整单词匹配
                    if ans_text == entity_lower:
                        answer_entities.add(i)
                    else:
                        # 检查是否为完整单词
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

class PathGuidedSelector:
    """路径引导的子图选择器"""
    def __init__(self, max_size: int = 20, max_depth: int = 3):  # 默认子图大小设为20
        self.max_size = max_size
        self.max_depth = max_depth 