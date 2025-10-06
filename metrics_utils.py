"""
评估指标工具模块 (Metrics Utilities)

本模块实现了基本的评估指标计算。主要功能包括：
1. 计算子图覆盖率和其他统计指标
2. 计算Hit@1指标
3. 路径提取工具函数

主要函数:
- calculate_coverage: 计算覆盖率
- calculate_hit_at_1: 计算Hit@1得分
- extract_paths: 路径提取工具

作者: [Your Name]
创建日期: 2024-03-19
"""

from typing import List, Set, Dict, Tuple, Any
from collections import defaultdict
import re

def calculate_coverage(subgraph: List[List], original_graph: List[List]) -> Dict[str, float]:
    """
    计算子图的覆盖率统计
    
    Args:
        subgraph: 选择的子图三元组列表
        original_graph: 原始图的三元组列表
    
    Returns:
        包含不同覆盖率指标的字典
    """
    # 将列表转换为元组以便进行集合操作
    subgraph_set = set(tuple(triple) for triple in subgraph)
    original_set = set(tuple(triple) for triple in original_graph)
    
    # 计算三元组覆盖率
    triple_coverage = len(subgraph_set) / len(original_set) if original_set else 0.0
    
    # 提取实体和关系
    subgraph_entities = set()
    subgraph_relations = set()
    for h, r, t in subgraph:
        subgraph_entities.add(h)
        subgraph_entities.add(t)
        subgraph_relations.add(r)
    
    original_entities = set()
    original_relations = set()
    for h, r, t in original_graph:
        original_entities.add(h)
        original_entities.add(t)
        original_relations.add(r)
    
    # 计算实体和关系覆盖率
    entity_coverage = len(subgraph_entities) / len(original_entities) if original_entities else 0.0
    relation_coverage = len(subgraph_relations) / len(original_relations) if original_relations else 0.0
    
    return {
        'triple_coverage': triple_coverage,
        'entity_coverage': entity_coverage,
        'relation_coverage': relation_coverage
    }

def extract_paths(triples: List[Tuple], 
                 start_entities: Set[int], 
                 end_entities: Set[int], 
                 max_length: int = 3) -> List[List[Tuple]]:
    """
    从三元组中提取路径
    
    Args:
        triples: 三元组列表
        start_entities: 起始实体集合
        end_entities: 目标实体集合
        max_length: 最大路径长度
    
    Returns:
        路径列表，每个路径是三元组列表
    """
    # 构建图结构
    graph = defaultdict(list)
    triple_dict = {}
    for h, r, t in triples:
        graph[h].append(t)
        graph[t].append(h)
        triple_dict[(h, t)] = (h, r, t)
        triple_dict[(t, h)] = (h, r, t)
    
    paths = []
    for start in start_entities:
        for end in end_entities:
            # 使用BFS找路径
            visited = {start}
            queue = [(start, [start])]
            
            while queue:
                vertex, path = queue.pop(0)
                if len(path) > max_length:
                    continue
                    
                if vertex == end:
                    # 转换为三元组路径
                    triple_path = []
                    for i in range(len(path) - 1):
                        h, t = path[i], path[i + 1]
                        if (h, t) in triple_dict:
                            triple_path.append(triple_dict[(h, t)])
                        elif (t, h) in triple_dict:
                            triple_path.append(triple_dict[(t, h)])
                    paths.append(triple_path)
                    continue
                
                for neighbor in graph[vertex]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append((neighbor, path + [neighbor]))
    
    return paths

def calculate_hit_at_1(model_prediction: str, correct_answers: List[str]) -> float:
    """
    计算Hit@1指标
    
    Args:
        model_prediction: 模型的第一个预测（置信度最高的预测）
        correct_answers: 正确答案列表
    
    Returns:
        如果模型的第一个预测在正确答案列表中返回1.0，否则返回0.0
    """
    # 标准化处理
    model_prediction = model_prediction.lower().strip()
    correct_answers = [ans.lower().strip() for ans in correct_answers]
    
    if model_prediction in correct_answers:
        return 1.0
    return 0.0

def extract_answer(generated_text: str) -> str:
    """
    从生成的文本中提取实际答案
    
    Args:
        generated_text: 模型生成的完整文本
    
    Returns:
        提取出的答案（去除了无关文本）
    """
    # 去除"ans:"等前缀
    text = re.sub(r'^.*?ans:\s*', '', generated_text, flags=re.IGNORECASE)
    
    # 去除"Here are the movies..."等引导语
    text = re.sub(r'^.*?here are.*?:', '', text, flags=re.IGNORECASE)
    
    # 如果有bullet points，提取第一个
    if '*' in text:
        text = text.split('*')[1].strip()
    
    # 去除年份和括号内容
    text = re.sub(r'\(\d{4}\)', '', text)
    text = re.sub(r'\(.*?\)', '', text)
    
    # 去除多余空格和标点
    text = text.strip('* \t\n.,')
    
    return text

def evaluate_results(predictions: List[Dict]) -> Dict:
    """
    评估整体结果
    
    Args:
        predictions: 预测结果列表
        
    Returns:
        包含Hit@1评估指标的字典
    """
    total_samples = len(predictions)
    hit_at_1_count = sum(1 for p in predictions if p['metrics']['hit_at_1'])
    
    return {
        'total_samples': total_samples,
        'hit_at_1': hit_at_1_count / total_samples * 100  # 转换为百分比
    } 