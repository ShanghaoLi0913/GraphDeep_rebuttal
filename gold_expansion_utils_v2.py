#!/usr/bin/env python3
"""
Gold Expansion Set 工具模块 v2.0 (改进版)

为FGAS计算构建语义丰富的三元组扩展集合，解决MetaQA-2hop中GASS分数分布异常问题。

改进版本特性:
- 基于InfoScore的信息量评估 (IDF + 语义关系权重)
- 基于DiversityBonus的多样性保证 (余弦相似度)
- 贪心选择算法避免语义冗余
- 针对MetaQA-2hop优化的参数设置 (max_size=8, max_per_entity=2)

核心算法:
1. 候选三元组发现 (与核心实体相关)
2. InfoScore计算 (实体/关系IDF + 语义关系加分)
3. 嵌入向量生成 (SentenceTransformer)
4. 贪心选择 (InfoScore + DiversityBonus)
5. 实体级别限制控制

预期效果:
- GES大小从10-12减少到6-8
- 平均余弦相似度从>0.9降低到~0.7
- GASS分布从集中在1.0转为正常分布0.4-0.6

作者: GraphDeEP Team
创建日期: 2025-01-15
版本: 2.0
"""

from typing import List, Dict, Set, Tuple, Any
import json
import math
import numpy as np
from collections import Counter
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import logging

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 全局嵌入模型（延迟加载）
_embedding_model = None

def get_embedding_model():
    """延迟加载嵌入模型"""
    global _embedding_model
    if _embedding_model is None:
        logger.info("Loading SentenceTransformer model for diversity calculation...")
        _embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    return _embedding_model

# 定义语义强的关系类型
SEMANTIC_RICH_RELATIONS = {
    'directed_by', 'acted_in', 'genre', 'release_year', 'language',
    'produced_by', 'written_by', 'cinematography', 'music_by',
    'starring', 'has_genre', 'release_date', 'runtime', 'budget',
    'award', 'nomination', 'country', 'company', 'sequel_to',
    'prequel_to', 'based_on', 'remake_of', 'character', 'role'
}

# 改进版本的默认参数（针对MetaQA-2hop优化）
DEFAULT_MAX_EXPANSION_SIZE = 20  # 与v1保持一致，最多20条
DEFAULT_MAX_PER_ENTITY = 3       # 与v1保持一致，每个核心实体最多3条
DEFAULT_SELECTION_THRESHOLD = 0.5  # 选择阈值

def calculate_idf_scores(all_triples: List[List[str]]) -> Dict[str, float]:
    """
    计算实体和关系的IDF分数
    
    Args:
        all_triples: 所有三元组列表
        
    Returns:
        IDF分数字典 {entity/relation: idf_score}
    """
    # 统计每个实体/关系的出现次数
    entity_counts = Counter()
    relation_counts = Counter()
    
    total_triples = len(all_triples)
    
    for h, r, t in all_triples:
        entity_counts[h] += 1
        entity_counts[t] += 1
        relation_counts[r] += 1
    
    # 计算IDF分数
    idf_scores = {}
    
    # 实体IDF
    for entity, count in entity_counts.items():
        idf_scores[entity] = math.log(total_triples / (count + 1))
    
    # 关系IDF
    for relation, count in relation_counts.items():
        idf_scores[relation] = math.log(total_triples / (count + 1))
    
    return idf_scores

def is_semantic_relation(relation: str) -> bool:
    """
    判断关系是否具有丰富的语义表达
    
    Args:
        relation: 关系名称
        
    Returns:
        是否为语义丰富的关系
    """
    relation_lower = relation.lower()
    # 检查是否包含语义丰富的关键词
    for semantic_rel in SEMANTIC_RICH_RELATIONS:
        if semantic_rel in relation_lower:
            return True
    return False

def calculate_info_score(triple: List[str], idf_scores: Dict[str, float]) -> float:
    """
    计算三元组的信息量分数
    
    Args:
        triple: 三元组 [h, r, t]
        idf_scores: IDF分数字典
        
    Returns:
        信息量分数
    """
    h, r, t = triple
    
    # 综合头实体、关系、尾实体的IDF分数
    h_idf = idf_scores.get(h, 0)
    r_idf = idf_scores.get(r, 0)
    t_idf = idf_scores.get(t, 0)
    
    # 加权平均：关系权重更高
    info_score = 0.3 * h_idf + 0.4 * r_idf + 0.3 * t_idf
    
    # 语义丰富关系额外加分
    if is_semantic_relation(r):
        info_score += 0.5
    
    return info_score

def calculate_diversity_bonus(candidate_triple: List[str], 
                            selected_triples: List[List[str]], 
                            triple_embeddings: Dict[str, np.ndarray]) -> float:
    """
    计算候选三元组相对于已选择三元组的多样性奖励
    
    Args:
        candidate_triple: 候选三元组
        selected_triples: 已选择的三元组列表
        triple_embeddings: 三元组嵌入向量字典
        
    Returns:
        多样性奖励分数
    """
    if not selected_triples:
        return 1.0  # 第一个三元组没有多样性惩罚
    
    candidate_key = tuple(candidate_triple)
    if candidate_key not in triple_embeddings:
        return 0.5  # 无法计算相似度时给中等分数
    
    candidate_embedding = triple_embeddings[candidate_key]
    
    # 计算与所有已选择三元组的最大相似度
    max_similarity = 0.0
    
    for selected_triple in selected_triples:
        selected_key = tuple(selected_triple)
        if selected_key in triple_embeddings:
            selected_embedding = triple_embeddings[selected_key]
            similarity = cosine_similarity(
                candidate_embedding.reshape(1, -1),
                selected_embedding.reshape(1, -1)
            )[0][0]
            max_similarity = max(max_similarity, similarity)
    
    # 多样性奖励 = 1 - max_similarity
    diversity_bonus = 1.0 - max_similarity
    
    return diversity_bonus

def create_triple_embeddings(triples: List[List[str]]) -> Dict[str, np.ndarray]:
    """
    为三元组创建嵌入向量
    
    Args:
        triples: 三元组列表
        
    Returns:
        三元组嵌入字典
    """
    model = get_embedding_model()
    
    # 将三元组转换为文本
    triple_texts = []
    triple_keys = []
    
    for triple in triples:
        h, r, t = triple
        # 构造语义化的三元组文本
        text = f"{h} {r} {t}"
        triple_texts.append(text)
        triple_keys.append(tuple(triple))
    
    # 生成嵌入
    embeddings = model.encode(triple_texts)
    
    # 创建映射字典
    triple_embeddings = {}
    for i, key in enumerate(triple_keys):
        triple_embeddings[key] = embeddings[i]
    
    return triple_embeddings

def extract_core_entities(gold_triples: List[List[str]]) -> Set[str]:
    """
    从gold triples中提取核心实体
    
    Args:
        gold_triples: 原始gold三元组列表 [[h, r, t], ...]
        
    Returns:
        核心实体集合
    """
    core_entities = set()
    for h, r, t in gold_triples:
        core_entities.add(h)
        core_entities.add(t)
    return core_entities

def find_candidate_triples(core_entities: Set[str], 
                         all_triples: List[List[str]], 
                         gold_triples: List[List[str]]) -> List[List[str]]:
    """
    查找候选扩展三元组
    
    Args:
        core_entities: 核心实体集合
        all_triples: 完整的三元组列表
        gold_triples: 原始gold三元组（需要排除）
        
    Returns:
        候选三元组列表
    """
    # 将gold triples转换为集合以便快速查找
    gold_set = set(tuple(triple) for triple in gold_triples)
    
    candidate_triples = []
    
    for triple in all_triples:
        h, r, t = triple
        triple_tuple = tuple(triple)
        
        # 跳过已在gold triples中的
        if triple_tuple in gold_set:
            continue
        
        # 检查是否与核心实体相关
        if h in core_entities or t in core_entities:
            candidate_triples.append(triple)
    
    return candidate_triples

def create_gold_expansion_set_v2(gold_triples: List[List[str]], 
                               trimmed_triples: List[List[str]],
                               max_expansion_size: int = DEFAULT_MAX_EXPANSION_SIZE,
                               max_per_entity: int = DEFAULT_MAX_PER_ENTITY,
                               selection_threshold: float = DEFAULT_SELECTION_THRESHOLD) -> List[List[str]]:
    """
    创建改进版的gold_expansion_set (v2.0)
    
    使用InfoScore和DiversityBonus的贪心选择算法
    
    Args:
        gold_triples: 原始gold三元组
        trimmed_triples: 完整的trimmed子图三元组
        max_expansion_size: gold_expansion_set的最大大小
        max_per_entity: 每个核心实体最多扩展的三元组数
        selection_threshold: 选择阈值
        
    Returns:
        gold_expansion_set三元组列表
    """
    logger.info(f"Creating GES v2.0 with improved algorithm")
    logger.info(f"Parameters: max_size={max_expansion_size}, max_per_entity={max_per_entity}, threshold={selection_threshold}")

    # 如果原始 gold triples 已经达到或超过最大限制，则直接返回，视作完整 GES
    if len(gold_triples) >= max_expansion_size:
        logger.info(
            "Gold triples length %d ≥ max_expansion_size %d. "
            "Skip expansion and use gold triples as GES.",
            len(gold_triples), max_expansion_size
        )
        return gold_triples
    
    # Step 1: 提取核心实体
    core_entities = extract_core_entities(gold_triples)
    logger.info(f"Found {len(core_entities)} core entities: {core_entities}")
    
    # Step 2: 查找候选三元组
    candidate_triples = find_candidate_triples(core_entities, trimmed_triples, gold_triples)
    logger.info(f"Found {len(candidate_triples)} candidate triples")
    
    if not candidate_triples:
        logger.warning("No candidate triples found, returning original gold triples")
        return gold_triples
    
    # Step 3: 计算IDF分数
    idf_scores = calculate_idf_scores(trimmed_triples)
    logger.info(f"Calculated IDF scores for {len(idf_scores)} entities/relations")
    
    # Step 4: 为所有相关三元组创建嵌入
    all_relevant_triples = gold_triples + candidate_triples
    triple_embeddings = create_triple_embeddings(all_relevant_triples)
    logger.info(f"Created embeddings for {len(triple_embeddings)} triples")
    
    # Step 5: 贪心选择算法
    selected_expansion = []
    entity_expansion_count = {entity: 0 for entity in core_entities}
    
    # 预计算所有候选三元组的InfoScore
    candidate_info_scores = []
    for candidate in candidate_triples:
        info_score = calculate_info_score(candidate, idf_scores)
        candidate_info_scores.append((candidate, info_score))
    
    # 按InfoScore排序
    candidate_info_scores.sort(key=lambda x: x[1], reverse=True)
    logger.info(f"Sorted {len(candidate_info_scores)} candidates by InfoScore")
    
    # 贪心选择
    for candidate, info_score in candidate_info_scores:
        # 检查是否达到总体限制
        if len(selected_expansion) >= max_expansion_size - len(gold_triples):
            logger.info(f"Reached max expansion size limit: {max_expansion_size}")
            break
        
        # 检查实体限制
        h, r, t = candidate
        h_count = entity_expansion_count.get(h, 0)
        t_count = entity_expansion_count.get(t, 0)
        
        if h in core_entities and h_count >= max_per_entity:
            continue
        if t in core_entities and t_count >= max_per_entity:
            continue
        
        # 计算多样性奖励
        diversity_bonus = calculate_diversity_bonus(
            candidate, 
            gold_triples + selected_expansion, 
            triple_embeddings
        )
        
        # 综合得分：InfoScore + DiversityBonus
        total_score = info_score + diversity_bonus
        
        # 选择阈值判断
        if total_score > selection_threshold:
            selected_expansion.append(candidate)
            
            # 更新实体计数
            if h in core_entities:
                entity_expansion_count[h] += 1
            if t in core_entities:
                entity_expansion_count[t] += 1
            
            logger.debug(f"Selected triple: {candidate}, InfoScore: {info_score:.3f}, DiversityBonus: {diversity_bonus:.3f}, Total: {total_score:.3f}")
    
    # Step 6: 构建最终的gold_expansion_set
    gold_expansion_set = gold_triples + selected_expansion
    
    logger.info(f"Final GES: {len(gold_triples)} gold + {len(selected_expansion)} expansion = {len(gold_expansion_set)} total")
    logger.info(f"Entity expansion counts: {entity_expansion_count}")
    
    return gold_expansion_set

def analyze_expansion_quality_v2(gold_triples: List[List[str]], 
                               gold_expansion_set: List[List[str]],
                               triple_embeddings: Dict[str, np.ndarray] = None) -> Dict[str, Any]:
    """
    分析gold_expansion_set的质量指标 (v2.0增强版)
    
    Args:
        gold_triples: 原始gold三元组
        gold_expansion_set: gold_expansion_set三元组列表
        triple_embeddings: 三元组嵌入字典（可选）
        
    Returns:
        质量分析结果
    """
    # 基础统计
    original_size = len(gold_triples)
    ges_size = len(gold_expansion_set)
    expansion_size = ges_size - original_size
    
    # 关系统计
    gold_relations = [triple[1] for triple in gold_triples]
    ges_relations = [triple[1] for triple in gold_expansion_set]
    
    # 语义丰富关系比例
    semantic_in_gold = sum(1 for r in gold_relations if is_semantic_relation(r))
    semantic_in_ges = sum(1 for r in ges_relations if is_semantic_relation(r))
    
    gold_semantic_ratio = semantic_in_gold / original_size if original_size > 0 else 0
    ges_semantic_ratio = semantic_in_ges / ges_size if ges_size > 0 else 0
    
    # 多样性分析（如果有嵌入）
    avg_similarity = None
    if triple_embeddings and len(gold_expansion_set) > 1:
        similarities = []
        for i, triple1 in enumerate(gold_expansion_set):
            for j, triple2 in enumerate(gold_expansion_set):
                if i < j:
                    key1, key2 = tuple(triple1), tuple(triple2)
                    if key1 in triple_embeddings and key2 in triple_embeddings:
                        sim = cosine_similarity(
                            triple_embeddings[key1].reshape(1, -1),
                            triple_embeddings[key2].reshape(1, -1)
                        )[0][0]
                        similarities.append(sim)
        
        if similarities:
            avg_similarity = np.mean(similarities)
    
    # 计算多样性分数（1 - 平均相似度）
    diversity_score = 1.0 - avg_similarity if avg_similarity is not None else None
    
    # 将 numpy 类型转换为 Python 原生类型，避免 JSON 序列化报错
    def to_py(value):
        if isinstance(value, (np.floating, np.integer)):
            return value.item()
        return value
    
    return {
        'original_size': original_size,
        'ges_size': ges_size,
        'expansion_size': expansion_size,
        'expansion_ratio': float(ges_size / original_size) if original_size > 0 else 0.0,
        'gold_semantic_ratio': float(gold_semantic_ratio),
        'ges_semantic_ratio': float(ges_semantic_ratio),
        'semantic_improvement': float(ges_semantic_ratio - gold_semantic_ratio),
        'avg_cosine_similarity': to_py(avg_similarity),
        'diversity_score': to_py(diversity_score)
    }

def process_sample_with_ges_v2(sample: Dict[str, Any], 
                             max_expansion_size: int = DEFAULT_MAX_EXPANSION_SIZE,
                             max_per_entity: int = DEFAULT_MAX_PER_ENTITY,
                             selection_threshold: float = DEFAULT_SELECTION_THRESHOLD) -> Dict[str, Any]:
    """
    为单个样本创建GES v2.0并更新样本数据
    
    Args:
        sample: trimming结果中的样本
        max_expansion_size: GES最大大小
        max_per_entity: 每个实体最大扩展数
        selection_threshold: 选择阈值
        
    Returns:
        更新后的样本，包含GES v2.0字段
    """
    gold_triples = sample.get('gold_triples', [])
    trimmed_triples = sample.get('trimmed_triples', [])
    
    # 创建GES v2.0
    ges_v2 = create_gold_expansion_set_v2(
        gold_triples, trimmed_triples, 
        max_expansion_size, max_per_entity, selection_threshold
    )
    
    # 创建嵌入用于质量分析
    triple_embeddings = create_triple_embeddings(ges_v2)
    
    # 分析质量
    quality_metrics = analyze_expansion_quality_v2(gold_triples, ges_v2, triple_embeddings)
    
    # 更新样本
    updated_sample = sample.copy()
    # 与v1保持一致的字段命名
    updated_sample['gold_expansion_set'] = ges_v2
    updated_sample['ges_quality_metrics'] = quality_metrics
    
    return updated_sample

def batch_create_ges_v2(input_file: str, 
                      output_file: str,
                      max_expansion_size: int = DEFAULT_MAX_EXPANSION_SIZE,
                      max_per_entity: int = DEFAULT_MAX_PER_ENTITY,
                      selection_threshold: float = DEFAULT_SELECTION_THRESHOLD) -> str:
    """
    批量为trimming结果创建GES v2.0
    
    Args:
        input_file: 输入的trimming结果文件
        output_file: 输出文件路径
        max_expansion_size: GES最大大小
        max_per_entity: 每个实体最大扩展数
        selection_threshold: 选择阈值
        
    Returns:
        输出文件路径
    """
    print(f"正在为 {input_file} 创建Gold Expansion Set v2.0...")
    
    processed_count = 0
    total_expansion_ratio = 0
    total_semantic_improvement = 0
    total_diversity_score = 0
    total_avg_similarity = 0
    
    with open(input_file, 'r', encoding='utf-8') as f_in, \
         open(output_file, 'w', encoding='utf-8') as f_out:
        
        for line_num, line in enumerate(f_in, 1):
            if not line.strip():
                continue
                
            try:
                data = json.loads(line)
                
                # 跳过配置和统计行
                if 'config' in data or 'final_stats' in data or 'batch_stats' in data:
                    f_out.write(line)
                    continue
                
                # 处理样本
                if 'sample_id' in data:
                    updated_sample = process_sample_with_ges_v2(
                        data, max_expansion_size, max_per_entity, selection_threshold
                    )
                    
                    # 累计统计
                    quality_metrics = updated_sample['ges_quality_metrics']
                    total_expansion_ratio += quality_metrics['expansion_ratio']
                    total_semantic_improvement += quality_metrics['semantic_improvement']
                    
                    if quality_metrics['diversity_score'] is not None:
                        total_diversity_score += quality_metrics['diversity_score']
                    if quality_metrics['avg_cosine_similarity'] is not None:
                        total_avg_similarity += quality_metrics['avg_cosine_similarity']
                    
                    processed_count += 1
                    
                    f_out.write(json.dumps(updated_sample, ensure_ascii=False) + '\n')
                    
                    if processed_count % 100 == 0:
                        print(f"已处理 {processed_count} 个样本...")
                        
                else:
                    f_out.write(line)
                    
            except json.JSONDecodeError as e:
                print(f"警告：第{line_num}行JSON解析失败: {e}")
                f_out.write(line)
                continue
            except Exception as e:
                print(f"警告：第{line_num}行处理失败: {e}")
                f_out.write(line)
                continue
    
    # 打印统计信息
    if processed_count > 0:
        avg_expansion_ratio = total_expansion_ratio / processed_count
        avg_semantic_improvement = total_semantic_improvement / processed_count
        avg_diversity_score = total_diversity_score / processed_count
        avg_avg_similarity = total_avg_similarity / processed_count
        
        print(f"\n=== GES v2.0 创建完成 ===")
        print(f"处理样本数: {processed_count}")
        print(f"平均扩展比例: {avg_expansion_ratio:.2f}x")
        print(f"平均语义关系改善: {avg_semantic_improvement:.3f}")
        print(f"平均多样性分数: {avg_diversity_score:.3f}")
        print(f"平均余弦相似度: {avg_avg_similarity:.3f}")
        print(f"结果保存至: {output_file}")
    
    return output_file

def main():
    """主函数，支持命令行参数"""
    import argparse
    import os
    from datetime import datetime
    
    parser = argparse.ArgumentParser(
        description='为JSONL文件添加Gold Expansion Set v2.0 (改进版)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python gold_expansion_utils_v2.py input.jsonl
  python gold_expansion_utils_v2.py input.jsonl --output custom_output.jsonl
  python gold_expansion_utils_v2.py input.jsonl --max-size 6 --max-per-entity 1 --threshold 0.6
        """
    )
    
    parser.add_argument('input_file', nargs='?',
                       help='输入的trimming结果JSONL文件路径 (可省略，若省略则根据 --dataset 自动推断)')
    parser.add_argument('--dataset', default='metaqa-2hop',
                       help='数据集名称 (默认: metaqa-2hop)，当未显式提供 input_file 时使用该数据集的默认 trimming 结果路径')
    parser.add_argument('--output', '-o', 
                       help='输出文件路径 (默认: 原文件名_with_ges_v2.jsonl)')
    parser.add_argument('--max-size', type=int, default=DEFAULT_MAX_EXPANSION_SIZE,
                       help=f'GES最大大小 (默认: {DEFAULT_MAX_EXPANSION_SIZE})')
    parser.add_argument('--max-per-entity', type=int, default=DEFAULT_MAX_PER_ENTITY,
                       help=f'每个实体最大扩展数 (默认: {DEFAULT_MAX_PER_ENTITY})')
    parser.add_argument('--threshold', type=float, default=DEFAULT_SELECTION_THRESHOLD,
                       help=f'选择阈值 (默认: {DEFAULT_SELECTION_THRESHOLD})')
    parser.add_argument('--force', action='store_true',
                       help='强制覆盖已存在的输出文件')
    
    args = parser.parse_args()
    
    # 如果未提供 input_file，则根据数据集名称自动生成
    if args.input_file:
        input_file = args.input_file
    else:
        input_file = f"experiment_records/trimming_results/{args.dataset}/dev_simple_trimming_results.jsonl"

    if not os.path.exists(input_file):
        print(f"❌ 错误: 输入文件不存在: {input_file}")
        return 1
    
    # 生成输出文件名
    if args.output:
        output_file = args.output
    else:
        input_base = os.path.splitext(input_file)[0]
        output_file = f"{input_base}_with_ges.jsonl"
    
    # 检查输出文件是否已存在
    if os.path.exists(output_file) and not args.force:
        print(f"❌ 错误: 输出文件已存在: {output_file}")
        print("使用 --force 参数强制覆盖，或使用 --output 指定不同的输出文件名")
        return 1
    
    # 打印配置信息
    print("="*70)
    print("�� Gold Expansion Set v2.0 生成工具 (改进版)")
    print("="*70)
    print(f"📁 输入文件: {input_file}")
    print(f"📁 输出文件: {output_file}")
    print(f"⚙️  最大GES大小: {args.max_size}")
    print(f"⚙️  每实体最大扩展数: {args.max_per_entity}")
    print(f"⚙️  选择阈值: {args.threshold}")
    print(f"🕐 开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("-"*70)
    
    try:
        # 执行批量处理
        start_time = datetime.now()
        result_file = batch_create_ges_v2(
            input_file, 
            output_file,
            args.max_size,
            args.max_per_entity,
            args.threshold
        )
        end_time = datetime.now()
        
        # 计算处理时间
        processing_time = (end_time - start_time).total_seconds()
        
        print("-"*70)
        print(f"✅ 处理完成!")
        print(f"🕐 总处理时间: {processing_time:.2f} 秒")
        print(f"📁 结果文件: {result_file}")
        print("="*70)
        
        # 获取数据集名称和输出文件路径
        dataset_name = args.dataset
        print(f"📊 数据集: {dataset_name} | 生成文件: {output_file}")
        
        return 0
        
    except Exception as e:
        print(f"❌ 处理过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    import sys
    sys.exit(main()) 