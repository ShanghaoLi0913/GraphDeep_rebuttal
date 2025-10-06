#!/usr/bin/env python3
"""
Gold Expansion Set 工具模块

为FGAS计算构建语义丰富的三元组扩展集合，解决原始gold triples
语义信号稀疏的问题。

核心思想：
- TUS使用原始gold triples（注意力聚焦）
- FGAS使用gold_expansion_set（语义表达对齐）

作者: GraphDeEP Team
创建日期: 2024-12-27
"""

from typing import List, Dict, Set, Tuple, Any
import json

# 定义语义强的关系类型
SEMANTIC_RICH_RELATIONS = {
    'directed_by', 'acted_in', 'genre', 'release_year', 'language',
    'produced_by', 'written_by', 'cinematography', 'music_by',
    'starring', 'has_genre', 'release_date', 'runtime', 'budget',
    'award', 'nomination', 'country', 'company', 'sequel_to',
    'prequel_to', 'based_on', 'remake_of', 'character', 'role'
}

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

def find_neighbor_triples(core_entities: Set[str], 
                         all_triples: List[List[str]], 
                         gold_triples: List[List[str]],
                         max_per_entity: int = 3) -> List[List[str]]:
    """
    查找与核心实体相关的邻居三元组
    
    Args:
        core_entities: 核心实体集合
        all_triples: 完整的三元组列表（来自trimmed subgraph）
        gold_triples: 原始gold三元组（需要排除）
        max_per_entity: 每个实体最多扩展的三元组数
        
    Returns:
        邻居三元组列表
    """
    # 将gold triples转换为集合以便快速查找
    gold_set = set(tuple(triple) for triple in gold_triples)
    
    # 为每个核心实体收集相关三元组
    entity_triples = {entity: [] for entity in core_entities}
    
    for triple in all_triples:
        h, r, t = triple
        triple_tuple = tuple(triple)
        
        # 跳过已在gold triples中的
        if triple_tuple in gold_set:
            continue
            
        # 只保留语义丰富的关系
        if not is_semantic_relation(r):
            continue
            
        # 如果头实体或尾实体在核心集合中，添加到对应实体的列表
        if h in core_entities:
            entity_triples[h].append(triple)
        if t in core_entities:
            entity_triples[t].append(triple)
    
    # 为每个实体限制扩展数量
    neighbor_triples = []
    for entity, triples in entity_triples.items():
        # 可以添加更复杂的排序逻辑，比如根据关系重要性
        selected = triples[:max_per_entity]
        neighbor_triples.extend(selected)
    
    # 去重
    unique_neighbors = []
    seen = set()
    for triple in neighbor_triples:
        triple_tuple = tuple(triple)
        if triple_tuple not in seen:
            seen.add(triple_tuple)
            unique_neighbors.append(triple)
    
    return unique_neighbors

def create_gold_expansion_set(gold_triples: List[List[str]], 
                            trimmed_triples: List[List[str]],
                            max_expansion_size: int = 20,
                            max_per_entity: int = 3) -> List[List[str]]:
    """
    创建gold_expansion_set
    
    Args:
        gold_triples: 原始gold三元组
        trimmed_triples: 完整的trimmed子图三元组
        max_expansion_size: gold_expansion_set的最大大小
        max_per_entity: 每个核心实体最多扩展的三元组数
        
    Returns:
        gold_expansion_set三元组列表
    """
    # Step 1: 提取核心实体
    core_entities = extract_core_entities(gold_triples)
    
    # Step 2: 查找邻居三元组
    neighbor_triples = find_neighbor_triples(
        core_entities, trimmed_triples, gold_triples, max_per_entity
    )
    
    # Step 3: 构建gold_expansion_set（包含原始gold triples + 扩展三元组）
    gold_expansion_set = gold_triples.copy()
    gold_expansion_set.extend(neighbor_triples)
    
    # Step 4: 限制总大小
    if len(gold_expansion_set) > max_expansion_size:
        # 保留所有gold triples，然后截取扩展部分
        remaining_slots = max_expansion_size - len(gold_triples)
        if remaining_slots > 0:
            gold_expansion_set = gold_triples + neighbor_triples[:remaining_slots]
        else:
            gold_expansion_set = gold_triples[:max_expansion_size]
    
    return gold_expansion_set

def analyze_expansion_quality(gold_triples: List[List[str]], 
                            gold_expansion_set: List[List[str]]) -> Dict[str, Any]:
    """
    分析gold_expansion_set的质量指标（精简版）
    
    Args:
        gold_triples: 原始gold三元组
        gold_expansion_set: gold_expansion_set三元组列表
        
    Returns:
        质量分析结果（4个核心指标）
    """
    # 统计关系类型
    gold_relations = [triple[1] for triple in gold_triples]
    ges_relations = [triple[1] for triple in gold_expansion_set]
    
    # 统计语义丰富关系的比例
    semantic_in_gold = sum(1 for r in gold_relations if is_semantic_relation(r))
    semantic_in_ges = sum(1 for r in ges_relations if is_semantic_relation(r))
    
    # 返回精简版指标
    return {
        'original_size': len(gold_triples),
        'ges_size': len(gold_expansion_set),
        'expansion_ratio': len(gold_expansion_set) / len(gold_triples) if gold_triples else 0,
        'semantic_improvement': (semantic_in_ges / len(gold_expansion_set)) - (semantic_in_gold / len(gold_triples)) if gold_triples and gold_expansion_set else 0
    }

# 别名函数
def analyze_ges_quality(gold_triples: List[List[str]], 
                       gold_expansion_set: List[List[str]]) -> Dict[str, Any]:
    """
    analyze_expansion_quality 的别名，保持向后兼容
    """
    return analyze_expansion_quality(gold_triples, gold_expansion_set)

def process_sample_with_ges(sample: Dict[str, Any], 
                           max_expansion_size: int = 20,
                           max_per_entity: int = 3) -> Dict[str, Any]:
    """
    为单个样本创建GES并更新样本数据
    
    Args:
        sample: trimming结果中的样本
        max_expansion_size: GES最大大小
        max_per_entity: 每个实体最大扩展数
        
    Returns:
        更新后的样本，包含GES字段
    """
    gold_triples = sample.get('gold_triples', [])
    trimmed_triples = sample.get('trimmed_triples', [])
    
    # 创建GES
    ges = create_gold_expansion_set(
        gold_triples, trimmed_triples, 
        max_expansion_size, max_per_entity
    )
    
    # 分析质量
    quality_metrics = analyze_ges_quality(gold_triples, ges)
    
    # 更新样本
    updated_sample = sample.copy()
    updated_sample['gold_expansion_set'] = ges
    updated_sample['ges_quality_metrics'] = quality_metrics
    
    return updated_sample

def batch_create_ges(input_file: str, 
                    output_file: str,
                    max_expansion_size: int = 20,
                    max_per_entity: int = 3) -> str:
    """
    批量为trimming结果创建GES
    
    Args:
        input_file: 输入的trimming结果文件
        output_file: 输出文件路径
        max_expansion_size: GES最大大小
        max_per_entity: 每个实体最大扩展数
        
    Returns:
        输出文件路径
    """
    print(f"正在为 {input_file} 创建Gold Expansion Set...")
    
    processed_count = 0
    total_expansion_ratio = 0
    total_semantic_improvement = 0
    
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
                    updated_sample = process_sample_with_ges(
                        data, max_expansion_size, max_per_entity
                    )
                    
                    # 累计统计
                    quality_metrics = updated_sample['ges_quality_metrics']
                    total_expansion_ratio += quality_metrics['expansion_ratio']
                    total_semantic_improvement += quality_metrics['semantic_improvement']
                    processed_count += 1
                    
                    f_out.write(json.dumps(updated_sample, ensure_ascii=False) + '\n')
                else:
                    f_out.write(line)
                    
            except json.JSONDecodeError as e:
                print(f"警告：第{line_num}行JSON解析失败: {e}")
                f_out.write(line)
                continue
    
    # 打印统计信息
    if processed_count > 0:
        avg_expansion_ratio = total_expansion_ratio / processed_count
        avg_semantic_improvement = total_semantic_improvement / processed_count
        
        print(f"\n=== GES创建完成 ===")
        print(f"处理样本数: {processed_count}")
        print(f"平均扩展比例: {avg_expansion_ratio:.2f}x")
        print(f"平均语义关系改善: {avg_semantic_improvement:.3f}")
        print(f"结果保存至: {output_file}")
    
    return output_file

def main():
    """主函数，支持命令行参数"""
    import argparse
    import os
    from datetime import datetime
    
    parser = argparse.ArgumentParser(
        description='为JSONL文件添加Gold Expansion Set',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python gold_expansion_utils.py input.jsonl
  python gold_expansion_utils.py input.jsonl --output custom_output.jsonl
  python gold_expansion_utils.py input.jsonl --max-size 15 --max-per-entity 2
        """
    )
    
    parser.add_argument('input_file', 
                       help='输入的trimming结果JSONL文件路径')
    parser.add_argument('--output', '-o', 
                       help='输出文件路径 (默认: 原文件名_with_ges.jsonl)')
    parser.add_argument('--max-size', type=int, default=20,
                       help='GES最大大小 (默认: 20)')
    parser.add_argument('--max-per-entity', type=int, default=3,
                       help='每个实体最大扩展数 (默认: 3)')
    parser.add_argument('--force', action='store_true',
                       help='强制覆盖已存在的输出文件')
    
    args = parser.parse_args()
    
    # 检查输入文件是否存在
    if not os.path.exists(args.input_file):
        print(f"❌ 错误: 输入文件不存在: {args.input_file}")
        return 1
    
    # 生成输出文件名
    if args.output:
        output_file = args.output
    else:
        # 自动生成输出文件名
        input_base = os.path.splitext(args.input_file)[0]
        output_file = f"{input_base}_with_ges.jsonl"
    
    # 检查输出文件是否已存在
    if os.path.exists(output_file) and not args.force:
        print(f"❌ 错误: 输出文件已存在: {output_file}")
        print("使用 --force 参数强制覆盖，或使用 --output 指定不同的输出文件名")
        return 1
    
    # 打印配置信息
    print("="*60)
    print("🚀 Gold Expansion Set (GES) 生成工具")
    print("="*60)
    print(f"📁 输入文件: {args.input_file}")
    print(f"📁 输出文件: {output_file}")
    print(f"⚙️  最大GES大小: {args.max_size}")
    print(f"⚙️  每实体最大扩展数: {args.max_per_entity}")
    print(f"🕐 开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("-"*60)
    
    try:
        # 执行批量处理
        start_time = datetime.now()
        result_file = batch_create_ges(
            args.input_file, 
            output_file,
            args.max_size,
            args.max_per_entity
        )
        end_time = datetime.now()
        
        # 计算处理时间
        processing_time = (end_time - start_time).total_seconds()
        
        print("-"*60)
        print(f"✅ 处理完成!")
        print(f"🕐 总处理时间: {processing_time:.2f} 秒")
        print(f"📁 结果文件: {result_file}")
        print("="*60)
        
        return 0
        
    except Exception as e:
        print(f"❌ 处理过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    import sys
    sys.exit(main()) 