"""
为GraphDeEP推理结果添加基于SQuAD标准的改进评估

功能：
1. 读取原始inference结果文件
2. 使用SQuAD标准重新评估每个样本
3. 添加新的评估字段
4. 保存改进后的结果文件
"""

import json
import re
import string
from collections import Counter
from typing import List, Dict
import os
from datetime import datetime

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def f1_score(prediction, ground_truth):
    """Calculate F1 score between prediction and ground truth."""
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def exact_match_score(prediction, ground_truth):
    """Calculate exact match score between prediction and ground truth."""
    return (normalize_answer(prediction) == normalize_answer(ground_truth))

def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    """Calculate max score over multiple ground truths."""
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)

def extract_answer_entity(raw_answer: str) -> str:
    """
    从模型输出中提取核心答案实体
    针对GraphDeEP项目的答案格式优化
    """
    # 移除"ans:"前缀
    if raw_answer.startswith('ans:'):
        answer = raw_answer[4:].strip()
    else:
        answer = raw_answer.strip()
    
    # 移除常见的描述性模式 - 保留核心实体
    patterns_to_remove = [
        # 移除主语+动词的描述，保留宾语实体
        r'^[^,]*\b(wrote|directed|starred?.*in|acts?.*in|appears?.*in|produced|created|stars?.*in)\s*:?\s*',
        r'^[^,]*\b(is|are|was|were)\s+',
        r'^the\s+(movie|film|book|song|answer)\s+(is|was)\s+',
    ]
    
    for pattern in patterns_to_remove:
        answer = re.sub(pattern, '', answer, flags=re.IGNORECASE)
    
    # 处理列表格式的答案 (如: "* movie1 * movie2 * movie3")
    if '* ' in answer:
        # 提取所有"* "后的内容
        items = re.findall(r'\*\s*([^*\n]+)', answer)
        if items:
            # 清理每个项目并连接
            cleaned_items = [item.strip() for item in items]
            answer = ', '.join(cleaned_items)
    
    # 移除引号
    answer = re.sub(r'["\']([^"\']*)["\']', r'\1', answer)
    
    # 移除句子结尾的无关信息
    patterns_to_remove_end = [
        r'\s*\(.*\)$',      # 移除括号内容
        r'\s*[.!?].*$',     # 移除句号后的内容  
    ]
    
    for pattern in patterns_to_remove_end:
        answer = re.sub(pattern, '', answer, flags=re.IGNORECASE)
    
    return answer.strip()

def squad_evaluate_sample(raw_answer: str, golden_answers: List[str]) -> Dict[str, float]:
    """
    使用SQuAD标准评估单个样本
    """
    # 提取核心答案实体
    extracted_answer = extract_answer_entity(raw_answer)
    
    # 使用SQuAD官方算法计算指标
    exact_match = metric_max_over_ground_truths(exact_match_score, extracted_answer, golden_answers)
    f1 = metric_max_over_ground_truths(f1_score, extracted_answer, golden_answers)
    
    return {
        'exact_match': exact_match,
        'f1': f1,
        'extracted_answer': extracted_answer,
        'normalized_extracted': normalize_answer(extracted_answer),
        'normalized_golden': [normalize_answer(ga) for ga in golden_answers]
    }

def squad_based_hallucination_judge(raw_answer: str, golden_answers: List[str], 
                                   f1_threshold: float = 0.3) -> Dict[str, any]:
    """
    基于SQuAD标准的改进幻觉判断器
    """
    eval_result = squad_evaluate_sample(raw_answer, golden_answers)
    
    # 基于SQuAD指标的判断
    is_exact_match = eval_result['exact_match'] == 1
    is_good_f1 = eval_result['f1'] >= f1_threshold
    
    # 综合判断：EM=1或F1>=阈值都不算幻觉
    is_hallucination = not (is_exact_match or is_good_f1)
    
    # 生成详细的判断信息
    if is_exact_match:
        confidence = 'high'
        reason = "Exact match after normalization"
    elif is_good_f1:
        confidence = 'medium'
        reason = f"Good word overlap (F1={eval_result['f1']:.3f})"
    else:
        confidence = 'low'
        reason = f"Low word overlap (F1={eval_result['f1']:.3f})"
    
    return {
        'squad_is_hallucination': is_hallucination,
        'squad_confidence': confidence,
        'squad_reason': reason,
        'squad_exact_match': eval_result['exact_match'],
        'squad_f1_score': eval_result['f1'],
        'squad_extracted_answer': eval_result['extracted_answer'],
        'squad_normalized_extracted': eval_result['normalized_extracted'],
        'squad_normalized_golden': eval_result['normalized_golden']
    }

def process_inference_file(input_file: str, output_file: str = None, f1_threshold: float = 0.3):
    """
    处理推理结果文件，添加SQuAD评估
    """
    if output_file is None:
        base_name = os.path.splitext(input_file)[0]
        output_file = f"{base_name}_with_squad_eval.jsonl"
    
    print(f"处理文件: {input_file}")
    print(f"输出文件: {output_file}")
    print(f"F1阈值: {f1_threshold}")
    
    # 统计信息
    total_samples = 0
    original_hallucinations = 0
    squad_hallucinations = 0
    corrections = 0  # 原来判断为幻觉，现在判断为非幻觉
    
    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8') as outfile:
        
        for line_num, line in enumerate(infile, 1):
            if not line.strip():
                continue
                
            try:
                data = json.loads(line.strip())
                
                # 跳过配置行
                if 'config' in data:
                    # 更新配置信息
                    data['config']['squad_evaluation_added'] = datetime.now().strftime("%Y%m%d_%H%M%S")
                    data['config']['squad_f1_threshold'] = f1_threshold
                    outfile.write(json.dumps(data, ensure_ascii=False) + '\n')
                    continue
                
                # 处理数据行
                total_samples += 1
                
                # 获取原始判断
                original_is_hallucination = not data.get('metrics', {}).get('hit@1', False)
                if original_is_hallucination:
                    original_hallucinations += 1
                
                # 进行SQuAD评估
                raw_answer = data.get('model_output', '')
                golden_answers = data.get('golden_answers', [])
                
                squad_result = squad_based_hallucination_judge(raw_answer, golden_answers, f1_threshold)
                
                # 添加SQuAD评估结果
                data['squad_evaluation'] = squad_result
                
                # 统计SQuAD判断结果
                if squad_result['squad_is_hallucination']:
                    squad_hallucinations += 1
                
                # 统计修正情况
                if original_is_hallucination and not squad_result['squad_is_hallucination']:
                    corrections += 1
                    print(f"样本 {total_samples} 被修正: {squad_result['squad_extracted_answer']} (F1={squad_result['squad_f1_score']:.3f})")
                
                # 写入结果
                outfile.write(json.dumps(data, ensure_ascii=False) + '\n')
                
                # 每1000个样本输出进度
                if total_samples % 1000 == 0:
                    print(f"已处理 {total_samples} 个样本...")
                    
            except json.JSONDecodeError as e:
                print(f"第{line_num}行JSON解析错误: {e}")
                continue
            except Exception as e:
                print(f"第{line_num}行处理错误: {e}")
                continue
    
    # 输出统计结果
    print(f"\n=== 处理完成 ===")
    print(f"总样本数: {total_samples}")
    print(f"原始幻觉判断: {original_hallucinations} ({original_hallucinations/total_samples*100:.1f}%)")
    print(f"SQuAD幻觉判断: {squad_hallucinations} ({squad_hallucinations/total_samples*100:.1f}%)")
    print(f"修正样本数: {corrections} ({corrections/total_samples*100:.1f}%)")
    print(f"结果已保存到: {output_file}")
    
    return {
        'total_samples': total_samples,
        'original_hallucinations': original_hallucinations,
        'squad_hallucinations': squad_hallucinations,
        'corrections': corrections,
        'correction_rate': corrections / total_samples if total_samples > 0 else 0
    }

def show_correction_examples(input_file: str, num_examples: int = 5):
    """
    显示修正案例示例
    """
    print(f"\n=== 修正案例示例 ===")
    examples_shown = 0
    
    with open(input_file, 'r', encoding='utf-8') as file:
        for line in file:
            if not line.strip():
                continue
                
            try:
                data = json.loads(line.strip())
                
                if 'config' in data:
                    continue
                
                # 检查是否被修正
                original_is_hallucination = not data.get('metrics', {}).get('hit@1', False)
                squad_is_hallucination = data.get('squad_evaluation', {}).get('squad_is_hallucination', True)
                
                if original_is_hallucination and not squad_is_hallucination:
                    examples_shown += 1
                    squad_eval = data['squad_evaluation']
                    
                    print(f"\n--- 修正案例 {examples_shown} ---")
                    print(f"问题: {data.get('question', 'N/A')}")
                    print(f"原始输出: {data.get('model_output', 'N/A')}")
                    print(f"提取答案: {squad_eval.get('squad_extracted_answer', 'N/A')}")
                    print(f"金标准: {data.get('golden_answers', [])}")
                    print(f"SQuAD指标: EM={squad_eval.get('squad_exact_match')}, F1={squad_eval.get('squad_f1_score', 0):.3f}")
                    print(f"判断: {squad_eval.get('squad_reason', 'N/A')}")
                    
                    if examples_shown >= num_examples:
                        break
                        
            except Exception as e:
                continue

if __name__ == "__main__":
    # 处理你的文件 - 修改为Qwen2.5结果文件
    input_file = "experiment_records/inference_results/llama2-7b/colab_semantic_drift.jsonl"
    
    print("开始处理inference结果文件...")
    stats = process_inference_file(input_file, f1_threshold=0.3)
    
    # 显示修正案例
    output_file = input_file.replace('.jsonl', '_with_squad_eval.jsonl')
    show_correction_examples(output_file, num_examples=5) 