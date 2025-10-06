"""
使用Qwen重新评估推理结果的正确性

这个脚本将：
1. 读取现有的推理结果文件
2. 使用Qwen模型重新判断每个预测是否正确
3. 生成新的truthful/hallucinated标签
4. 重新计算指标的显著性
"""

import json
import time
from typing import Dict, List, Tuple
import argparse
from tqdm import tqdm
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def load_model(model_name="Qwen/Qwen-7B-Chat"):
    """加载Qwen模型和tokenizer"""
    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16
    )
    return model, tokenizer

def create_evaluation_prompt(question: str, predicted_answer: str, golden_answers: List[str]) -> str:
    """创建用于评估的提示词"""
    
    golden_answers_str = ", ".join([f'"{ans}"' for ans in golden_answers])
    
    prompt = f"""请评估预测的答案是否正确回答了给定的问题。

问题：{question}
标准答案：{golden_answers_str}
预测答案：{predicted_answer}

评估标准：
1. 如果预测答案包含任何一个标准答案，就认为是正确的，即使：
   - 格式略有不同（例如，"电影" vs "这部电影"）
   - 包含额外的解释文字
   - 顺序不同
   - 有轻微的拼写变化

2. 在以下情况下预测答案是错误的：
   - 完全没有包含任何标准答案
   - 只包含无关信息
   - 给出了错误的事实信息

3. 对格式差异要宽容，但对事实准确性要严格。

请只回答一个词："CORRECT" 或 "INCORRECT"
"""
    return prompt

def evaluate_with_qwen(question: str, predicted_answer: str, golden_answers: List[str], 
                      model, tokenizer) -> str:
    """使用Qwen评估答案是truthful还是hallucinated"""
    
    prompt = create_evaluation_prompt(question, predicted_answer, golden_answers)
    
    try:
        response, history = model.chat(tokenizer, prompt, history=None)
        result = response.strip().upper()
        
        if result == "CORRECT":
            return "truthful"
        elif result == "INCORRECT":
            return "hallucinated"
        else:
            return "unknown"
        
    except Exception as e:
        print(f"Error using Qwen model: {e}")
        return "unknown"

def reevaluate_results_file(input_file: str, output_file: str, model_name: str = "Qwen/Qwen-7B-Chat"):
    """重新评估整个结果文件"""
    
    print(f"Reading {input_file}...")
    
    # 加载模型
    model, tokenizer = load_model(model_name)
    
    results = []
    config = None
    
    # 读取原始结果
    with open(input_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f):
            data = json.loads(line)
            
            if line_num == 0 and 'config' in data:
                config = data
                continue
                
            results.append(data)
    
    print(f"Found {len(results)} samples to evaluate")
    
    # 重新评估每个样本
    reevaluated_results = []
    truthful_count = 0
    hallucinated_count = 0
    error_count = 0
    
    for i, result in enumerate(tqdm(results, desc="Re-evaluating with Qwen")):
        try:
            question = result['question']
            predicted_answer = result.get('answer', result.get('extracted_answers', [''])[0])
            golden_answers = result['golden_answers']
            
            # 获取原始判断（仅用于记录）
            original_hit1 = result.get('metrics', {}).get('hit@1', False)
            
            # 使用Qwen评估
            qwen_judgment = evaluate_with_qwen(question, predicted_answer, golden_answers, model, tokenizer)
            
            if qwen_judgment == "unknown":
                error_count += 1
                qwen_judgment = "hallucinated"  # 默认为hallucinated
            
            # 更新结果
            updated_result = result.copy()
            updated_result['qwen_judgment'] = qwen_judgment
            updated_result['metrics']['hit@1'] = (qwen_judgment == "truthful")
            updated_result['original_hit@1'] = original_hit1
            updated_result['reevaluated'] = True
            
            reevaluated_results.append(updated_result)
            
            if qwen_judgment == "truthful":
                truthful_count += 1
            elif qwen_judgment == "hallucinated":
                hallucinated_count += 1
                
        except Exception as e:
            print(f"Error processing sample {i}: {e}")
            updated_result = result.copy()
            updated_result['qwen_judgment'] = "unknown"
            updated_result['reevaluated'] = False
            reevaluated_results.append(updated_result)
            error_count += 1
    
    # 保存结果
    with open(output_file, 'w', encoding='utf-8') as f:
        if config:
            updated_config = config.copy()
            updated_config['reevaluated'] = True
            updated_config['model'] = model_name
            updated_config['original_file'] = input_file
            f.write(json.dumps(updated_config, ensure_ascii=False) + '\n')
        
        for result in reevaluated_results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
    
    print(f"\nQwen Re-evaluation completed!")
    print(f"Total samples: {len(results)}")
    print(f"Qwen Judgments:")
    print(f"  - Truthful: {truthful_count}/{len(results)} ({truthful_count/len(results)*100:.1f}%)")
    print(f"  - Hallucinated: {hallucinated_count}/{len(results)} ({hallucinated_count/len(results)*100:.1f}%)")
    print(f"Errors: {error_count}")
    print(f"Results saved to: {output_file}")
    
    return output_file

def compare_evaluations(input_file: str):
    """比较原始评估和Qwen评估的差异"""
    
    original_correct = 0
    qwen_truthful = 0
    qwen_hallucinated = 0
    changed_count = 0
    total_count = 0
    
    changes = []
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f):
            data = json.loads(line)
            
            if line_num == 0 and 'config' in data:
                continue
                
            if 'reevaluated' in data:
                original_hit1 = data.get('original_hit@1', False)
                qwen_judgment = data.get('qwen_judgment', 'unknown')
                new_hit1 = (qwen_judgment == "truthful")
                
                original_correct += original_hit1
                if qwen_judgment == "truthful":
                    qwen_truthful += 1
                elif qwen_judgment == "hallucinated":
                    qwen_hallucinated += 1
                
                total_count += 1
                
                if original_hit1 != new_hit1:
                    changed_count += 1
                    changes.append({
                        'question': data['question'],
                        'answer': data.get('answer', ''),
                        'golden_answers': data['golden_answers'],
                        'original': "correct" if original_hit1 else "incorrect",
                        'qwen_judgment': qwen_judgment
                    })
    
    print(f"\n=== Original vs Qwen Evaluation Comparison ===")
    print(f"Total samples: {total_count}")
    print(f"Original accuracy: {original_correct}/{total_count} ({original_correct/total_count*100:.1f}%)")
    print(f"Qwen Judgments:")
    print(f"  - Truthful: {qwen_truthful}/{total_count} ({qwen_truthful/total_count*100:.1f}%)")
    print(f"  - Hallucinated: {qwen_hallucinated}/{total_count} ({qwen_hallucinated/total_count*100:.1f}%)")
    print(f"Changed evaluations: {changed_count} ({changed_count/total_count*100:.1f}%)")
    
    if changes:
        print(f"\nFirst 5 changes:")
        for i, change in enumerate(changes[:5]):
            print(f"{i+1}. Original: {change['original']} → Qwen: {change['qwen_judgment']}")
            print(f"   Q: {change['question']}")
            print(f"   A: {change['answer']}")
            print(f"   Golden: {change['golden_answers']}")
            print()

def main():
    parser = argparse.ArgumentParser(description='Re-evaluate inference results using Qwen')
    parser.add_argument('--input', '-i', nargs='+', required=True, 
                       help='Input inference results file(s). Can specify multiple files.')
    parser.add_argument('--model', '-m', default='Qwen/Qwen-7B-Chat',
                       help='Qwen model name (default: Qwen/Qwen-7B-Chat)')
    parser.add_argument('--compare', '-c', action='store_true', 
                       help='Compare evaluations after re-evaluation')
    parser.add_argument('--output_dir', '-o', default='experiment_records',
                       help='Output directory (default: experiment_records)')
    
    args = parser.parse_args()
    
    # 确保输出目录存在
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 处理每个输入文件
    for input_file in args.input:
        print(f"\n{'='*60}")
        print(f"Processing: {input_file}")
        print(f"{'='*60}")
        
        # 生成输出文件名
        base_name = os.path.basename(input_file)
        name_without_ext = os.path.splitext(base_name)[0]
        output_file = os.path.join(args.output_dir, f"{name_without_ext}_reevaluated.jsonl")
        
        # 重新评估
        output_file = reevaluate_results_file(input_file, output_file, args.model)
        
        # 比较结果
        if args.compare:
            compare_evaluations(output_file)

if __name__ == "__main__":
    main() 