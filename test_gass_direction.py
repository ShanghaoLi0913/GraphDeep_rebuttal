#!/usr/bin/env python3
"""
测试Llama3不同层的GASS方向性
关键：找到哪个层能够让 truthful_gass > hallucinated_gass
"""

import torch
import json
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from gass_metrics import calculate_gass
from dataset_processor import load_entities_and_relations
import numpy as np
from tqdm import tqdm
import warnings
import random
from datetime import datetime
import argparse

warnings.filterwarnings("ignore")

def setup_model():
    """加载Llama3-8B模型"""
    print("🤖 Loading Llama3-8B model...")
    
    model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        use_cache=False,
        token=True
    )
    
    return model, tokenizer

def load_test_data(num_samples=50):
    """加载测试数据样本"""
    print(f"📁 Loading {num_samples} test samples...")
    
    test_file = 'experiment_records/trimming_results/metaqa-1hop/test_simple_trimming_results_with_ges.jsonl'
    
    samples = []
    with open(test_file, 'r', encoding='utf-8') as f:
        # 跳过配置行
        config_line = f.readline()
        
        for i, line in enumerate(f):
            if i >= num_samples:
                break
            try:
                sample = json.loads(line.strip())
                samples.append(sample)
            except:
                continue
    
    print(f"✅ Loaded {len(samples)} samples")
    return samples

def generate_answer(question, trimmed_triples, model, tokenizer):
    """正常生成答案，使用与main_colab.py相同的prompt格式"""
    # 构建Knowledge部分
    triples_text = "Knowledge:\n"
    for h, r, t in trimmed_triples:
        triples_text += f"{h} {r} {t}\n"
    
    # 构建用户输入（与main_colab.py一致）
    user_content = f"{triples_text}\nQuestion: {question}\nAnswer:"
    
    # 系统提示
    system_prompt = """Based on the triples retrieved from a knowledge graph, please answer the question. Please return ONLY the answer entities as a list, each prefixed with "ans:". Do not include explanations or reasoning."""
    
    # 构建完整的提示（与main_colab.py一致）
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content}
    ]
    
    # 应用聊天模板
    formatted_input = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )
    
    inputs = tokenizer(formatted_input, return_tensors="pt", add_special_tokens=False)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    # 生成答案（使用与main_colab.py相同的参数）
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,  # 与main_colab.py一致
            do_sample=False,    # 贪婪解码，确保结果一致
            pad_token_id=tokenizer.eos_token_id
        )
    
    # 提取生成的答案部分
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # 提取生成的答案部分（移除输入部分）
    input_length = len(tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=True))
    answer_part = generated_text[input_length:].strip()
    
    # 清理答案格式
    if answer_part.startswith("assistant"):
        answer_part = answer_part[len("assistant"):].strip()
    
    # 移除可能的换行符
    answer_part = answer_part.replace('\n', ' ').strip()
    
    return answer_part if len(answer_part) > 0 else None

def generate_once_for_all_layers(question, trimmed_triples, model, tokenizer):
    """进行一次generation，返回所有layer计算需要的信息"""
    
    # 系统提示（与main_colab.py一致）
    system_prompt = """Based on the triples retrieved from a knowledge graph, please answer the question. Please return ONLY the answer entities as a list, each prefixed with "ans:". Do not include explanations or reasoning."""
    
    # 构建Knowledge部分（与main_colab.py一致）
    triples_text = "Knowledge:\n"
    for h, r, t in trimmed_triples:
        triples_text += f"{h} {r} {t}\n"
    
    # 构建用户输入（与main_colab.py一致）
    user_content = f"{triples_text}\nQuestion: {question}\nAnswer:"
    
    # 构建完整的提示（与main_colab.py一致）
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content}
    ]
    
    # 准备输入（与main_colab.py一致）
    input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    # 实际生成（与main_colab.py完全一致）
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
        try:
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                num_return_sequences=1,
                do_sample=False,  # 使用贪婪解码
                temperature=1.0,
                use_cache=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                output_attentions=False  # 生成时不需要attention
            )
        except Exception as gen_error:
            print(f"⚠️ Generation failed: {gen_error}")
            return None, None, None, None
    
    # 找到生成文本的开始位置（与main_colab.py一致）
    input_length = len(inputs['input_ids'][0])
    if len(outputs.shape) > 1:
        generated_ids = outputs[0][input_length:]
        full_output_ids = outputs[0]  # 完整的输出包括输入部分
    else:
        generated_ids = outputs[input_length:]
        full_output_ids = outputs
    
    # 解码生成的文本
    output_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    
    # 计算答案位置（与main_colab.py完全一致的逻辑）
    raw_answer_start = input_length
    raw_answer_end = len(full_output_ids)
    
    # 解码生成的文本来找真正的答案位置
    if raw_answer_end > raw_answer_start:
        generated_text = tokenizer.decode(full_output_ids[raw_answer_start:raw_answer_end], skip_special_tokens=True)
        
        # 尝试找到'ans:'后面的内容
        if 'ans:' in generated_text.lower():
            # 在生成的文本中找到ans:的位置
            ans_pos = generated_text.lower().find('ans:')
            if ans_pos >= 0:
                # 计算ans:后面的token位置
                ans_part = generated_text[ans_pos + 4:].strip()  # ans:后面的内容
                if ans_part:
                    # 重新tokenize来找准确位置
                    ans_prefix = generated_text[:ans_pos + 4]  # 包含'ans:'
                    ans_prefix_tokens = tokenizer(ans_prefix, add_special_tokens=False)['input_ids']
                    answer_start_idx = raw_answer_start + len(ans_prefix_tokens)
                else:
                    answer_start_idx = raw_answer_start
                answer_end_idx = raw_answer_end
            else:
                answer_start_idx = raw_answer_start
                answer_end_idx = raw_answer_end
        else:
            # 如果没有找到ans:，使用整个生成的部分
            answer_start_idx = raw_answer_start
            answer_end_idx = raw_answer_end
    else:
        answer_start_idx = raw_answer_start
        answer_end_idx = raw_answer_end
    
    # 确保答案范围有效（与main_colab.py一致）
    if answer_end_idx > len(full_output_ids):
        answer_end_idx = len(full_output_ids)
    if answer_start_idx >= answer_end_idx:
        answer_start_idx = raw_answer_start
        answer_end_idx = raw_answer_end
    
    if answer_start_idx >= answer_end_idx:
        return None, None, None, None
    
    return full_output_ids, output_text, answer_start_idx, answer_end_idx

def calculate_gass_for_layer(full_output_ids, trimmed_triples, gold_knowledge, 
                           answer_start_idx, answer_end_idx, model, tokenizer, layer):
    """使用已有的generation结果计算特定layer的GASS"""
    
    try:
        # 为了准确复现之前的行为，根据layer选择调用方式
        if layer == 32:  
            # Layer 32: 复现之前main_colab.py在Llama3上的行为（明确使用layer_indices=[-1]）
            gass_score = calculate_gass(
                model=model,
                tokenizer=tokenizer,
                input_ids=full_output_ids,
                retrieved_subgraph=trimmed_triples,
                gold_subgraph=gold_knowledge,
                answer_start_idx=answer_start_idx,
                answer_end_idx=answer_end_idx,
                debug=False,
                layer_indices=[-1],  # 明确指定最后一层
                model_type="llama3"
            )
        else:  
            # 其他层: 明确指定layer和model_type
            gass_score = calculate_gass(
                model=model,
                tokenizer=tokenizer,
                input_ids=full_output_ids,
                retrieved_subgraph=trimmed_triples,
                gold_subgraph=gold_knowledge,
                answer_start_idx=answer_start_idx,
                answer_end_idx=answer_end_idx,
                debug=False,
                layer_indices=[layer],
                model_type="llama3"
            )
        return gass_score
    except Exception as e:
        print(f"❌ Error calculating GASS for layer {layer}: {e}")
        return None

def calculate_gass_for_generated_answer(question, trimmed_triples, gold_knowledge, model, tokenizer, layer):
    """计算实际generation的GASS分数，完全复现main_colab.py的方式"""
    
    # 系统提示（与main_colab.py一致）
    system_prompt = """Based on the triples retrieved from a knowledge graph, please answer the question. Please return ONLY the answer entities as a list, each prefixed with "ans:". Do not include explanations or reasoning."""
    
    # 构建Knowledge部分（与main_colab.py一致）
    triples_text = "Knowledge:\n"
    for h, r, t in trimmed_triples:
        triples_text += f"{h} {r} {t}\n"
    
    # 构建用户输入（与main_colab.py一致）
    user_content = f"{triples_text}\nQuestion: {question}\nAnswer:"
    
    # 构建完整的提示（与main_colab.py一致）
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content}
    ]
    
    # 准备输入（与main_colab.py一致）
    input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    # 实际生成（与main_colab.py完全一致）
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
        try:
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                num_return_sequences=1,
                do_sample=False,  # 使用贪婪解码
                temperature=1.0,
                use_cache=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                output_attentions=False  # 生成时不需要attention
            )
        except Exception as gen_error:
            print(f"⚠️ Generation failed: {gen_error}")
            return None, None
    
    # 找到生成文本的开始位置（与main_colab.py一致）
    input_length = len(inputs['input_ids'][0])
    if len(outputs.shape) > 1:
        generated_ids = outputs[0][input_length:]
        full_output_ids = outputs[0]  # 完整的输出包括输入部分
    else:
        generated_ids = outputs[input_length:]
        full_output_ids = outputs
    
    # 解码生成的文本
    output_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    
    # 计算答案位置（与main_colab.py完全一致的逻辑）
    raw_answer_start = input_length
    raw_answer_end = len(full_output_ids)
    
    # 解码生成的文本来找真正的答案位置
    if raw_answer_end > raw_answer_start:
        generated_text = tokenizer.decode(full_output_ids[raw_answer_start:raw_answer_end], skip_special_tokens=True)
        
        # 尝试找到'ans:'后面的内容
        if 'ans:' in generated_text.lower():
            # 在生成的文本中找到ans:的位置
            ans_pos = generated_text.lower().find('ans:')
            if ans_pos >= 0:
                # 计算ans:后面的token位置
                ans_part = generated_text[ans_pos + 4:].strip()  # ans:后面的内容
                if ans_part:
                    # 重新tokenize来找准确位置
                    ans_prefix = generated_text[:ans_pos + 4]  # 包含'ans:'
                    ans_prefix_tokens = tokenizer(ans_prefix, add_special_tokens=False)['input_ids']
                    answer_start_idx = raw_answer_start + len(ans_prefix_tokens)
                else:
                    answer_start_idx = raw_answer_start
                answer_end_idx = raw_answer_end
            else:
                answer_start_idx = raw_answer_start
                answer_end_idx = raw_answer_end
        else:
            # 如果没有找到ans:，使用整个生成的部分
            answer_start_idx = raw_answer_start
            answer_end_idx = raw_answer_end
    else:
        answer_start_idx = raw_answer_start
        answer_end_idx = raw_answer_end
    
    # 确保答案范围有效（与main_colab.py一致）
    if answer_end_idx > len(full_output_ids):
        answer_end_idx = len(full_output_ids)
    if answer_start_idx >= answer_end_idx:
        answer_start_idx = raw_answer_start
        answer_end_idx = raw_answer_end
    
    if answer_start_idx >= answer_end_idx:
        return None, None
    
    try:
        # 为了准确复现之前的行为，所有层都明确指定layer_indices
        if layer == 32:  
            # Layer 32: 复现之前main_colab.py在Llama3上的行为（明确使用layer_indices=[-1]）
            gass_score = calculate_gass(
                model=model,
                tokenizer=tokenizer,
                input_ids=full_output_ids,
                retrieved_subgraph=trimmed_triples,
                gold_subgraph=gold_knowledge,
                answer_start_idx=answer_start_idx,
                answer_end_idx=answer_end_idx,
                debug=False,
                layer_indices=[-1],  # 明确指定最后一层
                model_type="llama3"
            )
        else:  
            # 其他层: 明确指定layer和model_type
            gass_score = calculate_gass(
                model=model,
                tokenizer=tokenizer,
                input_ids=full_output_ids,
                retrieved_subgraph=trimmed_triples,
                gold_subgraph=gold_knowledge,
                answer_start_idx=answer_start_idx,
                answer_end_idx=answer_end_idx,
                debug=False,
                layer_indices=[layer],
                model_type="llama3"
            )
        return gass_score, output_text
    except Exception as e:
        print(f"❌ Error calculating GASS: {e}")
        return None, output_text

def test_gass_layers(model, tokenizer, samples, candidate_layers=[32, 31, 30, 28, 25, 20], output_file=None):
    """测试不同层的GASS分数，边跑边保存"""
    print(f"🧪 Testing GASS for layers: {candidate_layers}")
    
    if output_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"llama3_layer_gass_results_{timestamp}.jsonl"
    
    processed_count = 0
    
    with open(output_file, 'w', encoding='utf-8') as f:
        # 写入配置信息（模仿main_colab.py）
        config = {
            'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S"),
            'script': 'test_gass_direction.py',
            'model': 'meta-llama/Meta-Llama-3-8B-Instruct',
            'total_samples': len(samples),
            'candidate_layers': candidate_layers,
            'pytorch_version': torch.__version__,
            'cuda_version': torch.version.cuda if torch.cuda.is_available() else 'N/A'
        }
        f.write(json.dumps({'config': config}, ensure_ascii=False) + '\n')
        f.flush()
        
        for i, sample in enumerate(tqdm(samples, desc="Processing samples")):
            try:
                question = sample.get('question', '')
                gold_answers = sample.get('golden_texts', [])
                gold_expansion_set = sample.get('gold_expansion_set', [])
                trimmed_triples = sample.get('trimmed_triples', [])
                
                if not gold_answers or not gold_expansion_set or not trimmed_triples:
                    continue
                
                # 处理金标准知识（使用gold_expansion_set）
                gold_knowledge = []
                for triple in gold_expansion_set:
                    if isinstance(triple, list) and len(triple) == 3:
                        gold_knowledge.append(triple)
                
                if not gold_knowledge:
                    continue
                
                print(f"\n样本 {i}:")
                print(f"  问题: {question}")
                print(f"  金标准: {gold_answers}")
                
                # 进行一次实际generation，获得完整的generation结果
                full_output_ids, generated_answer, answer_start_idx, answer_end_idx = generate_once_for_all_layers(
                    question, trimmed_triples, model, tokenizer
                )
                
                if generated_answer is None:
                    print(f"  ❌ Generation failed, skipping sample")
                    continue
                
                print(f"  生成答案: {generated_answer}")
                
                # 初始化结果
                sample_result = {
                    'sample_id': i,
                    'question': question,
                    'generated_answer': generated_answer,
                    'golden_answers': gold_answers,
                    'gold_expansion_set': gold_expansion_set,
                    'layer_gass': {}
                }
                
                # 使用相同的generation结果计算所有层的GASS分数
                for layer in candidate_layers:
                    gass_score = calculate_gass_for_layer(
                        full_output_ids, trimmed_triples, gold_knowledge, 
                        answer_start_idx, answer_end_idx, model, tokenizer, layer
                    )
                    
                    if gass_score is not None:
                        sample_result['layer_gass'][layer] = gass_score
                        print(f"  Layer {layer}: GASS={gass_score:.4f}")
                    else:
                        sample_result['layer_gass'][layer] = None
                        print(f"  Layer {layer}: GASS=ERROR")
                
                # 立即写入文件（模仿main_colab.py）
                f.write(json.dumps(sample_result, ensure_ascii=False) + '\n')
                f.flush()  # 确保及时写入
                processed_count += 1
                
                # 每10个样本输出一次进度
                if processed_count % 10 == 0:
                    print(f"✅ Processed {processed_count} samples, saved to {output_file}")
            
            except Exception as e:
                print(f"❌ Error processing sample {i}: {e}")
                continue
    
    print(f"\n🎉 Total processed: {processed_count} samples")
    print(f"💾 Results saved to: {output_file}")
    return output_file, processed_count

def analyze_layer_results(results):
    """分析不同层的GASS结果"""
    print("\n📊 Layer GASS Analysis:")
    print("=" * 60)
    
    # 收集各层数据
    layer_data = {}
    layers = set()
    
    for sample in results:
        for layer, gass in sample['layer_gass'].items():
            if layer not in layer_data:
                layer_data[layer] = []
            layer_data[layer].append(gass)
            layers.add(layer)
    
    # 输出统计
    print(f"{'Layer':<6} {'Samples':<8} {'Mean':<8} {'Std':<8} {'Min':<8} {'Max':<8}")
    print("=" * 60)
    
    for layer in sorted(layers, reverse=True):
        if layer in layer_data and layer_data[layer]:
            scores = layer_data[layer]
            mean_score = np.mean(scores)
            std_score = np.std(scores)
            min_score = np.min(scores)
            max_score = np.max(scores)
            
            print(f"{layer:<6} {len(scores):<8} {mean_score:<8.4f} {std_score:<8.4f} "
                  f"{min_score:<8.4f} {max_score:<8.4f}")
    
    return layer_data

def main():
    """主函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='Llama3 Layer GASS Test')
    parser.add_argument('-n', '--num_samples', type=int, default=50, 
                        help='Number of samples to process (default: 50)')
    parser.add_argument('--layers', nargs='+', type=int, default=[32, 28, 24, 20, 16, 12, 8, 4],
                        help='Layers to test (default: coarse grid 32 28 24 20 16 12 8 4)')
    parser.add_argument('-o', '--output', type=str, default=None,
                        help='Output file name (default: auto-generated)')
    
    args = parser.parse_args()
    
    print("🔬 Llama3 Layer GASS Test - 粗粒度栅格扫描")
    print("目标：寻找GASS方向拐点 (truthful > hallucinated)")
    print("=" * 60)
    print(f"📊 Processing {args.num_samples} samples")
    print(f"🔢 Coarse grid layers: {args.layers}")
    print("策略：每隔4-5层采样，快速锁定反转点位置")
    
    # 候选层
    candidate_layers = args.layers
    
    # 加载模型
    model, tokenizer = setup_model()
    
    # 加载测试数据
    samples = load_test_data(num_samples=args.num_samples)
    
    # 测试不同层（边跑边保存）
    output_file, processed_count = test_gass_layers(model, tokenizer, samples, candidate_layers, args.output)
    
    print(f"\n✅ Layer analysis complete!")
    print(f"📁 Output file: {output_file}")
    print(f"📊 Processed samples: {processed_count}")
    print("\n💡 Tip: 可以随时中断并查看已保存的结果！")

if __name__ == "__main__":
    main()