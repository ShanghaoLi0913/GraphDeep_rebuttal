#!/usr/bin/env python3
"""
快速运行baseline方法 - Colab L4 GPU优化版
优先运行轻量级方法，最后运行重GPU方法
"""

import subprocess
import sys
import argparse
import os
from datetime import datetime
import torch

def get_gpu_info():
    """获取GPU信息"""
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        return gpu_name, gpu_memory_gb
    return "CPU", 0

def run_baseline(baseline_name, script_path, train_samples=100, test_samples=50):
    """运行单个baseline"""
    print(f"\n{'='*60}")
    print(f"🚀 Running {baseline_name}")
    print(f"{'='*60}")
    
    cmd = [
        sys.executable, script_path,
        '--train_samples', str(train_samples),
        '--test_samples', str(test_samples)
    ]
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False, text=True)
        print(f"✅ {baseline_name} completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {baseline_name} failed with error code {e.returncode}")
        return False
    except Exception as e:
        print(f"❌ Error running {baseline_name}: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Fast Baseline Runner for Colab L4 GPU')
    parser.add_argument('--method', type=str, choices=[
        'bertscore', 'perplexity', 'entity_overlap', 
        'embedding_semantic_divergence', 'nli_contradiction',
        'token_confidence', 'max_token_prob', 'all', 'fast'
    ], default='fast', help='Which baseline method to run')
    parser.add_argument('--train_samples', type=int, default=300, help='Number of training samples')
    parser.add_argument('--test_samples', type=int, default=100, help='Number of test samples')
    args = parser.parse_args()
    
    # 确保当前在正确的目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    # GPU信息
    gpu_name, gpu_memory_gb = get_gpu_info()
    print(f"🖥️ GPU: {gpu_name} ({gpu_memory_gb:.1f} GB)")
    
    # 🚀 按计算复杂度排序的baseline方法 (轻量级 -> 重量级)
    # 轻量级方法：不需要大模型
    lightweight_baselines = {
        'bertscore': ('BERTScore', 'baseline_bertscore.py'),
        'entity_overlap': ('Entity Overlap', 'baseline_entity_overlap.py'),
        'embedding_semantic_divergence': ('Embedding-Based Semantic Divergence', 'baseline_embedding_semantic_divergence.py'),
    }
    
    # 中等方法：需要模型但计算较轻
    medium_baselines = {
        'perplexity': ('Perplexity', 'baseline_perplexity.py'),
        'nli_contradiction': ('NLI-based Contradiction Detection', 'baseline_nli_contradiction.py'),
    }
    
    # 重量级方法：占用大量GPU资源
    heavy_baselines = {
        'token_confidence': ('Token Confidence', 'baseline_token_confidence.py'),
        'max_token_prob': ('Max Token Probability', 'baseline_max_token_prob.py'),
    }
    
    print("🔥 Fast Baseline Methods Runner - L4 GPU Optimized")
    print(f"⏰ Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📊 Configuration: Train={args.train_samples}, Test={args.test_samples}")
    
    results = {}
    
    if args.method == 'fast':
        # 快速模式：只运行最有效的方法
        fast_baselines = {
            'bertscore': lightweight_baselines['bertscore'],
            'entity_overlap': lightweight_baselines['entity_overlap'],
            'perplexity': medium_baselines['perplexity'],
        }
        print("⚡ Fast mode: Running top 3 most effective baselines")
        baselines_to_run = fast_baselines
        
    elif args.method == 'all':
        # 全部方法：按顺序运行
        baselines_to_run = {**lightweight_baselines, **medium_baselines, **heavy_baselines}
        print("🔥 Running all baselines in optimized order")
        
    else:
        # 单个方法
        all_baselines = {**lightweight_baselines, **medium_baselines, **heavy_baselines}
        if args.method in all_baselines:
            baselines_to_run = {args.method: all_baselines[args.method]}
        else:
            print(f"❌ Unknown method: {args.method}")
            return
    
    # 运行选定的baseline
    for method_key, (method_name, script_path) in baselines_to_run.items():
        success = run_baseline(method_name, script_path, args.train_samples, args.test_samples)
        results[method_name] = success
        
        # 重量级方法后强制清理内存
        if method_key in heavy_baselines and torch.cuda.is_available():
            print("🧹 Cleaning GPU memory after heavy baseline...")
            torch.cuda.empty_cache()
    
    # 总结结果
    print(f"\n{'='*60}")
    print("📋 SUMMARY")
    print(f"{'='*60}")
    
    successful = 0
    failed = 0
    
    for method_name, success in results.items():
        status = "✅ Success" if success else "❌ Failed"
        print(f"{method_name:<30}: {status}")
        if success:
            successful += 1
        else:
            failed += 1
    
    print(f"\n📊 Total: {successful} successful, {failed} failed")
    print(f"⏰ Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 显示结果文件位置
    results_dir = "results/individual_baselines"
    if os.path.exists(results_dir):
        print(f"\n💾 Results saved in: {os.path.abspath(results_dir)}")
        
        # 列出最新的结果文件
        try:
            files = [f for f in os.listdir(results_dir) if f.endswith('.json')]
            if files:
                files.sort(key=lambda x: os.path.getmtime(os.path.join(results_dir, x)), reverse=True)
                print("📁 Latest result files:")
                for f in files[:5]:  # 显示最新的5个文件
                    print(f"   • {f}")
        except Exception as e:
            print(f"⚠️ Could not list result files: {e}")

if __name__ == "__main__":
    main()