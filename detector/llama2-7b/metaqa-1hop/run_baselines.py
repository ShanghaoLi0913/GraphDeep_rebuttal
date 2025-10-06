#!/usr/bin/env python3
"""
运行所有独立baseline方法的脚本
"""

import subprocess
import sys
import argparse
import os
from datetime import datetime

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
    parser = argparse.ArgumentParser(description='Run Individual Baseline Methods')
    parser.add_argument('--method', type=str, choices=[
        'bertscore', 'perplexity', 'token_confidence', 'entity_overlap', 
        'max_token_prob', 'nli_contradiction', 
        'embedding_semantic_divergence', 'all'
    ], default='all', help='Which baseline method to run')
    parser.add_argument('--train_samples', type=int, default=100, help='Number of training samples')
    parser.add_argument('--test_samples', type=int, default=50, help='Number of test samples')
    args = parser.parse_args()
    
    # 确保当前在正确的目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    # 定义所有baseline方法
    baselines = {
        'bertscore': ('BERTScore', 'baseline_bertscore.py'),
        'perplexity': ('Perplexity', 'baseline_perplexity.py'),
        'token_confidence': ('Token Confidence', 'baseline_token_confidence.py'),
        'entity_overlap': ('Entity Overlap', 'baseline_entity_overlap.py'),
        'max_token_prob': ('Max Token Probability', 'baseline_max_token_prob.py'),
        'nli_contradiction': ('NLI-based Contradiction Detection', 'baseline_nli_contradiction.py'),
        'embedding_semantic_divergence': ('Embedding-Based Semantic Divergence', 'baseline_embedding_semantic_divergence.py'),
        # 'uncertainty_quantification': 移除 - AUC=0.5，无区分能力
    }
    
    print("🔥 Individual Baseline Methods Runner")
    print(f"⏰ Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📊 Configuration: Train={args.train_samples}, Test={args.test_samples}")
    
    results = {}
    
    if args.method == 'all':
        # 运行所有baseline
        for method_key, (method_name, script_path) in baselines.items():
            success = run_baseline(method_name, script_path, args.train_samples, args.test_samples)
            results[method_name] = success
    else:
        # 运行单个baseline
        if args.method in baselines:
            method_name, script_path = baselines[args.method]
            success = run_baseline(method_name, script_path, args.train_samples, args.test_samples)
            results[method_name] = success
        else:
            print(f"❌ Unknown method: {args.method}")
            return
    
    # 总结结果
    print(f"\n{'='*60}")
    print("📋 SUMMARY")
    print(f"{'='*60}")
    
    successful = 0
    failed = 0
    
    for method_name, success in results.items():
        status = "✅ Success" if success else "❌ Failed"
        print(f"{method_name:<20}: {status}")
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