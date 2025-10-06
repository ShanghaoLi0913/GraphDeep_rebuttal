"""
应用Llama-2平衡校准到数据
"""

import json
import os
from balanced_calibration import BalancedCalibrator
from tqdm import tqdm

def apply_balanced_calibration_to_file(input_file, output_file, calibrator):
    """对单个文件应用平衡校准"""
    processed_lines = []
    total_samples = 0
    calibrated_samples = 0
    
    print(f"📝 Processing: {os.path.basename(input_file)}")
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Processing lines"):
            if not line.strip():
                continue
            
            try:
                data = json.loads(line)
                
                # 跳过配置和统计行
                if 'config' in data or 'stats' in data:
                    processed_lines.append(line.strip())
                    continue
                
                total_samples += 1
                
                # 应用平衡校准（如果没有平衡校准分数）
                if 'balanced_calibrated_gass' not in data and 'gass_score' in data:
                    original_gass = data['gass_score']
                    balanced_calibrated = calibrator.calibrate_score(original_gass)
                    data['balanced_calibrated_gass'] = balanced_calibrated
                    calibrated_samples += 1
                
                processed_lines.append(json.dumps(data, ensure_ascii=False))
                
            except Exception as e:
                processed_lines.append(line.strip())
    
    # 写入输出文件
    with open(output_file, 'w', encoding='utf-8') as f:
        for line in processed_lines:
            f.write(line + '\n')
    
    print(f"✅ 处理了 {total_samples} 个样本，校准了 {calibrated_samples} 个")
    return True

def main():
    """主函数"""
    print("🚀 应用Llama-2平衡校准到现有数据")
    
    # 加载训练好的校准器
    print("📂 Loading Llama-2 balanced calibrator...")
    calibrator = BalancedCalibrator()
    calibrator.load_model("calibration_models/balanced_calibration_llama2_dev_20250711_143226")
    
    # 要处理的文件
    files_to_process = [
        {
            'input': 'experiment_records/inference_results/Llama2-7b/colab_train_simple_part1&2.jsonl',
            'output': 'experiment_records/inference_results/Llama2-7b/colab_train_simple_part1&2_balanced_calibrated.jsonl'
        }
    ]
    
    for file_config in files_to_process:
        input_file = file_config['input']
        output_file = file_config['output']
        
        if not os.path.exists(input_file):
            print(f"⚠️ 文件不存在: {input_file}")
            continue
        
        if os.path.exists(output_file):
            print(f"⏭️ 输出文件已存在，跳过: {output_file}")
            continue
        
        # 应用校准
        apply_balanced_calibration_to_file(input_file, output_file, calibrator)
        print(f"🎉 已创建: {os.path.basename(output_file)}")
    
    print("✅ Llama-2平衡校准应用完成！")

if __name__ == "__main__":
    main()