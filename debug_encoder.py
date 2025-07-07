#!/usr/bin/env python3
"""
调试英文编码器的脚本
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from xlm.components.encoder.finbert import FinbertEncoder
from config.parameters import Config
import torch

def test_encoder():
    print("=== 调试英文编码器 ===")
    
    config = Config()
    
    # 1. 检查设备
    print(f"1. 检查设备...")
    print(f"   CUDA可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   CUDA设备数量: {torch.cuda.device_count()}")
        print(f"   当前设备: {torch.cuda.current_device()}")
        print(f"   设备名称: {torch.cuda.get_device_name()}")
    
    # 2. 初始化编码器
    print(f"\n2. 初始化英文编码器...")
    print(f"   模型路径: {config.encoder.english_model_path}")
    print(f"   缓存目录: {config.encoder.cache_dir}")
    print(f"   设备: {config.encoder.device}")
    
    try:
        encoder = FinbertEncoder(
            model_name=config.encoder.english_model_path,
            cache_dir=config.encoder.cache_dir,
            device=config.encoder.device
        )
        print("   ✅ 编码器初始化成功")
    except Exception as e:
        print(f"   ❌ 编码器初始化失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 3. 测试简单编码
    print(f"\n3. 测试简单编码...")
    test_texts = [
        "How was internally developed software capitalised?",
        "This is a test sentence for encoding.",
        "Internally developed software is capitalised at cost less accumulated amortisation."
    ]
    
    try:
        print(f"   编码 {len(test_texts)} 个测试文本...")
        embeddings = encoder.encode(test_texts, batch_size=2, show_progress_bar=True)
        print(f"   ✅ 编码成功，嵌入向量形状: {embeddings.shape}")
        print(f"   嵌入维度: {embeddings.shape[1]}")
        
        # 检查嵌入向量是否合理
        print(f"   嵌入向量统计:")
        print(f"     最小值: {embeddings.min():.4f}")
        print(f"     最大值: {embeddings.max():.4f}")
        print(f"     平均值: {embeddings.mean():.4f}")
        print(f"     标准差: {embeddings.std():.4f}")
        
    except Exception as e:
        print(f"   ❌ 编码失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 4. 测试单个文本编码
    print(f"\n4. 测试单个文本编码...")
    try:
        single_embedding = encoder.encode(["Single test sentence"])
        print(f"   ✅ 单个文本编码成功，形状: {single_embedding.shape}")
    except Exception as e:
        print(f"   ❌ 单个文本编码失败: {e}")
        import traceback
        traceback.print_exc()
    
    # 5. 测试从数据文件加载的文本
    print(f"\n5. 测试数据文件中的文本...")
    try:
        import json
        with open("data/unified/tatqa_knowledge_base_combined.jsonl", 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= 3:  # 只测试前3行
                    break
                item = json.loads(line.strip())
                context = item.get('context', '')
                if context:
                    print(f"   测试第{i+1}行文本 (长度: {len(context)})...")
                    # 截取前500字符进行测试
                    test_context = context[:500]
                    try:
                        context_embedding = encoder.encode([test_context])
                        print(f"   ✅ 第{i+1}行编码成功，形状: {context_embedding.shape}")
                    except Exception as e:
                        print(f"   ❌ 第{i+1}行编码失败: {e}")
                        break
    except Exception as e:
        print(f"   ❌ 读取数据文件失败: {e}")
    
    print(f"\n=== 调试完成 ===")

if __name__ == "__main__":
    test_encoder() 