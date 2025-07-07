#!/usr/bin/env python3
"""
诊断英文嵌入向量问题
确定为什么英文嵌入向量为空
"""

import sys
import os
from pathlib import Path
import traceback

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_encoder_loading():
    """测试编码器加载"""
    print("=" * 80)
    print("🔍 测试编码器加载")
    print("=" * 80)
    
    try:
        from xlm.components.encoder.encoder import Encoder
        
        # 测试英文编码器加载
        print("测试英文编码器加载...")
        encoder_en = Encoder(
            model_name="models/finetuned_tatqa_mixed_enhanced",
            device="cuda:0"
        )
        print(f"✅ 英文编码器加载成功")
        print(f"  模型名称: {encoder_en.model_name}")
        print(f"  设备: {encoder_en.device}")
        print(f"  嵌入维度: {encoder_en.model.get_sentence_embedding_dimension()}")
        
        # 测试简单编码
        print("\n测试简单编码...")
        test_text = "This is a test sentence."
        test_embedding = encoder_en.encode([test_text])
        print(f"✅ 简单编码成功，形状: {test_embedding.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ 编码器加载失败: {e}")
        traceback.print_exc()
        return False

def test_document_format():
    """测试文档格式"""
    print("\n" + "=" * 80)
    print("🔍 测试文档格式")
    print("=" * 80)
    
    try:
        # 检查英文数据格式
        data_path = Path("data/unified/tatqa_knowledge_base_combined.jsonl")
        
        if not data_path.exists():
            print(f"❌ 英文数据文件不存在: {data_path}")
            return False
        
        print(f"📁 英文数据文件: {data_path}")
        
        # 读取前几行JSONL数据
        english_records = []
        chinese_records = []
        
        with open(data_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= 10:  # 只检查前10行
                    break
                if line.strip():
                    import json
                    record = json.loads(line)
                    content = record.get('content', '')
                    if content:
                        # 简单检测语言（检查是否包含中文字符）
                        if any('\u4e00' <= char <= '\u9fff' for char in content):
                            chinese_records.append(record)
                        else:
                            english_records.append(record)
        
        print(f"📊 前10条记录分析:")
        print(f"  英文记录: {len(english_records)}")
        print(f"  中文记录: {len(chinese_records)}")
        
        if english_records:
            print(f"\n📋 英文记录示例:")
            for i, record in enumerate(english_records[:3]):
                content = record.get('content', '')[:100] + "..." if len(record.get('content', '')) > 100 else record.get('content', '')
                print(f"  {i+1}. content长度: {len(record.get('content', ''))}, 内容: {content}")
        
        if chinese_records:
            print(f"\n📋 中文记录示例:")
            for i, record in enumerate(chinese_records[:3]):
                content = record.get('content', '')[:100] + "..." if len(record.get('content', '')) > 100 else record.get('content', '')
                print(f"  {i+1}. content长度: {len(record.get('content', ''))}, 内容: {content}")
        
        return True
        
    except Exception as e:
        print(f"❌ 文档格式检查失败: {e}")
        traceback.print_exc()
        return False

def test_batch_encoding():
    """测试批量编码"""
    print("\n" + "=" * 80)
    print("🔍 测试批量编码")
    print("=" * 80)
    
    try:
        from xlm.components.encoder.encoder import Encoder
        
        # 加载编码器
        encoder = Encoder(
            model_name="models/finetuned_tatqa_mixed_enhanced",
            device="cuda:0"
        )
        
        # 准备测试文本
        test_texts = [
            "This is the first test sentence.",
            "This is the second test sentence.",
            "This is the third test sentence.",
            "This is the fourth test sentence.",
            "This is the fifth test sentence."
        ]
        
        print(f"📝 测试文本数量: {len(test_texts)}")
        
        # 测试小批量编码
        print("\n测试小批量编码...")
        embeddings = encoder.encode(test_texts)
        print(f"✅ 小批量编码成功，形状: {embeddings.shape}")
        
        # 测试大批量编码
        print("\n测试大批量编码...")
        large_texts = test_texts * 100  # 500个文本
        print(f"📝 大批量文本数量: {len(large_texts)}")
        
        embeddings_large = encoder.encode(large_texts)
        print(f"✅ 大批量编码成功，形状: {embeddings_large.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ 批量编码测试失败: {e}")
        traceback.print_exc()
        return False

def test_gpu_memory():
    """测试GPU内存"""
    print("\n" + "=" * 80)
    print("🔍 测试GPU内存")
    print("=" * 80)
    
    try:
        import torch
        
        if torch.cuda.is_available():
            print(f"✅ CUDA可用")
            print(f"  GPU数量: {torch.cuda.device_count()}")
            print(f"  当前GPU: {torch.cuda.current_device()}")
            print(f"  GPU名称: {torch.cuda.get_device_name()}")
            
            # 检查GPU内存
            for i in range(torch.cuda.device_count()):
                memory_allocated = torch.cuda.memory_allocated(i) / 1024**3
                memory_reserved = torch.cuda.memory_reserved(i) / 1024**3
                memory_total = torch.cuda.get_device_properties(i).total_memory / 1024**3
                
                print(f"  GPU {i}:")
                print(f"    总内存: {memory_total:.2f} GB")
                print(f"    已分配: {memory_allocated:.2f} GB")
                print(f"    已保留: {memory_reserved:.2f} GB")
                print(f"    可用: {memory_total - memory_reserved:.2f} GB")
        else:
            print("❌ CUDA不可用")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ GPU内存检查失败: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 开始英文嵌入向量问题诊断")
    
    # 测试1: 编码器加载
    test1_passed = test_encoder_loading()
    
    # 测试2: 文档格式
    test2_passed = test_document_format()
    
    # 测试3: 批量编码
    test3_passed = test_batch_encoding()
    
    # 测试4: GPU内存
    test4_passed = test_gpu_memory()
    
    print("\n" + "=" * 80)
    print("�� 诊断结果")
    print("=" * 80)
    print(f"✅ 编码器加载: {'通过' if test1_passed else '失败'}")
    print(f"✅ 文档格式: {'通过' if test2_passed else '失败'}")
    print(f"✅ 批量编码: {'通过' if test3_passed else '失败'}")
    print(f"✅ GPU内存: {'通过' if test4_passed else '失败'}")
    
    if all([test1_passed, test2_passed, test3_passed, test4_passed]):
        print("\n✅ 所有测试通过，英文嵌入向量问题可能是其他原因")
    else:
        print("\n❌ 发现问题，请根据失败的测试进行修复")
    
    print("=" * 80) 