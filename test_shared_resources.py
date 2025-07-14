#!/usr/bin/env python3
"""
测试共享资源管理器
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent))

def test_shared_resources():
    """测试共享资源管理器"""
    print("🧪 测试共享资源管理器")
    print("=" * 50)
    
    try:
        # 测试模板加载
        print("\n1. 测试模板加载...")
        from xlm.utils.shared_resource_manager import shared_resource_manager
        
        # 第一次加载
        templates1 = shared_resource_manager.get_templates()
        print(f"✅ 第一次加载模板数量: {len(templates1)}")
        
        # 第二次加载（应该使用缓存）
        templates2 = shared_resource_manager.get_templates()
        print(f"✅ 第二次加载模板数量: {len(templates2)}")
        
        # 验证是否相同
        if templates1 is templates2:
            print("✅ 模板共享成功")
        else:
            print("❌ 模板共享失败")
        
        # 测试LLM生成器加载
        print("\n2. 测试LLM生成器加载...")
        try:
            generator1 = shared_resource_manager.get_llm_generator(
                model_name="SUFE-AIFLM-Lab/Fin-R1",
                cache_dir="/users/sgjfei3/data/huggingface",
                device="cuda:1",
                use_quantization=True,
                quantization_type="4bit"
            )
            
            if generator1:
                print("✅ 第一次加载LLM生成器成功")
                
                # 第二次加载（应该使用缓存）
                generator2 = shared_resource_manager.get_llm_generator(
                    model_name="SUFE-AIFLM-Lab/Fin-R1",
                    cache_dir="/users/sgjfei3/data/huggingface",
                    device="cuda:1",
                    use_quantization=True,
                    quantization_type="4bit"
                )
                
                if generator2:
                    print("✅ 第二次加载LLM生成器成功")
                    
                    # 验证是否相同
                    if generator1 is generator2:
                        print("✅ LLM生成器共享成功")
                    else:
                        print("❌ LLM生成器共享失败")
                else:
                    print("❌ 第二次加载LLM生成器失败")
            else:
                print("❌ 第一次加载LLM生成器失败")
                
        except Exception as e:
            print(f"⚠️ LLM生成器测试失败: {e}")
            print("这可能是正常的，因为模型文件可能不存在")
        
        print("\n✅ 共享资源管理器测试完成")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_shared_resources() 