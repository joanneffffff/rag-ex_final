#!/usr/bin/env python3
"""
测试多阶段检索系统初始化
"""

import sys
import os
from pathlib import Path

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent))

def test_multi_stage_init():
    """测试多阶段检索系统初始化"""
    print("=== 测试多阶段检索系统初始化 ===")
    
    # 1. 检查数据文件
    data_path = Path("data/alphafin/alphafin_merged_generated_qa.json")
    print(f"1. 检查数据文件: {data_path}")
    print(f"   文件存在: {data_path.exists()}")
    if data_path.exists():
        print(f"   文件大小: {data_path.stat().st_size} bytes")
    
    # 2. 检查模块导入
    print("\n2. 检查模块导入...")
    try:
        from alphafin_data_process.multi_stage_retrieval_final import MultiStageRetrievalSystem
        print("   ✅ 模块导入成功")
    except ImportError as e:
        print(f"   ❌ 模块导入失败: {e}")
        return
    
    # 3. 尝试初始化
    print("\n3. 尝试初始化多阶段检索系统...")
    try:
        print("   开始初始化...")
        retrieval_system = MultiStageRetrievalSystem(
            data_path=data_path,
            dataset_type="chinese",
            use_existing_config=True
        )
        print("   ✅ 多阶段检索系统初始化成功")
        
        # 4. 测试简单查询
        print("\n4. 测试简单查询...")
        try:
            results = retrieval_system.search(
                query="德赛电池的业绩如何？",
                top_k=5
            )
            print(f"   ✅ 查询成功，返回 {len(results.get('retrieved_documents', []))} 个结果")
        except Exception as e:
            print(f"   ❌ 查询失败: {e}")
            
    except Exception as e:
        print(f"   ❌ 初始化失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_multi_stage_init() 