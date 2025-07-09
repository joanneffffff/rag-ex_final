#!/usr/bin/env python3
"""
测试reranker加载
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from xlm.ui.optimized_rag_ui import try_load_qwen_reranker
from config.parameters import config

def test_reranker_loading():
    """测试reranker加载"""
    print("=" * 60)
    print("测试reranker加载")
    print("=" * 60)
    
    try:
        print(f"尝试加载reranker: {config.reranker.model_name}")
        print(f"缓存目录: {config.reranker.cache_dir}")
        
        reranker = try_load_qwen_reranker(
            model_name=config.reranker.model_name,
            cache_dir=config.reranker.cache_dir
        )
        
        if reranker is not None:
            print("✅ Reranker加载成功")
            print(f"Reranker类型: {type(reranker)}")
        else:
            print("❌ Reranker加载失败")
            
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_reranker_loading() 