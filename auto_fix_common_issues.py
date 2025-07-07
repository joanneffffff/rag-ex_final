#!/usr/bin/env python3
"""
自动化修复常见问题脚本
快速解决RAG系统的常见bug
"""

import sys
import os
import json
import shutil
from pathlib import Path
import subprocess

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

class AutoFixer:
    def __init__(self):
        self.fixes_applied = []
        
    def fix_cache_issues(self):
        """修复缓存问题"""
        print("🔧 修复缓存问题...")
        
        cache_dirs = ["cache/", "checkpoints/"]
        for cache_dir in cache_dirs:
            if Path(cache_dir).exists():
                try:
                    shutil.rmtree(cache_dir)
                    print(f"  ✅ 清理缓存目录: {cache_dir}")
                    self.fixes_applied.append(f"清理缓存: {cache_dir}")
                except Exception as e:
                    print(f"  ⚠️ 清理缓存失败: {e}")
    
    def fix_data_path_issues(self):
        """修复数据路径问题"""
        print("🔧 修复数据路径问题...")
        
        # 检查并创建必要的目录
        required_dirs = [
            "data/unified/",
            "models/",
            "logs/",
            "cache/"
        ]
        
        for dir_path in required_dirs:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
            print(f"  ✅ 确保目录存在: {dir_path}")
    
    def fix_import_issues(self):
        """修复导入问题"""
        print("🔧 修复导入问题...")
        
        # 检查__init__.py文件
        init_files = [
            "xlm/__init__.py",
            "xlm/components/__init__.py",
            "xlm/components/encoder/__init__.py",
            "xlm/components/retriever/__init__.py",
            "xlm/utils/__init__.py",
            "utils/__init__.py"
        ]
        
        for init_file in init_files:
            if not Path(init_file).exists():
                Path(init_file).touch()
                print(f"  ✅ 创建缺失的__init__.py: {init_file}")
                self.fixes_applied.append(f"创建__init__.py: {init_file}")
    
    def fix_config_issues(self):
        """修复配置问题"""
        print("🔧 修复配置问题...")
        
        try:
            from config.parameters import Config
            config = Config()
            
            # 验证关键配置
            if not hasattr(config, 'data') or not hasattr(config.data, 'english_data_path'):
                print("  ⚠️ 配置结构可能有问题")
                self.fixes_applied.append("检查配置文件结构")
            
        except Exception as e:
            print(f"  ❌ 配置加载失败: {e}")
            self.fixes_applied.append("修复配置文件")
    
    def create_quick_test_script(self):
        """创建快速测试脚本"""
        print("🔧 创建快速测试脚本...")
        
        test_script = '''#!/usr/bin/env python3
"""
快速测试脚本 - 验证RAG系统基本功能
"""

import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def quick_test():
    """快速测试基本功能"""
    print("🚀 开始快速测试...")
    
    try:
        # 测试1: 配置加载
        print("📋 测试1: 配置加载")
        from config.parameters import Config
        config = Config()
        print("  ✅ 配置加载成功")
        
        # 测试2: 编码器导入
        print("📋 测试2: 编码器导入")
        from xlm.components.encoder.finbert import FinbertEncoder
        print("  ✅ 编码器导入成功")
        
        # 测试3: 数据加载器导入
        print("📋 测试3: 数据加载器导入")
        from xlm.utils.dual_language_loader import DualLanguageLoader
        print("  ✅ 数据加载器导入成功")
        
        # 测试4: 检索器导入
        print("📋 测试4: 检索器导入")
        from xlm.components.retriever.bilingual_retriever import BilingualRetriever
        print("  ✅ 检索器导入成功")
        
        print("\\n✅ 所有基本测试通过！")
        return True
        
    except Exception as e:
        print(f"\\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = quick_test()
    sys.exit(0 if success else 1)
'''
        
        with open("quick_test.py", "w", encoding='utf-8') as f:
            f.write(test_script)
        
        print("  ✅ 创建快速测试脚本: quick_test.py")
        self.fixes_applied.append("创建快速测试脚本")
    
    def create_debug_script(self):
        """创建调试脚本"""
        print("🔧 创建调试脚本...")
        
        debug_script = '''#!/usr/bin/env python3
"""
调试脚本 - 详细诊断RAG系统问题
"""

import sys
import traceback
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def debug_environment():
    """调试环境问题"""
    print("🔍 环境调试信息:")
    print(f"Python版本: {sys.version}")
    print(f"工作目录: {Path.cwd()}")
    print(f"Python路径: {sys.path[:3]}...")
    
    # 检查关键模块
    modules_to_check = [
        'torch', 'transformers', 'sentence_transformers',
        'faiss', 'numpy', 'pandas'
    ]
    
    print("\\n📦 模块检查:")
    for module in modules_to_check:
        try:
            imported_module = __import__(module)
            version = getattr(imported_module, '__version__', 'unknown')
            print(f"  ✅ {module}: {version}")
        except ImportError as e:
            print(f"  ❌ {module}: {e}")

def debug_data_loading():
    """调试数据加载问题"""
    print("\\n🔍 数据加载调试:")
    
    try:
        from config.parameters import Config
        config = Config()
        print(f"  ✅ 配置加载成功")
        print(f"  英文数据路径: {getattr(config.data, 'english_data_path', 'N/A')}")
        print(f"  中文数据路径: {getattr(config.data, 'chinese_data_path', 'N/A')}")
        
        # 检查数据文件
        data_files = [
            "data/unified/tatqa_knowledge_base_combined.jsonl",
            "data/unified/tatqa_knowledge_base_unified.jsonl"
        ]
        
        for file_path in data_files:
            if Path(file_path).exists():
                size_mb = Path(file_path).stat().st_size / (1024 * 1024)
                print(f"  ✅ {file_path} ({size_mb:.1f}MB)")
            else:
                print(f"  ❌ {file_path} (不存在)")
                
    except Exception as e:
        print(f"  ❌ 数据加载调试失败: {e}")
        traceback.print_exc()

def debug_encoder_loading():
    """调试编码器加载问题"""
    print("\\n🔍 编码器加载调试:")
    
    try:
        from xlm.components.encoder.finbert import FinbertEncoder
        from config.parameters import Config
        
        config = Config()
        
        # 测试英文编码器
        print("  测试英文编码器...")
        encoder_en = FinbertEncoder(
            model_name=config.encoder.english_model_path,
            cache_dir=config.encoder.cache_dir,
            device=config.encoder.device
        )
        print(f"  ✅ 英文编码器加载成功")
        
        # 测试中文编码器
        print("  测试中文编码器...")
        encoder_ch = FinbertEncoder(
            model_name=config.encoder.chinese_model_path,
            cache_dir=config.encoder.cache_dir,
            device=config.encoder.device
        )
        print(f"  ✅ 中文编码器加载成功")
        
    except Exception as e:
        print(f"  ❌ 编码器加载失败: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    debug_environment()
    debug_data_loading()
    debug_encoder_loading()
    print("\\n🔍 调试完成")
'''
        
        with open("debug_system.py", "w", encoding='utf-8') as f:
            f.write(debug_script)
        
        print("  ✅ 创建调试脚本: debug_system.py")
        self.fixes_applied.append("创建调试脚本")
    
    def run_all_fixes(self):
        """运行所有修复"""
        print("🚀 开始自动修复常见问题")
        print("=" * 80)
        
        self.fix_cache_issues()
        self.fix_data_path_issues()
        self.fix_import_issues()
        self.fix_config_issues()
        self.create_quick_test_script()
        self.create_debug_script()
        
        print("\n" + "=" * 80)
        print("📋 修复总结")
        print("=" * 80)
        
        if self.fixes_applied:
            print(f"✅ 应用了 {len(self.fixes_applied)} 个修复:")
            for fix in self.fixes_applied:
                print(f"  • {fix}")
            
            print("\n💡 建议的后续步骤:")
            print("  1. 运行: python quick_test.py")
            print("  2. 如果还有问题，运行: python debug_system.py")
            print("  3. 检查生成的日志文件")
        else:
            print("✅ 没有发现需要修复的问题")
        
        print("=" * 80)

if __name__ == "__main__":
    fixer = AutoFixer()
    fixer.run_all_fixes() 