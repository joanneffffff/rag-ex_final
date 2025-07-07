#!/usr/bin/env python3
"""
环境健康检查脚本
快速诊断和修复RAG系统的常见问题
"""

import sys
import os
import json
import traceback
from pathlib import Path
import subprocess
import importlib

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

class EnvironmentHealthChecker:
    def __init__(self):
        self.issues = []
        self.fixes = []
        
    def check_python_environment(self):
        """检查Python环境"""
        print("=" * 80)
        print("🔍 检查Python环境")
        print("=" * 80)
        
        # Python版本
        python_version = sys.version_info
        print(f"Python版本: {python_version.major}.{python_version.minor}.{python_version.micro}")
        
        if python_version < (3, 8):
            self.issues.append("Python版本过低，建议使用3.8+")
            self.fixes.append("升级Python到3.8或更高版本")
        
        # 检查关键依赖
        critical_packages = [
            'torch', 'transformers', 'sentence_transformers', 
            'faiss-cpu', 'numpy', 'pandas', 'gradio'
        ]
        
        print("\n📦 检查关键依赖包:")
        for package in critical_packages:
            try:
                module = importlib.import_module(package)
                version = getattr(module, '__version__', 'unknown')
                print(f"  ✅ {package}: {version}")
            except ImportError:
                print(f"  ❌ {package}: 未安装")
                self.issues.append(f"缺少依赖包: {package}")
                self.fixes.append(f"pip install {package}")
    
    def check_cuda_environment(self):
        """检查CUDA环境"""
        print("\n" + "=" * 80)
        print("🔍 检查CUDA环境")
        print("=" * 80)
        
        try:
            import torch
            print(f"PyTorch版本: {torch.__version__}")
            print(f"CUDA可用: {torch.cuda.is_available()}")
            
            if torch.cuda.is_available():
                print(f"CUDA版本: {torch.version.cuda}")
                print(f"GPU数量: {torch.cuda.device_count()}")
                
                for i in range(torch.cuda.device_count()):
                    gpu_name = torch.cuda.get_device_name(i)
                    gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
                    print(f"  GPU {i}: {gpu_name} ({gpu_memory:.1f}GB)")
            else:
                self.issues.append("CUDA不可用")
                self.fixes.append("检查CUDA安装或使用CPU版本")
                
        except ImportError:
            self.issues.append("PyTorch未安装")
            self.fixes.append("pip install torch torchvision torchaudio")
    
    def check_file_structure(self):
        """检查文件结构"""
        print("\n" + "=" * 80)
        print("🔍 检查文件结构")
        print("=" * 80)
        
        critical_paths = [
            "config/parameters.py",
            "data/unified/tatqa_knowledge_base_combined.jsonl",
            "data/unified/tatqa_knowledge_base_unified.jsonl",
            "models/finetuned_tatqa_mixed_enhanced",
            "models/finetuned_alphafin_zh_optimized",
            "xlm/components/encoder/finbert.py",
            "xlm/components/retriever/bilingual_retriever.py",
            "xlm/utils/dual_language_loader.py"
        ]
        
        for path in critical_paths:
            if Path(path).exists():
                print(f"  ✅ {path}")
            else:
                print(f"  ❌ {path}")
                self.issues.append(f"缺少关键文件: {path}")
                self.fixes.append(f"检查文件路径: {path}")
    
    def check_data_files(self):
        """检查数据文件"""
        print("\n" + "=" * 80)
        print("🔍 检查数据文件")
        print("=" * 80)
        
        data_files = [
            "data/unified/tatqa_knowledge_base_combined.jsonl",
            "data/unified/tatqa_knowledge_base_unified.jsonl"
        ]
        
        for file_path in data_files:
            path = Path(file_path)
            if path.exists():
                size_mb = path.stat().st_size / (1024 * 1024)
                print(f"  ✅ {file_path} ({size_mb:.1f}MB)")
                
                # 检查文件内容
                try:
                    with open(path, 'r', encoding='utf-8') as f:
                        first_line = f.readline().strip()
                        if first_line:
                            data = json.loads(first_line)
                            if 'content' in data:
                                content_preview = data['content'][:50] + "..."
                                print(f"    内容预览: {content_preview}")
                            else:
                                print(f"    ⚠️ 文件格式可能不正确")
                        else:
                            print(f"    ❌ 文件为空")
                            self.issues.append(f"数据文件为空: {file_path}")
                except Exception as e:
                    print(f"    ❌ 文件读取错误: {e}")
                    self.issues.append(f"数据文件损坏: {file_path}")
            else:
                print(f"  ❌ {file_path}")
                self.issues.append(f"缺少数据文件: {file_path}")
    
    def check_model_files(self):
        """检查模型文件"""
        print("\n" + "=" * 80)
        print("🔍 检查模型文件")
        print("=" * 80)
        
        model_paths = [
            "models/finetuned_tatqa_mixed_enhanced",
            "models/finetuned_alphafin_zh_optimized"
        ]
        
        for model_path in model_paths:
            path = Path(model_path)
            if path.exists():
                # 检查关键文件
                config_file = path / "config.json"
                model_file = path / "pytorch_model.bin"
                
                if config_file.exists() and model_file.exists():
                    model_size_mb = model_file.stat().st_size / (1024 * 1024)
                    print(f"  ✅ {model_path} ({model_size_mb:.1f}MB)")
                else:
                    print(f"  ⚠️ {model_path} (文件不完整)")
                    self.issues.append(f"模型文件不完整: {model_path}")
            else:
                print(f"  ❌ {model_path}")
                self.issues.append(f"缺少模型文件: {model_path}")
                self.fixes.append(f"下载或检查模型: {model_path}")
    
    def run_quick_test(self):
        """运行快速测试"""
        print("\n" + "=" * 80)
        print("🔍 运行快速测试")
        print("=" * 80)
        
        try:
            # 测试配置加载
            from config.parameters import Config
            config = Config()
            print("  ✅ 配置加载成功")
            
            # 测试编码器导入
            from xlm.components.encoder.finbert import FinbertEncoder
            print("  ✅ 编码器导入成功")
            
            # 测试检索器导入
            from xlm.components.retriever.bilingual_retriever import BilingualRetriever
            print("  ✅ 检索器导入成功")
            
            # 测试数据加载器导入
            from xlm.utils.dual_language_loader import DualLanguageLoader
            print("  ✅ 数据加载器导入成功")
            
        except Exception as e:
            print(f"  ❌ 快速测试失败: {e}")
            self.issues.append(f"快速测试失败: {e}")
            traceback.print_exc()
    
    def generate_fix_script(self):
        """生成修复脚本"""
        if not self.fixes:
            print("\n" + "=" * 80)
            print("✅ 环境检查通过，无需修复")
            print("=" * 80)
            return
        
        print("\n" + "=" * 80)
        print("🔧 生成修复脚本")
        print("=" * 80)
        
        fix_script = "#!/bin/bash\n"
        fix_script += "# 自动生成的修复脚本\n"
        fix_script += "echo '开始修复环境问题...'\n\n"
        
        for fix in self.fixes:
            if fix.startswith("pip install"):
                fix_script += f"{fix}\n"
            elif fix.startswith("检查"):
                fix_script += f"echo '{fix}'\n"
            else:
                fix_script += f"echo '{fix}'\n"
        
        fix_script += "\necho '修复完成！'\n"
        
        with open("fix_environment.sh", "w") as f:
            f.write(fix_script)
        
        print("📝 修复脚本已生成: fix_environment.sh")
        print("💡 运行命令: bash fix_environment.sh")
    
    def run_full_check(self):
        """运行完整检查"""
        print("🚀 开始环境健康检查")
        
        self.check_python_environment()
        self.check_cuda_environment()
        self.check_file_structure()
        self.check_data_files()
        self.check_model_files()
        self.run_quick_test()
        
        # 生成报告
        print("\n" + "=" * 80)
        print("📋 检查报告")
        print("=" * 80)
        
        if self.issues:
            print(f"❌ 发现 {len(self.issues)} 个问题:")
            for i, issue in enumerate(self.issues, 1):
                print(f"  {i}. {issue}")
            
            print(f"\n🔧 建议的修复方案:")
            for i, fix in enumerate(self.fixes, 1):
                print(f"  {i}. {fix}")
            
            self.generate_fix_script()
        else:
            print("✅ 环境检查通过，没有发现问题")
        
        print("=" * 80)

if __name__ == "__main__":
    checker = EnvironmentHealthChecker()
    checker.run_full_check() 