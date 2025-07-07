#!/usr/bin/env python3
"""
测试文件清理脚本
安全删除重复和过时的测试文件
"""

import os
import shutil
from pathlib import Path
import json

class TestFileCleaner:
    def __init__(self):
        self.project_root = Path.cwd()
        self.backup_dir = self.project_root / "test_backup"
        self.deleted_files = []
        
    def analyze_test_files(self):
        """分析测试文件"""
        print("🔍 分析测试文件...")
        
        test_files = []
        for file_path in self.project_root.glob("test_*.py"):
            test_files.append(file_path)
        
        print(f"📊 发现 {len(test_files)} 个测试文件")
        
        # 按功能分类
        categories = {
            "core": [],      # 核心功能测试
            "bilingual": [], # 双语系统测试
            "template": [],  # 模板测试
            "decision": [],  # 决策逻辑测试
            "context": [],   # 上下文分离测试
            "faiss": [],     # FAISS相关测试
            "duplicate": [], # 重复测试
            "obsolete": []   # 过时测试
        }
        
        for file_path in test_files:
            filename = file_path.name
            
            # 核心功能测试
            if any(keyword in filename for keyword in [
                "complete_english_loading", "integration", "comprehensive_evaluation"
            ]):
                categories["core"].append(file_path)
            
            # 双语系统测试
            elif any(keyword in filename for keyword in [
                "chinese_english", "bilingual", "mixed_content"
            ]):
                categories["bilingual"].append(file_path)
            
            # 模板测试
            elif any(keyword in filename for keyword in [
                "template", "prompt", "assistant"
            ]):
                categories["template"].append(file_path)
            
            # 决策逻辑测试
            elif any(keyword in filename for keyword in [
                "decision", "hybrid", "enhanced"
            ]):
                categories["decision"].append(file_path)
            
            # 上下文分离测试
            elif any(keyword in filename for keyword in [
                "context_separation", "samples"
            ]):
                categories["context"].append(file_path)
            
            # FAISS相关测试
            elif any(keyword in filename for keyword in [
                "faiss", "retrieval", "search"
            ]):
                categories["faiss"].append(file_path)
            
            # 重复测试（简单版本）
            elif any(keyword in filename for keyword in [
                "simple", "basic", "old"
            ]):
                categories["duplicate"].append(file_path)
            
            # 过时测试
            elif any(keyword in filename for keyword in [
                "backward", "compatibility", "raw_data", "train_data"
            ]):
                categories["obsolete"].append(file_path)
            
            else:
                categories["core"].append(file_path)
        
        return categories
    
    def show_analysis(self, categories):
        """显示分析结果"""
        print("\n" + "=" * 80)
        print("📋 测试文件分析结果")
        print("=" * 80)
        
        for category, files in categories.items():
            if files:
                print(f"\n📁 {category.upper()} ({len(files)} 个文件):")
                for file_path in files:
                    print(f"  • {file_path.name}")
    
    def get_user_choice(self, categories):
        """获取用户选择"""
        print("\n" + "=" * 80)
        print("🗑️ 选择要删除的测试文件类别")
        print("=" * 80)
        
        print("\n可删除的类别:")
        print("1. duplicate - 重复测试文件")
        print("2. obsolete - 过时测试文件")
        print("3. faiss - FAISS相关测试（如果FAISS已修复）")
        print("4. context - 上下文分离测试（如果功能稳定）")
        print("5. template - 模板测试（如果模板稳定）")
        
        print("\n建议保留的类别:")
        print("• core - 核心功能测试")
        print("• bilingual - 双语系统测试")
        print("• decision - 决策逻辑测试")
        
        choice = input("\n请输入要删除的类别编号（用逗号分隔，如：1,2,3）: ").strip()
        
        if not choice:
            return []
        
        try:
            indices = [int(x.strip()) - 1 for x in choice.split(",")]
            category_names = ["duplicate", "obsolete", "faiss", "context", "template"]
            selected_categories = [category_names[i] for i in indices if 0 <= i < len(category_names)]
            return selected_categories
        except (ValueError, IndexError):
            print("❌ 输入格式错误")
            return []
    
    def backup_files(self, files_to_delete):
        """备份要删除的文件"""
        if not files_to_delete:
            return
        
        print(f"\n📦 备份 {len(files_to_delete)} 个文件...")
        
        # 创建备份目录
        self.backup_dir.mkdir(exist_ok=True)
        
        for file_path in files_to_delete:
            backup_path = self.backup_dir / file_path.name
            shutil.copy2(file_path, backup_path)
            print(f"  ✅ 备份: {file_path.name}")
        
        # 创建备份信息文件
        backup_info = {
            "backup_time": str(Path.cwd()),
            "deleted_files": [str(f) for f in files_to_delete],
            "total_files": len(files_to_delete)
        }
        
        with open(self.backup_dir / "backup_info.json", "w") as f:
            json.dump(backup_info, f, indent=2)
        
        print(f"📁 备份完成，位置: {self.backup_dir}")
    
    def delete_files(self, files_to_delete):
        """删除文件"""
        if not files_to_delete:
            return
        
        print(f"\n🗑️ 删除 {len(files_to_delete)} 个文件...")
        
        for file_path in files_to_delete:
            try:
                file_path.unlink()
                self.deleted_files.append(str(file_path))
                print(f"  ✅ 删除: {file_path.name}")
            except Exception as e:
                print(f"  ❌ 删除失败 {file_path.name}: {e}")
    
    def cleanup(self):
        """执行清理"""
        print("🚀 开始测试文件清理")
        
        # 分析文件
        categories = self.analyze_test_files()
        self.show_analysis(categories)
        
        # 获取用户选择
        selected_categories = self.get_user_choice(categories)
        
        if not selected_categories:
            print("❌ 未选择任何类别，取消清理")
            return
        
        # 收集要删除的文件
        files_to_delete = []
        for category in selected_categories:
            if category in categories:
                files_to_delete.extend(categories[category])
        
        if not files_to_delete:
            print("❌ 没有找到要删除的文件")
            return
        
        # 确认删除
        print(f"\n⚠️ 将要删除 {len(files_to_delete)} 个文件:")
        for file_path in files_to_delete:
            print(f"  • {file_path.name}")
        
        confirm = input("\n确认删除？(y/N): ").strip().lower()
        if confirm != 'y':
            print("❌ 取消删除")
            return
        
        # 备份文件
        self.backup_files(files_to_delete)
        
        # 删除文件
        self.delete_files(files_to_delete)
        
        # 生成清理报告
        self.generate_report()
    
    def generate_report(self):
        """生成清理报告"""
        print("\n" + "=" * 80)
        print("📋 清理报告")
        print("=" * 80)
        
        print(f"✅ 成功删除 {len(self.deleted_files)} 个文件")
        print(f"📦 备份位置: {self.backup_dir}")
        
        if self.deleted_files:
            print("\n🗑️ 已删除的文件:")
            for file_path in self.deleted_files:
                print(f"  • {Path(file_path).name}")
        
        print("\n💡 建议:")
        print("• 保留核心功能测试文件")
        print("• 定期运行重要测试")
        print("• 如需恢复文件，查看备份目录")

if __name__ == "__main__":
    cleaner = TestFileCleaner()
    cleaner.cleanup() 