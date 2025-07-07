#!/usr/bin/env python3
"""
删除剩余测试文件脚本
将文件备份到test_backup目录后删除
"""

import shutil
from pathlib import Path
import json

def delete_remaining_tests():
    """删除剩余的测试文件"""
    
    # 要删除的测试文件列表
    files_to_delete = [
        # 功能修复测试（如果功能已稳定）
        "test_content_priority_fix.py",
        "test_doc_id_mapping.py", 
        "test_reranker_mapping_fix.py",
        "test_updated_preview_length.py",
        
        # UI相关测试（如果UI已稳定）
        "test_ui_content_verification.py",
        "test_ui_summary_context.py",
        "test_read_more_functionality.py",
        
        # 实验性测试
        "test_clean.py",
        "test_gpu_config.py",
        "test_smart_content_selection.py",
        "test_summary_context_integration.py",
        
        # 其他可能不需要的测试
        "test_english_template_detailed.py",
        "test_english_template_integrator.py",
        "test_english_template_multi.py",
        "test_english_encoder.py"
    ]
    
    # 备份目录
    backup_dir = Path("test_backup")
    backup_dir.mkdir(exist_ok=True)
    
    deleted_files = []
    backuped_files = []
    
    print("🚀 开始备份和删除剩余测试文件")
    print("=" * 60)
    
    for filename in files_to_delete:
        file_path = Path(filename)
        
        if file_path.exists():
            # 备份文件
            backup_path = backup_dir / filename
            shutil.copy2(file_path, backup_path)
            backuped_files.append(filename)
            print(f"📦 备份: {filename}")
            
            # 删除文件
            file_path.unlink()
            deleted_files.append(filename)
            print(f"🗑️ 删除: {filename}")
        else:
            print(f"⚠️ 文件不存在: {filename}")
    
    # 创建备份信息
    backup_info = {
        "backup_time": str(Path.cwd()),
        "deleted_files": deleted_files,
        "backuped_files": backuped_files,
        "total_deleted": len(deleted_files)
    }
    
    with open(backup_dir / "backup_info_remaining.json", "w") as f:
        json.dump(backup_info, f, indent=2)
    
    print("\n" + "=" * 60)
    print("📋 操作完成")
    print("=" * 60)
    print(f"✅ 成功删除: {len(deleted_files)} 个文件")
    print(f"📦 备份位置: {backup_dir}")
    print(f"📄 备份信息: {backup_dir}/backup_info_remaining.json")
    
    if deleted_files:
        print("\n🗑️ 已删除的文件:")
        for filename in deleted_files:
            print(f"  • {filename}")
    
    # 显示保留的核心测试文件
    print("\n✅ 保留的核心测试文件:")
    core_tests = [
        "test_complete_english_loading.py",
        "test_bilingual_retriever.py", 
        "test_english_embedding_issue.py",
        "test_data_loading.py"
    ]
    
    for test_file in core_tests:
        if Path(test_file).exists():
            print(f"  • {test_file}")

if __name__ == "__main__":
    delete_remaining_tests() 