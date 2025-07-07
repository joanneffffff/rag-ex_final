#!/usr/bin/env python3
"""
简单备份和删除脚本
"""

import shutil
from pathlib import Path
import json

def backup_and_delete():
    """备份并删除测试文件"""
    
    # 要删除的测试文件列表
    test_files_to_delete = [
        "test_15_samples_context_separation.py",
        "test_backward_compatibility.py", 
        "test_chinese_compatibility.py",
        "test_chinese_english_functions.py",
        "test_chinese_english_separation.py",
        "test_chinese_multi_stage_template.py",
        "test_chinese_multi_stage_template_cuda0.py",
        "test_chinese_retrieval_modes.py",
        "test_comprehensive_evaluation_english_only.py",
        "test_content_ratio_fix.py",
        "test_context_separation.py",
        "test_context_separation_samples.jsonl",
        "test_context_separation_simple.py",
        "test_current_faiss_logic.py",
        "test_english_assistant_fewshot.py",
        "test_english_template_detection.py",
        "test_enhanced_decision_logic.py",
        "test_faiss_fix_verification.py",
        "test_faiss_logic_fix.py",
        "test_faiss_performance_fix.py",
        "test_faiss_search_fix.py",
        "test_few_shot_assistant.py",
        "test_finbert_encoder.py",
        "test_hybrid_decision.py",
        "test_hybrid_decision_consistency.py",
        "test_hybrid_decision_english_only.py",
        "test_hybrid_decision_logic.py",
        "test_hybrid_template_integration.py",
        "test_hybrid_template_simple.py",
        "test_integration.py",
        "test_local_llm_generator_no_language_detection.py",
        "test_local_llm_generator_simple.py",
        "test_local_llm_generator_simple_role.py",
        "test_mixed_content_decision.py",
        "test_prefilter_switch.py",
        "test_raw_data_matching.py",
        "test_retrieval_modes_fix.py",
        "test_template_loading.py",
        "test_template_multiturn.py",
        "test_train_data_matching.py",
        "test_type_matching_logic.py"
    ]
    
    # 创建备份目录
    backup_dir = Path("test_backup")
    backup_dir.mkdir(exist_ok=True)
    
    deleted_files = []
    backuped_files = []
    
    print("🚀 开始备份和删除测试文件")
    print("=" * 60)
    
    for filename in test_files_to_delete:
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
    
    with open(backup_dir / "backup_info.json", "w") as f:
        json.dump(backup_info, f, indent=2)
    
    print("\n" + "=" * 60)
    print("📋 操作完成")
    print("=" * 60)
    print(f"✅ 成功删除: {len(deleted_files)} 个文件")
    print(f"📦 备份位置: {backup_dir}")
    print(f"📄 备份信息: {backup_dir}/backup_info.json")
    
    if deleted_files:
        print("\n🗑️ 已删除的文件:")
        for filename in deleted_files:
            print(f"  • {filename}")

if __name__ == "__main__":
    backup_and_delete() 