import json
from pathlib import Path
import hashlib
from collections import defaultdict

def calculate_content_hash(content: str) -> str:
    """计算内容的哈希值，用于检测重复 - 与我们的脚本保持一致"""
    return hashlib.md5(content.encode('utf-8')).hexdigest()

def generate_record_hash(record: dict) -> str:
    """为记录生成哈希值，用于去重"""
    context = record.get('original_context', record.get('context', ''))
    answer = record.get('original_answer', record.get('answer', ''))
    question = record.get('original_question', record.get('query', ''))
    
    # 使用与我们的脚本相同的哈希计算方式
    full_content = f"{context}|{answer}|{question}"
    return calculate_content_hash(full_content)

def merge_and_deduplicate_data(
    original_json_path: Path, 
    generated_json_paths: list[Path], 
    merged_output_path: Path,
    missing_records_output_path: Path
):
    """
    合并多个LLM生成的JSON文件，并进行严格的去重
    """
    print(f"正在合并生成的JSON文件：{[str(p) for p in generated_json_paths]}...")
    
    # 使用哈希值作为键进行去重
    unique_generated_records_map = {}
    duplicate_count = 0

    for gen_path in generated_json_paths:
        try:
            with open(gen_path, 'r', encoding='utf-8') as f:
                current_gen_data = json.load(f)
                print(f"处理文件 {gen_path}: {len(current_gen_data)} 条记录")
                
                for record in current_gen_data:
                    # 生成哈希值
                    record_hash = generate_record_hash(record)
                    
                    if record_hash in unique_generated_records_map:
                        duplicate_count += 1
                        print(f"发现重复记录 (哈希: {record_hash[:8]}...), 跳过")
                    else:
                        unique_generated_records_map[record_hash] = record

        except FileNotFoundError:
            print(f"警告：生成文件未找到：{gen_path}")
        except json.JSONDecodeError:
            print(f"错误：生成文件格式不正确：{gen_path}")
        except Exception as e:
            print(f"处理生成文件 {gen_path} 时发生错误: {e}")

    merged_generated_data = list(unique_generated_records_map.values())
    print(f"合并完成。总计 {len(merged_generated_data)} 条唯一记录，删除了 {duplicate_count} 条重复记录。")

    # 保存合并后的文件
    merged_output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(merged_output_path, 'w', encoding='utf-8') as f:
        json.dump(merged_generated_data, f, ensure_ascii=False, indent=2)
    print(f"合并后的生成数据已保存到：{merged_output_path}")

    # 加载原始数据
    print(f"正在加载原始文件：{original_json_path}...")
    try:
        with open(original_json_path, 'r', encoding='utf-8') as f:
            original_data = json.load(f)
        print(f"原始文件加载完成。总计 {len(original_data)} 条原始记录。")
    except Exception as e:
        print(f"加载原始文件时发生错误: {e}")
        return

    # 构建已生成记录的哈希集合
    generated_hash_set = set()
    for record in merged_generated_data:
        record_hash = generate_record_hash(record)
        generated_hash_set.add(record_hash)

    # 找出漏掉的原始记录
    missing_records = []
    for i, original_record in enumerate(original_data):
        original_hash = generate_record_hash(original_record)
        if original_hash not in generated_hash_set:
            missing_records.append(original_record)

    print(f"检查完成。发现 {len(missing_records)} 条漏掉的原始记录。")

    # 保存漏掉的原始记录
    if missing_records:
        missing_records_output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(missing_records_output_path, 'w', encoding='utf-8') as f:
            json.dump(missing_records, f, ensure_ascii=False, indent=2)
        print(f"漏掉的原始记录已保存到：{missing_records_output_path}")
    else:
        print("没有发现漏掉的原始记录。")

if __name__ == '__main__':
    # 配置文件路径
    original_data_file = Path("data/alphafin/alphafin_rag_ready_0627.json") 

    generated_files = [
        Path("data/alphafin/alphafin_summarized_and_structured_qa_0627_b8_s50_fullsentence.json"), 
        Path("data/alphafin/alphafin_summarized_and_structured_qa_0627_colab_backward.json"),
        Path("data/alphafin/alphafin_summarized_and_structured_qa_0628_colab_backward.json"),
        Path("data/alphafin/alphafin_summarized_and_structured_qa_0628_colab_missing.json"),
    ]

    merged_generated_output = Path("data/alphafin/alphafin_merged_generated_qa_improved.json")
    missing_records_output = Path("data/alphafin/alphafin_missing_original_records_improved.json")

    # 调用改进的合并函数
    merge_and_deduplicate_data(
        original_json_path=original_data_file,
        generated_json_paths=generated_files,
        merged_output_path=merged_generated_output,
        missing_records_output_path=missing_records_output
    )

    print("\n改进的合并任务完成。") 