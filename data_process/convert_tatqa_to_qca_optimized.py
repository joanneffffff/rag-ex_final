import json
import re
from pathlib import Path
from tqdm import tqdm
from typing import List, Dict, Any, Union

def extract_unit_from_paragraph(paragraphs: List[Union[str, Dict]]) -> str:
    """
    从段落中提取常见的单位信息（如millions/billions USD）。
    """
    for para in paragraphs:
        text = para.get("text", "") if isinstance(para, dict) else para
        # ### 优化点：增加对 "$000's" 的识别
        match = re.search(r'dollars in (millions|billions)|in (millions|billions)|\(in (millions|billions)\)|expressed in (us )?\$000\'s', text, re.IGNORECASE)
        if match:
            if match.group(0).lower().endswith("000's"):
                return "thousands of USD"
            # 捕获第一个非空的匹配组
            for i in [1, 2, 3, 4, 5]: # 对应 (millions|billions) 和 (us )?\$000\'s
                unit = match.group(i)
                if unit:
                    return unit.lower().replace('s', '') + " of USD" 
    return ""

# ### 优化点：辅助函数，将 QA scale 转换为统一的单位表达
def format_qa_scale_to_unit_info(scale_str: str) -> str:
    if not scale_str:
        return ""
    scale_str = scale_str.lower().strip()
    if scale_str in ["million", "millions"]:
        return "millions of USD"
    elif scale_str in ["billion", "billions"]:
        return "billions of USD"
    elif scale_str in ["thousand", "thousands"]:
        return "thousands of USD"
    elif scale_str in ["percent", "percentage"]:
        return "percentage" # 百分比单独处理，不带USD
    elif scale_str == "$000's": # 兼容某些原始单位
        return "thousands of USD"
    # 可以根据需要添加更多单位
    return "" # 无法识别的单位

def table_to_natural_text(table_data: Dict, caption: str = "", unit_info: str = "") -> str:
    """
    优化的表格转文本函数，处理多级表头和更自然的数值表达。
    """
    if not isinstance(table_data, dict) or "table" not in table_data:
        return ""
    
    rows = table_data["table"]
    table_uid = table_data.get("uid", "")
    
    if not rows:
        return ""
    
    lines = []
    
    if table_uid:
        lines.append(f"Table ID: {table_uid}")
    if caption:
        lines.append(f"Table Topic: {caption.strip().replace('.', '')}.")

    # ### 更健壮的多级表头识别与合并
    effective_header_rows = []
    data_start_row_idx = 0
    unit_row_info_from_table = ""

    # 遍历行，尝试区分表头和数据
    for r_idx, row in enumerate(rows):
        is_potential_header_row = True
        is_purely_empty_row = all(str(c).strip() == "" for c in row)
        
        if is_purely_empty_row: # 空行直接跳过，不计入表头或数据
            continue

        # 检查是否是单位行 (如 "(in millions)")
        unit_match = re.search(r'\(in (millions|billions|thousands)\)', str(row), re.IGNORECASE)
        if unit_match:
            unit_row_info_from_table = format_qa_scale_to_unit_info(unit_match.group(1))
            continue # 识别为单位行后，不作为表头或数据行处理

        # 检查除第一列外，是否有大量数字或类似数字的单元格，如果是，则可能是数据行
        num_cols_with_numbers = sum(1 for c in row[1:] if re.match(r'^-?\$?\s*[\d,.]+$|^\([\s\$?\d,.]+\)$', str(c).strip()))
        if num_cols_with_numbers > (len(row) - 1) / 2 and len(row) > 1: # 超过一半的列是数字，或者有实际数字
            # 如果第一列是空或像类别，并且有数字，则很可能是数据行
            if str(row[0]).strip() == "" or not re.match(r'^-?[\d,.]+$', str(row[0]).strip()):
                is_potential_header_row = False
        
        # 启发式：如果第一列是纯年份，或者行中大部分是年份，也可能是表头
        elif all(re.match(r'^\d{4}$', str(c).strip()) or str(c).strip() == "" for c in row[1:]) and len(row) > 1 and str(row[0]).strip() == "":
            is_potential_header_row = True

        if is_potential_header_row:
            effective_header_rows.append(row)
        else:
            data_start_row_idx = r_idx
            break # 找到第一个数据行，停止表头识别

    # 如果没有识别到数据行，或者所有行都被认为是表头
    if not data_start_row_idx and effective_header_rows:
        data_start_row_idx = len(effective_header_rows) # 默认数据从表头后开始

    header_rows_to_process = rows[:data_start_row_idx]
    data_rows = rows[data_start_row_idx:]

    final_headers = []
    if len(header_rows_to_process) == 1:
        final_headers = [str(h).strip() for h in header_rows_to_process[0]]
    elif len(header_rows_to_process) > 1:
        # 复杂的多级表头合并逻辑
        top_headers = [str(h).strip() for h in header_rows_to_process[0]]
        sub_headers = [str(h).strip() for h in header_rows_to_process[1]]
        
        combined_headers = []
        # 处理第一列的特殊情况，通常为空或者行名占位
        if len(top_headers) > 0 and top_headers[0].strip() == "" and len(sub_headers) > 0 and sub_headers[0].strip() == "":
            combined_headers.append("") # 保持第一列为空
            # 从第二列开始处理
            for i in range(1, max(len(top_headers), len(sub_headers))):
                top_h = top_headers[i] if i < len(top_headers) else ""
                sub_h = sub_headers[i] if i < len(sub_headers) else ""

                if sub_h.strip() != "":
                    if top_h.strip() != "" and top_h.strip() not in sub_h:
                        combined_headers.append(f"{sub_h} ({top_h.strip()})")
                    else:
                        combined_headers.append(sub_h.strip())
                else:
                    combined_headers.append(top_h.strip()) # 如果子表头为空，用父表头

        else: # 更通用的两级表头合并
            # 找到最长的行作为基准
            max_len = max(len(top_headers), len(sub_headers))
            for i in range(max_len):
                top_h = top_headers[i] if i < len(top_headers) else ""
                sub_h = sub_headers[i] if i < len(sub_headers) else ""

                if sub_h.strip() != "":
                    if top_h.strip() != "" and top_h.strip() not in sub_h:
                        # 尝试将父标题和子标题合并
                        combined_headers.append(f"{sub_h} ({top_h.strip()})")
                    else:
                        combined_headers.append(sub_h.strip())
                else:
                    combined_headers.append(top_h.strip()) # 如果子表头为空，用父表头

        final_headers = [h for h in combined_headers if h.strip() != ""] # 过滤空表头

    # 添加整体表头信息
    if final_headers:
        lines.append(f"Table columns: {', '.join(final_headers)}.")

    effective_unit_info = unit_info or unit_row_info_from_table
    if effective_unit_info:
        lines.append(f"All monetary amounts are in {effective_unit_info}.")
    
    # ### 优化点：处理数据行
    for i, row in enumerate(data_rows):
        if not row or all(str(v).strip() == "" for v in row):
            continue
            
        row_label = str(row[0]).strip().replace('.', '') if row[0] else ""
        
        # ### 优化点：识别并处理分类/章节标题行 (例如 "Transportation Solutions:")
        # 如果第一列有值，且其他列为空，且第一列不是纯数字，则认为是分类行
        # 同时，确保这个分类行不是表头行被误识别成数据行
        if row_label and all(str(v).strip() == "" for v in row[1:]) and not re.match(r'^-?[\d,.]+$', row_label):
            lines.append(f"Category: {row_label}.")
            continue

        data_points = []
        
        # 确定数据开始的列索引：通常是1（跳过行名），除非整个表格的第一列都是空，或者行名是复合头的一部分
        # 这里需要更精细判断，以适应多级表头中第一列是空白的情况
        start_col_idx = 1 # 默认跳过第一列作为行名
        if len(final_headers) > 0 and (final_headers[0].strip() == "" or final_headers[0].strip() == row_label):
            # 如果第一个最终表头是空或直接是行名，说明数据从第二列开始
            start_col_idx = 1
        else:
            # 否则，数据可能从第一列开始，并且第一个元素不是行名而是数据点
            # 这是需要根据具体表格结构微调的启发式
            start_col_idx = 0 
        
        # 遍历数据列，从 start_col_idx 开始
        for col_idx_in_row, value in enumerate(row[start_col_idx:], start=start_col_idx):
            value_str = str(value).strip()
            if not value_str:
                continue

            # 获取对应的列头
            header_for_value = ""
            if col_idx_in_row < len(final_headers):
                header_for_value = final_headers[col_idx_in_row]
            else:
                header_for_value = f"Column {col_idx_in_row+1}"
            
            header_for_value_clean = str(header_for_value).strip().replace('.', '')

            formatted_value = value_str
            # ### 优化点：改进数值格式化，移除 $ 和 ,，处理负数表达，并尝试转换为数字类型
            if re.match(r'^-?\$?\s*[\d,.]+$|^\([\s\$?\d,.]+\)$', value_str): 
                clean_value = value_str.replace('$', '').replace(',', '').strip() 
                try: 
                    if clean_value.startswith('(') and clean_value.endswith(')'):
                        val_num = float(clean_value[1:-1])
                        formatted_value = f"a negative {val_num}" if '.' in clean_value else f"a negative {int(val_num)}"
                    else:
                        val_num = float(clean_value)
                        formatted_value = str(val_num) if '.' in clean_value else str(int(val_num))
                except ValueError:
                    formatted_value = value_str 
            else:
                formatted_value = value_str

            if header_for_value_clean:
                data_points.append(f"{header_for_value_clean} is {formatted_value}")
            else: 
                data_points.append(f"Value is {formatted_value}") 

        # ### 优化点：构建行描述，更自然地组合
        if row_label and data_points:
            lines.append(f"For {row_label}: {', '.join(data_points)}.")
        elif data_points: 
            lines.append(f"Row {i+1} data: {', '.join(data_points)}.")
        elif row_label: # 只有行名，没有数据点（理论上应该被上面的分类行处理）
            lines.append(f"Item: {row_label}.") 

    return "\n".join(lines)


def process_tatqa_to_qca_enhanced(input_paths: List[Union[str, Path]], output_path: Union[str, Path]) -> None:
    """
    处理 TAT-QA 原始数据，将其转换为 QCA (Question-Context-Answer) 格式，
    并进行表格文本化和相关文档 ID 的精确匹配。
    """
    all_data = []
    for input_path in input_paths:
        with open(input_path, "r", encoding="utf-8") as f:
            all_data.extend(json.load(f))

    processed_qa_chunks = []

    for item in tqdm(all_data, desc=f"Processing {Path(output_path).name}"):
        doc_paragraphs = item.get("paragraphs", [])
        doc_tables = item.get("tables", [])
        
        if "table" in item and not doc_tables:
            doc_tables = [item["table"]]
        
        qa_pairs = item.get("qa_pairs", item.get("questions", []))

        doc_wide_unit_info = extract_unit_from_paragraph(doc_paragraphs)
        
        for qa in qa_pairs:
            question = qa.get("question", "").strip()
            answer = qa.get("answer", "")
            
            if isinstance(answer, list):
                answer_str = "; ".join(str(a) for a in answer)
            elif not isinstance(answer, str):
                answer_str = str(answer)
            else:
                answer_str = answer.strip()

            if not question or not answer_str:
                continue 

            correct_chunk_content = ""
            relevant_doc_ids_for_qa = [] 
            
            answer_type = qa.get("answer_from") 
            raw_rel_paragraphs_indices = [str(idx) for idx in qa.get("rel_paragraphs", [])]
            raw_rel_tables_uids = [str(uid) for uid in qa.get("rel_tables", [])]
            
            # ### 优化点：为当前 QA 对确定最准确的单位信息，优先使用 qa["scale"]
            qa_specific_unit_info = format_qa_scale_to_unit_info(qa.get("scale", "")) or doc_wide_unit_info
            
            # --- 优先处理表格相关内容 ---
            if answer_type in ["table-text", "table"] and doc_tables:
                found_table_content = False
                for rel_table_uid in raw_rel_tables_uids:
                    for table in doc_tables:
                        if table.get("uid") == rel_table_uid:
                            # ### 优化点：将 qa_specific_unit_info 传递给 table_to_natural_text
                            table_text = table_to_natural_text(table, table.get("caption", ""), qa_specific_unit_info)
                            if table_text.strip():
                                correct_chunk_content = table_text
                                if table.get("uid"): 
                                    relevant_doc_ids_for_qa.append(table.get("uid").replace('-', '').lower())
                                found_table_content = True
                                break
                    if found_table_content:
                        break
                
                if not found_table_content and doc_tables:
                    for table in doc_tables:
                        # ### 优化点：将 qa_specific_unit_info 传递给 table_to_natural_text
                        table_text = table_to_natural_text(table, table.get("caption", ""), qa_specific_unit_info)
                        if table_text.strip():
                            correct_chunk_content = table_text
                            if table.get("uid"): 
                                relevant_doc_ids_for_qa.append(table.get("uid").replace('-', '').lower())
                            break 

            # --- 处理段落相关内容 ---
            if answer_type in ["text", "table-text"] and doc_paragraphs and raw_rel_paragraphs_indices:
                paragraph_contents = []
                for p_idx_str in raw_rel_paragraphs_indices:
                    try:
                        p_idx = int(p_idx_str) - 1 
                        if 0 <= p_idx < len(doc_paragraphs):
                            para_dict = doc_paragraphs[p_idx]
                            para_text = para_dict.get("text", "")
                            para_uid = para_dict.get("uid")
                            
                            if para_text.strip():
                                if para_uid:
                                    paragraph_contents.append(f"Paragraph ID: {para_uid}\n{para_text}")
                                    relevant_doc_ids_for_qa.append(para_uid.replace('-', '').lower())
                                else:
                                    paragraph_contents.append(para_text)
                    except (ValueError, IndexError):
                        continue
                
                if paragraph_contents:
                    if correct_chunk_content: 
                        correct_chunk_content += "\n\n" + "\n\n".join(paragraph_contents)
                    else: 
                        correct_chunk_content = "\n\n".join(paragraph_contents)
            
            # --- 构建最终的 QCA 样本 ---
            if correct_chunk_content.strip() and relevant_doc_ids_for_qa:
                main_doc_id = relevant_doc_ids_for_qa[0] 
                
                processed_qa_chunks.append({
                    "query": question,
                    "context": correct_chunk_content.strip(), 
                    "answer": answer_str,
                    "doc_id": main_doc_id, 
                    "relevant_doc_ids": list(set(relevant_doc_ids_for_qa)), 
                    "answer_from": answer_type
                })
            else:
                pass


    with open(output_path, "w", encoding="utf-8") as fout:
        for item in processed_qa_chunks:
            fout.write(json.dumps(item, ensure_ascii=False) + "\n")
    
    print(f"Generated enhanced Q-C-A data (total {len(processed_qa_chunks)} pairs): {output_path}")

if __name__ == "__main__":
    base_raw_data_path = "data/tatqa_dataset_raw/"
    base_output_path = "evaluate_mrr/" 

    Path(base_output_path).mkdir(parents=True, exist_ok=True)

    train_dev_inputs = [
        Path(base_raw_data_path) / "tatqa_dataset_train.json",
        Path(base_raw_data_path) / "tatqa_dataset_dev.json"
    ]
    
    process_tatqa_to_qca_enhanced(
        input_paths=train_dev_inputs,
        output_path=Path(base_output_path) / "tatqa_train_qc_enhanced_optimized.jsonl" 
    )
    
    eval_inputs = [
        Path(base_raw_data_path) / "tatqa_dataset_test_gold.json"
    ]
    
    process_tatqa_to_qca_enhanced(
        input_paths=eval_inputs,
        output_path=Path(base_output_path) / "tatqa_eval_enhanced_optimized.jsonl" 
    ) 
    
    print(f"\nProcessing complete. Check your '{base_output_path}' directory for optimized files.")