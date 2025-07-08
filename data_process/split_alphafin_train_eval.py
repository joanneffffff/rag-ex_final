import json
import random
from pathlib import Path

def split_alphafin_json(
    input_json,
    train_jsonl,
    eval_jsonl,
    train_ratio=0.8,
    seed=42
):
    """分割AlphaFin数据为训练集和评估集"""
    print(f"📖 加载原始数据: {input_json}")
    
    with open(input_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    print(f"✅ 加载了 {len(data)} 个样本")
    
    # 随机打乱数据
    random.seed(seed)
    random.shuffle(data)
    
    # 计算分割点
    n_train = int(len(data) * train_ratio)
    train_data = data[:n_train]
    eval_data = data[n_train:]
    
    print(f"📊 分割结果:")
    print(f"  - 训练集: {len(train_data)} 个样本 ({train_ratio*100:.0f}%)")
    print(f"  - 评估集: {len(eval_data)} 个样本 ({(1-train_ratio)*100:.0f}%)")
    
    # 保存训练集（保留generated_question、summary和doc_id）
    print(f"💾 保存训练集: {train_jsonl}")
    with open(train_jsonl, "w", encoding="utf-8") as f:
        for item in train_data:
            train_item = {
                "generated_question": item.get("generated_question", item.get("question", "")),
                "summary": item.get("summary", ""),
                "doc_id": item.get("doc_id", "")
            }
            f.write(json.dumps(train_item, ensure_ascii=False) + "\n")
    
    # 保存评估集（保留完整Q-C-A）
    print(f"💾 保存评估集: {eval_jsonl}")
    with open(eval_jsonl, "w", encoding="utf-8") as f:
        for item in eval_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    
    print(f"✅ 分割完成！")
    print(f"  - 训练集: {train_jsonl} ({len(train_data)}条)")
    print(f"  - 评估集: {eval_jsonl} ({len(eval_data)}条)")

def analyze_data_distribution(train_jsonl, eval_jsonl):
    """分析训练集和评估集的数据分布"""
    print(f"\n📊 数据分布分析:")
    
    # 分析训练集
    train_samples = []
    with open(train_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            train_samples.append(json.loads(line))
    
    # 分析评估集
    eval_samples = []
    with open(eval_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            eval_samples.append(json.loads(line))
    
    print(f"训练集统计:")
    print(f"  - 样本数: {len(train_samples)}")
    print(f"  - 平均问题长度: {sum(len(s['generated_question']) for s in train_samples)/len(train_samples):.1f} 字符")
    print(f"  - 平均摘要长度: {sum(len(s['summary']) for s in train_samples)/len(train_samples):.1f} 字符")
    
    print(f"评估集统计:")
    print(f"  - 样本数: {len(eval_samples)}")
    print(f"  - 平均问题长度: {sum(len(s['question']) for s in eval_samples)/len(eval_samples):.1f} 字符")
    print(f"  - 平均上下文长度: {sum(len(s['context']) for s in eval_samples)/len(eval_samples):.1f} 字符")
    print(f"  - 平均答案长度: {sum(len(s['answer']) for s in eval_samples)/len(eval_samples):.1f} 字符")

if __name__ == "__main__":
    # 创建输出目录
    Path("evaluate_mrr").mkdir(exist_ok=True)
    
    # 分割AlphaFin数据
    split_alphafin_json(
        input_json="data/alphafin/alphafin_final_clean.json",      # AlphaFin的清理后数据
        train_jsonl="evaluate_mrr/alphafin_train_qc.jsonl",     # 输出AlphaFin训练集Q-C
        eval_jsonl="evaluate_mrr/alphafin_eval.jsonl",           # 输出AlphaFin评估集Q-C-A
        train_ratio=0.9,  # 改为9/1分割
        seed=42
    )
    
    # 分析数据分布
    analyze_data_distribution(
        "evaluate_mrr/alphafin_train_qc.jsonl",
        "evaluate_mrr/alphafin_eval.jsonl"
    ) 