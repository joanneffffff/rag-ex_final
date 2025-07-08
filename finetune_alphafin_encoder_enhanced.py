#!/usr/bin/env python3
"""
AlphaFin中文编码器微调脚本 (增强版)
使用generated_question作为query，summary作为context进行微调
"""

import os
import json
import argparse
import torch
import numpy as np
from sentence_transformers import SentenceTransformer, InputExample, losses
from sentence_transformers.evaluation import SentenceEvaluator
from torch.utils.data import DataLoader
from tqdm import tqdm
import csv

class MRREvaluator(SentenceEvaluator):
    """
    对给定的 (generated_question, summary) 数据集计算 Mean Reciprocal Rank (MRR)
    使用doc_id进行正确的匹配
    """
    def __init__(self, dataset, name='', show_progress_bar=False, write_csv=True):
        self.dataset = dataset
        self.name = name
        self.show_progress_bar = show_progress_bar
        self.write_csv = write_csv

        # 添加调试信息
        if dataset and len(dataset) > 0:
            print(f"调试：第一个数据项字段: {list(dataset[0].keys())}")
            if 'query' not in dataset[0]:
                print(f"错误：数据项缺少'query'字段，可用字段: {list(dataset[0].keys())}")
                raise KeyError("数据项缺少'query'字段")

        # 确保数据字段存在
        self.queries = [item['query'] for item in dataset]
        self.contexts = [item['context'] for item in dataset]
        self.answers = [item['answer'] for item in dataset] 

        self.csv_file: str = ""
        self.csv_headers = ["epoch", "steps", "MRR"]

    def __call__(self, model, output_path: str = None, epoch: int = -1, steps: int = -1) -> float:
        if epoch != -1:
            if self.write_csv:
                self.csv_file = os.path.join(output_path, self.name + "_mrr_evaluation_results.csv")
                if not os.path.isfile(self.csv_file) or epoch == 0:
                    with open(self.csv_file, newline="", mode="w", encoding="utf-8") as f:
                        writer = csv.writer(f)
                        writer.writerow(self.csv_headers)
                        
        print(f"\n--- 开始 MRR 评估 (Epoch: {epoch}, Steps: {steps}) ---")

        if not self.dataset:
            print("警告：评估数据集为空，MRR为0。")
            mrr = 0.0
        else:
            print(f"编码 {len(self.contexts)} 个评估上下文...")
            # 编码所有上下文
            context_embeddings = model.encode(self.contexts, batch_size=64, convert_to_tensor=True,
                                              show_progress_bar=self.show_progress_bar)

            mrrs = []
            iterator = tqdm(self.dataset, desc='评估 MRR', disable=not self.show_progress_bar)
            
            # 创建doc_id到索引的映射
            doc_id_to_idx = {}
            for idx, item in enumerate(self.dataset):
                doc_id = item.get('doc_id') or str(idx)
                doc_id_to_idx[doc_id] = idx
            
            for i, item in enumerate(iterator):
                query_emb = model.encode(item['query'], convert_to_tensor=True)
                scores = torch.cosine_similarity(query_emb.unsqueeze(0), context_embeddings)[0].cpu().numpy()

                # 使用doc_id找到目标上下文的索引
                target_doc_id = item.get('doc_id') or str(i)
                target_context_idx = doc_id_to_idx.get(target_doc_id, i)

                sorted_indices = np.argsort(scores)[::-1]
                
                rank = -1
                for r, idx in enumerate(sorted_indices):
                    if idx == target_context_idx:
                        rank = r + 1
                        break
                
                if rank != -1:
                    mrr_score = 1.0 / rank
                    mrrs.append(mrr_score)
                else:
                    mrrs.append(0.0) 
            
            mrr = float(np.mean(mrrs)) if mrrs else 0.0 

        print(f"MRR (Epoch: {epoch}, Steps: {steps}): {mrr:.4f}")
        print(f"--- MRR 评估结束 ---")

        if output_path is not None and self.write_csv:
            with open(self.csv_file, newline="", mode="a", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([epoch, steps, round(mrr, 4)])

        return mrr

def load_training_data(jsonl_path, max_samples=None):
    """加载训练数据，使用generated_question作为query，summary作为context"""
    examples = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                # 使用generated_question作为query，summary作为context
                query = item.get('generated_question', item.get('query', ''))
                context = item.get('summary', item.get('context', ''))
                
                if query and context:
                    examples.append(InputExample(texts=[query, context]))
                    
                    if max_samples and len(examples) >= max_samples:
                        break
    
    print(f"加载了 {len(examples)} 个有效训练样本。")
    return examples

def load_eval_data(jsonl_path, max_samples=None):
    """加载评估数据，使用generated_question作为query，summary作为context"""
    data = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                # 使用generated_question作为query，summary作为context
                query = item.get('generated_question', item.get('query', ''))
                context = item.get('summary', item.get('context', ''))
                answer = item.get('answer', '')
                doc_id = item.get('doc_id', '')
                
                if query and context:
                    data.append({
                        'query': query,
                        'context': context,
                        'answer': answer,
                        'doc_id': doc_id
                    })
                    
                    if max_samples and len(data) >= max_samples:
                        break
    
    print(f"加载了 {len(data)} 个有效评估样本。")
    return data

def main():
    parser = argparse.ArgumentParser(description="AlphaFin中文编码器微调 (增强版)")
    parser.add_argument("--model_name", type=str, default="Langboat/mengzi-bert-base-fin",
                       help="基础模型名称")
    parser.add_argument("--train_jsonl", type=str, default="evaluate_mrr/alphafin_train_qc.jsonl",
                       help="训练数据文件")
    parser.add_argument("--eval_jsonl", type=str, default="evaluate_mrr/alphafin_eval.jsonl",
                       help="评估数据文件")
    parser.add_argument("--output_dir", type=str, default="models/alphafin_encoder_finetuned",
                       help="输出目录")
    parser.add_argument("--batch_size", type=int, default=32, help="批次大小")
    parser.add_argument("--epochs", type=int, default=5, help="训练轮数")
    parser.add_argument("--max_samples", type=int, default=None, help="最大样本数")
    parser.add_argument("--eval_steps", type=int, default=500, help="评估步数")
    
    args = parser.parse_args()
    
    print("🚀 AlphaFin中文编码器微调 (增强版)")
    print(f"📊 配置:")
    print(f"  - 基础模型: {args.model_name}")
    print(f"  - 训练数据: {args.train_jsonl}")
    print(f"  - 评估数据: {args.eval_jsonl}")
    print(f"  - 输出目录: {args.output_dir}")
    print(f"  - 批次大小: {args.batch_size}")
    print(f"  - 训练轮数: {args.epochs}")
    print(f"  - 最大样本数: {args.max_samples}")
    print(f"  - 评估步数: {args.eval_steps}")
    print(f"  - 使用字段: generated_question -> summary")
    
    # 加载训练数据
    print(f"\n📖 加载训练数据：{args.train_jsonl}")
    train_examples = load_training_data(args.train_jsonl, args.max_samples)
    if not train_examples:
        print("❌ 没有加载到有效的训练样本")
        return

    # 加载评估数据
    print(f"📖 加载评估数据：{args.eval_jsonl}")
    eval_data = load_eval_data(args.eval_jsonl, args.max_samples)
    if not eval_data:
        print("❌ 没有加载到有效的评估样本")
        evaluator = None
    else:
        evaluator = MRREvaluator(dataset=eval_data, name='mrr_eval', show_progress_bar=True)

    # 检查GPU内存
    print(f"\n🔍 检查GPU内存...")
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        gpu_memory_allocated = torch.cuda.memory_allocated(0) / 1024**3
        gpu_memory_free = gpu_memory - gpu_memory_allocated
        print(f"  GPU内存: 总计 {gpu_memory:.1f}GB, 已用 {gpu_memory_allocated:.1f}GB, 可用 {gpu_memory_free:.1f}GB")
        
        if gpu_memory_free < 2.0:
            print("⚠️  警告：GPU内存不足，建议清理内存或减少批次大小")
            print("💡 清理命令: nvidia-smi --gpu-reset")
    else:
        print("  CPU模式")
    
    # 加载模型
    print(f"\n🤖 加载模型：{args.model_name}")
    try:
        # 设置设备
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"  使用设备: {device}")
        
        # 尝试加载模型
        model = SentenceTransformer(args.model_name)
        if device == "cuda":
            model = model.to(device)
        print("✅ 模型加载成功")
        
        # 测试模型推理
        print("  测试模型推理...")
        test_text = "测试文本"
        test_embedding = model.encode(test_text, convert_to_tensor=True)
        print(f"  推理测试成功，嵌入维度: {test_embedding.shape}")
        
    except torch.cuda.OutOfMemoryError as e:
        print(f"❌ CUDA内存不足: {e}")
        print("💡 解决方案:")
        print("  1. 减少批次大小: --batch_size 4 或 8")
        print("  2. 清理GPU内存: nvidia-smi --gpu-reset")
        print("  3. 使用CPU训练: 设置 CUDA_VISIBLE_DEVICES=''")
        return
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        print("💡 可能的原因:")
        print("  1. 模型名称错误")
        print("  2. 网络连接问题")
        print("  3. 依赖包版本不兼容")
        return

    # 准备训练
    print(f"\n🎯 准备训练:")
    print(f"  - 训练样本数: {len(train_examples)}")
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=args.batch_size)
    train_loss = losses.MultipleNegativesRankingLoss(model)
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 开始训练
    print(f"\n🚀 开始训练...")
    
    # 确保evaluator不为None
    if evaluator is None:
        print("⚠️  警告：没有评估器，将跳过评估")
        model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            epochs=args.epochs,
            evaluation_steps=args.eval_steps,
            output_path=args.output_dir,
            show_progress_bar=True,
            optimizer_params={'lr': 2e-5, 'weight_decay': 0.01},
            scheduler='WarmupCosine',
            warmup_steps=100
        )
    else:
        model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            epochs=args.epochs,
            evaluator=evaluator,
            evaluation_steps=args.eval_steps,
            output_path=args.output_dir,
            show_progress_bar=True,
            optimizer_params={'lr': 2e-5, 'weight_decay': 0.01},
            scheduler='WarmupCosine',
            warmup_steps=100
        )
    
    # 保存最终模型
    model.save(args.output_dir)
    print(f"\n✅ 微调完成！模型已保存到：{args.output_dir}")

if __name__ == "__main__":
    main() 