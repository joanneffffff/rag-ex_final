#!/usr/bin/env python3
"""
英文模板优化工具
用于测试、比较和优化英文提示模板在不同场景下的效果
"""

import json
import re
from pathlib import Path
from typing import List, Dict, Any, Optional
import random
from datetime import datetime

class TemplateOptimizer:
    def __init__(self):
        self.templates = {}
        self.test_results = {}
        self.optimization_history = []
        
    def register_template(self, name: str, template_func, description: str = ""):
        """注册一个模板函数"""
        self.templates[name] = {
            "func": template_func,
            "description": description,
            "created_at": datetime.now().isoformat()
        }
    
    def generate_template(self, name: str, context: str, question: str, **kwargs) -> str:
        """生成指定模板的文本"""
        if name not in self.templates:
            raise ValueError(f"模板 '{name}' 未注册")
        
        template_func = self.templates[name]["func"]
        return template_func(context, question, **kwargs)
    
    def test_template(self, name: str, test_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """测试模板效果"""
        if name not in self.templates:
            raise ValueError(f"模板 '{name}' 未注册")
        
        results = []
        for i, data in enumerate(test_data):
            try:
                template_text = self.generate_template(
                    name, 
                    data["context"], 
                    data["question"],
                    **data.get("kwargs", {})
                )
                
                result = {
                    "sample_id": i,
                    "template_name": name,
                    "template_text": template_text,
                    "template_length": len(template_text),
                    "context_length": len(data["context"]),
                    "question_length": len(data["question"]),
                    "expected_answer": data["answer"],
                    "answer_type": data.get("answer_type", "unknown"),
                    "complexity_score": self._calculate_complexity_score(template_text),
                    "clarity_score": self._calculate_clarity_score(template_text)
                }
                results.append(result)
                
            except Exception as e:
                print(f"测试模板 '{name}' 样本 {i} 时出错: {e}")
                results.append({
                    "sample_id": i,
                    "template_name": name,
                    "error": str(e)
                })
        
        # 计算统计信息
        stats = self._calculate_template_stats(results)
        
        return {
            "template_name": name,
            "results": results,
            "statistics": stats,
            "tested_at": datetime.now().isoformat()
        }
    
    def _calculate_complexity_score(self, template_text: str) -> float:
        """计算模板复杂度分数"""
        # 基于句子数量、词汇复杂度等
        sentences = len(re.split(r'[.!?]+', template_text))
        words = len(template_text.split())
        avg_sentence_length = words / max(sentences, 1)
        
        # 复杂度分数：句子数量 + 平均句子长度
        complexity = sentences * 0.3 + avg_sentence_length * 0.7
        return min(complexity / 10, 1.0)  # 归一化到0-1
    
    def _calculate_clarity_score(self, template_text: str) -> float:
        """计算模板清晰度分数"""
        # 基于指令明确性、格式清晰度等
        clarity_indicators = [
            "please", "answer", "question", "context", "step", "instruction",
            "based on", "according to", "provide", "give", "find"
        ]
        
        score = 0
        for indicator in clarity_indicators:
            if indicator.lower() in template_text.lower():
                score += 1
        
        return min(score / len(clarity_indicators), 1.0)
    
    def _calculate_template_stats(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """计算模板统计信息"""
        valid_results = [r for r in results if "error" not in r]
        
        if not valid_results:
            return {"error": "No valid results"}
        
        lengths = [r["template_length"] for r in valid_results]
        complexity_scores = [r["complexity_score"] for r in valid_results]
        clarity_scores = [r["clarity_score"] for r in valid_results]
        
        return {
            "total_samples": len(valid_results),
            "avg_length": sum(lengths) / len(lengths),
            "min_length": min(lengths),
            "max_length": max(lengths),
            "avg_complexity": sum(complexity_scores) / len(complexity_scores),
            "avg_clarity": sum(clarity_scores) / len(clarity_scores),
            "efficiency_score": self._calculate_efficiency_score(lengths, clarity_scores)
        }
    
    def _calculate_efficiency_score(self, lengths: List[int], clarity_scores: List[float]) -> float:
        """计算效率分数（清晰度/长度）"""
        avg_length = sum(lengths) / len(lengths)
        avg_clarity = sum(clarity_scores) / len(clarity_scores)
        
        # 效率 = 清晰度 / (长度/1000) - 鼓励短而清晰的模板
        efficiency = avg_clarity / (avg_length / 1000)
        return min(efficiency, 10.0)  # 限制最大值
    
    def compare_templates(self, template_names: List[str], test_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """比较多个模板的效果"""
        comparison_results = {}
        
        for name in template_names:
            if name in self.templates:
                result = self.test_template(name, test_data)
                comparison_results[name] = result["statistics"]
        
        # 排序和排名
        sorted_templates = sorted(
            comparison_results.items(),
            key=lambda x: x[1].get("efficiency_score", 0),
            reverse=True
        )
        
        return {
            "comparison_results": comparison_results,
            "ranking": sorted_templates,
            "best_template": sorted_templates[0][0] if sorted_templates else None,
            "compared_at": datetime.now().isoformat()
        }
    
    def optimize_template(self, base_template: str, optimization_goals: List[str]) -> List[Dict[str, Any]]:
        """基于目标优化模板"""
        optimizations = []
        
        if "shorter" in optimization_goals:
            # 缩短模板
            short_variant = self._create_shorter_variant(base_template)
            optimizations.append({
                "goal": "shorter",
                "variant": short_variant,
                "description": "移除冗余词汇，保持核心指令"
            })
        
        if "clearer" in optimization_goals:
            # 提高清晰度
            clear_variant = self._create_clearer_variant(base_template)
            optimizations.append({
                "goal": "clearer",
                "variant": clear_variant,
                "description": "添加明确指令和格式要求"
            })
        
        if "more_structured" in optimization_goals:
            # 增加结构化
            structured_variant = self._create_structured_variant(base_template)
            optimizations.append({
                "goal": "more_structured",
                "variant": structured_variant,
                "description": "添加步骤和结构化格式"
            })
        
        return optimizations
    
    def _create_shorter_variant(self, template: str) -> str:
        """创建更短的变体"""
        # 移除冗余词汇
        replacements = [
            ("please answer the question", "answer"),
            ("based on the following context", "context"),
            ("please provide", "provide"),
            ("please give", "give"),
            ("as follows", ""),
            ("as shown below", ""),
        ]
        
        result = template
        for old, new in replacements:
            result = result.replace(old, new)
        
        return result.strip()
    
    def _create_clearer_variant(self, template: str) -> str:
        """创建更清晰的变体"""
        # 添加明确指令
        if "Context:" not in template:
            template = template.replace("Context", "Context:")
        if "Question:" not in template:
            template = template.replace("Question", "Question:")
        if "Answer:" not in template:
            template += "\nAnswer:"
        
        return template
    
    def _create_structured_variant(self, template: str) -> str:
        """创建更结构化的变体"""
        # 添加步骤和格式
        if "step" not in template.lower():
            template = template.replace("Answer:", """Steps:
1. Read the context carefully
2. Identify the key information
3. Answer the question directly

Answer:""")
        
        return template
    
    def save_results(self, filename: str):
        """保存测试结果"""
        # 创建可序列化的模板信息
        serializable_templates = {}
        for name, template_info in self.templates.items():
            serializable_templates[name] = {
                "description": template_info["description"],
                "created_at": template_info["created_at"]
            }
        
        data = {
            "templates": serializable_templates,
            "test_results": self.test_results,
            "optimization_history": self.optimization_history,
            "exported_at": datetime.now().isoformat()
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        print(f"✅ 结果已保存到: {filename}")

# 预定义模板函数
def basic_template(context: str, question: str, **kwargs) -> str:
    return f"""Based on the following context, please answer the question.

Context: {context}

Question: {question}

Answer:"""

def concise_template(context: str, question: str, **kwargs) -> str:
    return f"""Context: {context}
Question: {question}
Answer:"""

def instructional_template(context: str, question: str, **kwargs) -> str:
    return f"""You are a helpful AI assistant. Answer the question based on the provided context.

Context: {context}

Question: {question}

Please provide a clear and accurate answer.

Answer:"""

def step_by_step_template(context: str, question: str, **kwargs) -> str:
    return f"""Let's solve this step by step.

Context: {context}

Question: {question}

Step 1: Read and understand the context
Step 2: Identify what the question is asking for
Step 3: Find the relevant information
Step 4: Provide the answer

Answer:"""

def qa_format_template(context: str, question: str, **kwargs) -> str:
    return f"""Q: {question}

Context: {context}

A:"""

def financial_specialist_template(context: str, question: str, **kwargs) -> str:
    return f"""You are a financial analyst expert. Analyze the financial information and provide a precise answer.

Context: {context}

Question: {question}

As a financial expert, provide the answer with appropriate units and precision.

Answer:"""

def detailed_template(context: str, question: str, **kwargs) -> str:
    return f"""Please carefully analyze the following information and answer the question accurately.

Financial Context:
{context}

Question to Answer:
{question}

Instructions:
1. Read the context thoroughly
2. Identify the specific information needed
3. Provide a direct and accurate answer
4. Include units if applicable
5. If the answer is not explicitly stated, indicate that

Your Answer:"""

def main():
    print("=" * 80)
    print("英文模板优化工具")
    print("=" * 80)
    
    # 初始化优化器
    optimizer = TemplateOptimizer()
    
    # 注册模板
    optimizer.register_template("Basic", basic_template, "基础模板")
    optimizer.register_template("Concise", concise_template, "简洁模板")
    optimizer.register_template("Instructional", instructional_template, "指令式模板")
    optimizer.register_template("Step-by-Step", step_by_step_template, "分步思考模板")
    optimizer.register_template("Q&A Format", qa_format_template, "问答格式模板")
    optimizer.register_template("Financial Specialist", financial_specialist_template, "金融专家模板")
    optimizer.register_template("Detailed", detailed_template, "详细模板")
    
    print(f"✅ 注册了 {len(optimizer.templates)} 个模板")
    
    # 加载测试数据
    test_data = [
        {
            "context": "Table ID: dc9d58a4e24a74d52f719372c1a16e7f\nHeaders: Current assets | As Reported | Adjustments | Balances without Adoption of Topic 606\nReceivables, less allowance for doubtful accounts: As Reported is $831.7 million USD; Adjustments is $8.7 million USD; Balances without Adoption of Topic 606 is $840.4 million USD\nInventories : As Reported is $1,571.7 million USD; Adjustments is ($3.1 million USD); Balances without Adoption of Topic 606 is $1,568.6 million USD\nPrepaid expenses and other current assets: As Reported is $93.8 million USD; Adjustments is ($16.6 million USD); Balances without Adoption of Topic 606 is $77.2 million USD\nCategory: Current liabilities\nOther accrued liabilities: As Reported is $691.6 million USD; Adjustments is ($1.1 million USD); Balances without Adoption of Topic 606 is $690.5 million USD\nOther noncurrent liabilities : As Reported is $1,951.8 million USD; Adjustments is ($2.5 million USD); Balances without Adoption of Topic 606 is $1,949.3 million USD",
            "question": "What are the balances (without Adoption of Topic 606, in millions) of inventories and other accrued liabilities, respectively?",
            "answer": "1,568.6; 690.5",
            "answer_type": "table"
        },
        {
            "context": "We utilized a comprehensive approach to evaluate and document the impact of the guidance on our current accounting policies and practices in order to identify material differences, if any, that would result from applying the new requirements\u00a0to our revenue contracts. We did not identify any material differences resulting from applying the new requirements to our revenue contracts. In addition, we did not identify any significant changes to our business processes, systems, and controls to support recognition and disclosure requirements under the new guidance. We adopted the provisions of Topic 606 in fiscal 2019 utilizing the modified retrospective method. We recorded a $0.5 million cumulative effect adjustment, net of tax, to the opening balance of fiscal 2019 retained earnings, a decrease to receivables of $7.6 million, an increase to inventories of $2.8 million, an increase to prepaid expenses and other current assets of $6.9 million, an increase to other accrued liabilities of $1.4 million, and an increase to other noncurrent liabilities of $0.2 million. The adjustments primarily related to the timing of recognition of certain customer charges, trade promotional expenditures, and volume discounts.",
            "question": "What method did the company use when Topic 606 in fiscal 2019 was adopted?",
            "answer": "the modified retrospective method",
            "answer_type": "paragraph"
        },
        {
            "context": "Table ID: 33295076b558d53b86fd6e5537022af6\nHeaders: Years Ended\nRow 1:  is July 27, 2019; Years Ended is July 28, 2018;  is July 29, 2017;  is Variance in Dollars;  is Variance in Percent\nResearch and development:  is $ 6,577; Years Ended is $ 6,332;  is $6,059 million USD;  is $245 million USD;  is 4%\nPercentage of revenue:  is 12.7%; Years Ended is 12.8%;  is 12.6%\nSales and marketing:  is $9,571 million USD; Years Ended is $9,242 million USD;  is $9,184 million USD;  is $329 million USD;  is 4%\nPercentage of revenue:  is 18.4%; Years Ended is 18.7%;  is 19.1%\nGeneral and administrative:  is $1,827 million USD; Years Ended is $2,144 million USD;  is $1,993 million USD;  is ($317 million USD);  is (15)%\nPercentage of revenue:  is 3.5%; Years Ended is 4.3%;  is 4.2%\nTotal:  is $17,975 million USD; Years Ended is $17,718 million USD;  is $17,236 million USD;  is $257 million USD;  is 1%\nPercentage of revenue:  is 34.6%; Years Ended is 35.9%;  is 35.9%",
            "question": "Which years does the table provide information for R&D, sales and marketing, and G&A expenses?",
            "answer": "2019; 2018; 2017",
            "answer_type": "table+text"
        }
    ]
    
    print(f"✅ 加载了 {len(test_data)} 个测试样本")
    
    # 测试所有模板
    print("\n🔍 测试所有模板...")
    all_template_names = list(optimizer.templates.keys())
    
    for template_name in all_template_names:
        print(f"\n测试模板: {template_name}")
        result = optimizer.test_template(template_name, test_data)
        optimizer.test_results[template_name] = result
        
        stats = result["statistics"]
        if "error" not in stats:
            print(f"  平均长度: {stats['avg_length']:.0f} 字符")
            print(f"  平均清晰度: {stats['avg_clarity']:.3f}")
            print(f"  效率分数: {stats['efficiency_score']:.3f}")
    
    # 比较模板
    print("\n" + "=" * 80)
    print("📊 模板比较结果")
    print("=" * 80)
    
    comparison = optimizer.compare_templates(all_template_names, test_data)
    
    print("模板排名（按效率分数）:")
    for i, (name, stats) in enumerate(comparison["ranking"], 1):
        print(f"{i}. {name}:")
        print(f"   效率分数: {stats['efficiency_score']:.3f}")
        print(f"   平均长度: {stats['avg_length']:.0f} 字符")
        print(f"   平均清晰度: {stats['avg_clarity']:.3f}")
        print(f"   平均复杂度: {stats['avg_complexity']:.3f}")
    
    print(f"\n🏆 最佳模板: {comparison['best_template']}")
    
    # 模板优化示例
    print("\n" + "=" * 80)
    print("🔄 模板优化示例")
    print("=" * 80)
    
    base_template = optimizer.generate_template("Basic", test_data[0]["context"], test_data[0]["question"])
    print(f"原始模板:\n{base_template}")
    
    optimizations = optimizer.optimize_template(base_template, ["shorter", "clearer", "more_structured"])
    
    for i, opt in enumerate(optimizations, 1):
        print(f"\n优化 {i} ({opt['goal']}):")
        print(f"描述: {opt['description']}")
        print(f"优化后:\n{opt['variant']}")
    
    # 保存结果
    optimizer.save_results("template_optimization_results.json")
    
    # 生成使用建议
    print("\n" + "=" * 80)
    print("💡 使用建议")
    print("=" * 80)
    
    print("基于测试结果，推荐以下使用策略:")
    print("1. 快速问答场景: 使用 Concise 或 Q&A Format 模板")
    print("2. 专业分析场景: 使用 Financial Specialist 模板")
    print("3. 复杂推理场景: 使用 Step-by-Step 模板")
    print("4. 教学指导场景: 使用 Instructional 模板")
    
    print("\n🎯 优化建议:")
    print("- 对于简单问题，优先选择短而清晰的模板")
    print("- 对于复杂问题，使用结构化模板提高准确性")
    print("- 根据具体场景选择合适的专业模板")
    print("- 定期测试和优化模板效果")

if __name__ == "__main__":
    main() 