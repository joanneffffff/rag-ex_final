#!/usr/bin/env python3
"""
优化RAG英文模板
简化Few-Shot示例，减少Token使用，提高答案质量
"""

def create_optimized_rag_template() -> str:
    """创建优化的RAG模板，简化Few-Shot示例"""
    
    optimized_template = """===SYSTEM===
You are a world-class quantitative financial analyst AI. Your mission is to solve complex financial questions with extreme precision, based on a given context that may include both tables and text.

### Core Directives

1. **Reasoning Process (Internal Thought)**: For every question, you MUST first perform a step-by-step reasoning process. Break down the question, identify necessary data from the table and text, formulate the calculation, and derive the solution. This is your internal monologue.
2. **Final Output (Your Public Answer)**: Your final, visible output MUST BE the answer ONLY. It should be stripped of all reasoning, explanations, units (unless asked), and introductory phrases.
3. **Output Format**:
   * For numerical or list-based answers, separate items with a semicolon and a space (e.g., `Value1; Value2`).
   * For text-based answers, provide only the minimal, essential phrase.
   * If the answer is impossible to find, state exactly: `The answer cannot be found in the provided context.`

### Simplified Examples

Q: What was the percentage increase in capital expenditure from 2018 to 2019?
Context: Table: Capital expenditures 1: 2019 is $2,807; 2018 is $2,790.
Thought: Extract values: 2019=2807, 2018=2790. Calculate: (2807-2790)/2790*100 = 0.61%
A: 0.61%

Q: What was the adjusted operating income, excluding restructuring charges?
Context: Operating Income: $500M; Restructuring Charges: $20M.
Thought: Find operating income (500) and restructuring charges (20). Adjusted = 500 + 20 = 520
A: 520

Q: What is the total R&D and G&A expense for 2019?
Context: R&D: $6,577; G&A: $1,827.
Thought: Extract R&D (6577) and G&A (1827). Total = 6577 + 1827 = 8404
A: 8404

===USER===
Context:
{context}

Question:
{question}

A:"""

    return optimized_template

def create_minimal_rag_template() -> str:
    """创建极简RAG模板，只保留核心指令"""
    
    minimal_template = """===SYSTEM===
You are a financial analyst AI. Answer questions based on the provided context.

**Instructions:**
1. Think step by step to find the answer
2. Output ONLY the final answer, no explanations
3. For multiple values, separate with semicolon and space
4. If answer not found, say "The answer cannot be found in the provided context"

**Examples:**
Q: What is the percentage change from 2018 to 2019?
Context: 2019: $2,807; 2018: $2,790
A: 0.61%

Q: What are the values for A and B?
Context: A: $500; B: $300
A: 500; 300

===USER===
Context:
{context}

Question:
{question}

A:"""

    return minimal_template

def create_focused_rag_template() -> str:
    """创建专注型RAG模板，强调答案格式"""
    
    focused_template = """===SYSTEM===
You are a precise financial analyst. Extract exact answers from the context.

**CRITICAL RULES:**
- Think internally, output only the answer
- No explanations, no units unless asked
- Multiple values: separate with "; "
- If not found: "The answer cannot be found in the provided context"

**Example:**
Q: What are the balances for inventories and liabilities?
Context: Inventories: $1,568.6; Liabilities: $690.5
A: 1568.6; 690.5

===USER===
Context:
{context}

Question:
{question}

A:"""

    return focused_template

def save_optimized_templates():
    """保存优化的模板文件"""
    
    # 创建优化版本
    optimized = create_optimized_rag_template()
    with open("data/prompt_templates/rag_english_template_optimized.txt", "w", encoding="utf-8") as f:
        f.write(optimized)
    
    # 创建极简版本
    minimal = create_minimal_rag_template()
    with open("data/prompt_templates/rag_english_template_minimal.txt", "w", encoding="utf-8") as f:
        f.write(minimal)
    
    # 创建专注版本
    focused = create_focused_rag_template()
    with open("data/prompt_templates/rag_english_template_focused.txt", "w", encoding="utf-8") as f:
        f.write(focused)
    
    print("✅ 已创建3个优化版本:")
    print("  1. rag_english_template_optimized.txt - 简化Few-Shot")
    print("  2. rag_english_template_minimal.txt - 极简版本")
    print("  3. rag_english_template_focused.txt - 专注格式")

def estimate_token_count(text: str) -> int:
    """估算Token数量"""
    return len(text) // 4

def compare_templates():
    """比较不同模板的Token使用量"""
    
    templates = {
        "Original": open("data/prompt_templates/rag_english_template.txt", "r").read(),
        "Optimized": create_optimized_rag_template(),
        "Minimal": create_minimal_rag_template(),
        "Focused": create_focused_rag_template()
    }
    
    print("\n📊 模板Token使用量对比:")
    print("="*50)
    
    for name, template in templates.items():
        tokens = estimate_token_count(template)
        print(f"{name:12}: {tokens:4d} tokens ({len(template):4d} chars)")
    
    print("\n💡 建议:")
    print("- 使用Minimal版本进行快速测试")
    print("- 使用Focused版本强调答案格式")
    print("- 使用Optimized版本平衡质量和效率")

if __name__ == "__main__":
    print("🔧 RAG模板优化工具")
    print("="*40)
    
    save_optimized_templates()
    compare_templates()
    
    print("\n🎯 下一步建议:")
    print("1. 测试minimal版本，看是否能解决格式违规问题")
    print("2. 如果仍有问题，使用focused版本强调答案格式")
    print("3. 根据测试结果选择最佳版本") 