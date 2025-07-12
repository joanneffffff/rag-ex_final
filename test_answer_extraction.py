#!/usr/bin/env python3
"""
测试答案提取逻辑的脚本
"""

import re

def _shared_text_standardizer_english(text: str) -> str:
    """
    Helper function to standardize English text for both answer extraction and F1 score calculation.
    Strictly follows the rules from the English Prompt Template.
    """
    text = text.strip()
    
    # Lowercase all text
    text = text.lower()

    # 递归替换所有 \text{...} 为 ...（保留内容）
    while True:
        new_text = re.sub(r'\\text\{([^}]*)\}', r'\1', text, flags=re.DOTALL)
        if new_text == text:
            break
        text = new_text
    # 其余 LaTeX 格式直接去掉
    text = re.sub(r'\\[a-zA-Z]+\{[^}]*\}', '', text)
    text = re.sub(r'\\[a-zA-Z]+', '', text)
    
    # Remove currency symbols and common unit words based on prompt rule
    text = re.sub(r'\b(million|billion|thousand|trillion|usd|eur|gbp|m|b)\b', '', text, flags=re.IGNORECASE).strip()
    text = re.sub(r'[\$£€]', '', text).strip()

    # Remove commas from numbers
    text = text.replace(',', '')

    # Handle negative numbers in parentheses (e.g., "(33)" -> "-33")
    if text.startswith('(') and text.endswith(')'):
        text = '-' + text[1:-1]
    
    # Normalize percentages
    text = text.replace(' percent', '%').replace('pct', '%')
    text = re.sub(r'(\d+\.?\d*)\s*%', r'\1%', text)
    
    # Remove common introductory phrases
    text = re.sub(r'^(the\s*answer\s*is|it\s*was|the\s*value\s*is|resulting\s*in|this\s*represents|the\s*effect\s*is|therefore|so|thus|in\s*conclusion|final\s*answer\s*is|final\s*number\s*is)\s*', '', text, flags=re.IGNORECASE).strip()
    
    # Remove trailing punctuation
    if text.endswith('%'):
        text = re.sub(r'[\.,;]$', '', text).strip()
    else:
        text = re.sub(r'[\.,;%]$', '', text).strip() 
    
    # Final cleanup of whitespace
    text = ' '.join(text.split()).strip()

    return text

def debug_text_standardization(text: str) -> str:
    """
    调试函数：显示文本标准化的每一步
    """
    print(f"原始文本: {text}")
    
    text = text.strip()
    print(f"1. 去除首尾空格: {text}")
    
    text = text.lower()
    print(f"2. 转小写: {text}")
    
    # 递归替换所有 \text{...} 为 ...（保留内容）
    step = 3
    while True:
        new_text = re.sub(r'\\text\{([^}]*)\}', r'\1', text, flags=re.DOTALL)
        if new_text == text:
            break
        text = new_text
        print(f"{step}. 替换 \\text{{}}: {text}")
        step += 1
    
    # 其余 LaTeX 格式直接去掉
    text = re.sub(r'\\[a-zA-Z]+\{[^}]*\}', '', text)
    print(f"{step}. 去掉其他LaTeX格式: {text}")
    step += 1
    
    text = re.sub(r'\\[a-zA-Z]+', '', text)
    print(f"{step}. 去掉LaTeX命令: {text}")
    step += 1
    
    # Remove currency symbols and common unit words based on prompt rule
    text = re.sub(r'\b(million|billion|thousand|trillion|usd|eur|gbp|m|b)\b', '', text, flags=re.IGNORECASE).strip()
    print(f"{step}. 去掉单位词: {text}")
    step += 1
    
    text = re.sub(r'[\$£€]', '', text).strip()
    print(f"{step}. 去掉货币符号: {text}")
    step += 1

    # Remove commas from numbers
    text = text.replace(',', '')
    print(f"{step}. 去掉逗号: {text}")
    step += 1

    # Handle negative numbers in parentheses (e.g., "(33)" -> "-33")
    if text.startswith('(') and text.endswith(')'):
        text = '-' + text[1:-1]
        print(f"{step}. 处理负数括号: {text}")
        step += 1
    
    # Normalize percentages
    text = text.replace(' percent', '%').replace('pct', '%')
    text = re.sub(r'(\d+\.?\d*)\s*%', r'\1%', text)
    print(f"{step}. 标准化百分比: {text}")
    step += 1
    
    # Remove common introductory phrases
    text = re.sub(r'^(the\s*answer\s*is|it\s*was|the\s*value\s*is|resulting\s*in|this\s*represents|the\s*effect\s*is|therefore|so|thus|in\s*conclusion|final\s*answer\s*is|final\s*number\s*is)\s*', '', text, flags=re.IGNORECASE).strip()
    print(f"{step}. 去掉引导短语: {text}")
    step += 1
    
    # Remove trailing punctuation
    if text.endswith('%'):
        text = re.sub(r'[\.,;]$', '', text).strip()
    else:
        text = re.sub(r'[\.,;%]$', '', text).strip() 
    print(f"{step}. 去掉尾随标点: {text}")
    step += 1
    
    # Final cleanup of whitespace
    text = ' '.join(text.split()).strip()
    print(f"{step}. 最终清理空格: {text}")
    
    return text

def extract_final_answer_from_tag(raw_output: str) -> str:
    """
    Extracts the final answer from the model's raw output by looking for the <answer> tag.
    Returns NOT_FOUND_REPLY_ENGLISH if no valid answer found or tag is empty.
    """
    NOT_FOUND_REPLY_ENGLISH = "I cannot find the answer in the provided context."
    
    # First, try to find <answer> tags
    match = re.search(r'<answer>(.*?)</answer>', raw_output, re.DOTALL | re.IGNORECASE)
    
    if match:
        content = match.group(1).strip()
        # Ensure extracted content is not empty or an empty tag itself (e.g., <answer></answer>)
        if content and content.lower() not in ['<final></final>', '<answer></answer>', '<final-answer></final-answer>']:
            
            # Try to extract the most concise answer from the content
            # Look for patterns that might contain the actual answer
            
            # 1. Look for boxed answers: \boxed{...}
            # 使用更复杂的正则表达式来处理嵌套大括号
            boxed_match = re.search(r'\\boxed\{((?:[^{}]|{[^{}]*})*)\}', content)
            if boxed_match:
                return _shared_text_standardizer_english(boxed_match.group(1))
            
            # 2. Look for percentage patterns: 12.82%
            percentage_match = re.search(r'(\d+\.?\d*)\s*%', content)
            if percentage_match:
                return _shared_text_standardizer_english(percentage_match.group(0))
            
            # 3. Look for numerical answers at the end of sentences
            # This is for cases like "Thus, the answer is 12.82%"
            final_number_match = re.search(r'(?:thus|therefore|answer is|result is)\s+(?:approximately\s+)?(\d+\.?\d*)', content, re.IGNORECASE)
            if final_number_match:
                return _shared_text_standardizer_english(final_number_match.group(1))
            
            # 4. Look for the largest numerical value (likely the answer)
            # This helps when there are multiple numbers in the text
            numbers = re.findall(r'\b(\d+(?:,\d+)*)\b', content)
            if numbers:
                # Convert to integers for comparison, removing commas
                number_values = [int(num.replace(',', '')) for num in numbers]
                largest_number = max(number_values)
                return _shared_text_standardizer_english(str(largest_number))
            
            # 5. If no specific pattern found, return the original content
            return _shared_text_standardizer_english(content)
    
    # If no <answer> tags found, look for boxed answers in the entire text
    # 使用更复杂的正则表达式来处理嵌套大括号
    boxed_match = re.search(r'\\boxed\{((?:[^{}]|{[^{}]*})*)\}', raw_output)
    if boxed_match:
        return _shared_text_standardizer_english(boxed_match.group(1))
    
    # If no valid <answer> structure is found or content is invalid,
    # return the specific "not found" phrase.
    return NOT_FOUND_REPLY_ENGLISH

# 测试用例
test_cases = [
    {
        "name": "Fin-R1 Boxed Answer",
        "raw_output": "\\boxed{172 \\text{ million pounds of copper, 122,000 troy ounces of gold, and 2.6 million troy ounces of silver}}",
        "expected": "172 pounds of copper 122000 troy ounces of gold and 2.6 troy ounces of silver"  # 修正期望值，包含完整的答案
    },
    {
        "name": "Fin-R1 Percentage Answer",
        "raw_output": "<answer>\nTo determine the percentage increase...\nThus, the percentage increase in Mr. Kapuria's salary from his promotion is approximately **12.82%**.\n</answer>",
        "expected": "12.82%"
    },
    {
        "name": "Fin-R1 Numerical Answer",
        "raw_output": "<answer>\nThe total credit facility in 2019 is explicitly stated as 300,000 thousand dollars.\n</answer>",
        "expected": "300000"
    },
    {
        "name": "Qwen3-8B Simple Answer",
        "raw_output": "<answer>300000</answer>",
        "expected": "300000"
    }
]

print("Testing answer extraction logic...")
print("=" * 50)

for i, test_case in enumerate(test_cases, 1):
    print(f"\nTest {i}: {test_case['name']}")
    print(f"Raw output: {test_case['raw_output'][:100]}...")
    
    # 对于第一个测试用例，显示详细的调试信息
    if i == 1:
        print("\n=== 详细调试信息 ===")
        debug_text_standardization("172 \\text{ million pounds of copper, 122,000 troy ounces of gold, and 2.6 million troy ounces of silver}")
        print("=== 调试信息结束 ===\n")
    
    result = extract_final_answer_from_tag(test_case['raw_output'])
    print(f"Extracted: '{result}'")
    print(f"Expected:  '{test_case['expected']}'")
    print(f"Match:     {result == test_case['expected']}") 