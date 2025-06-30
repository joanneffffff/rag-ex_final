#!/usr/bin/env python3
"""
测试上下文优化在不同类型查询中的通用性
"""

import sys
import os
import re
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def extract_keywords_general(query: str, domain: str = "general") -> list:
    """
    通用关键词提取函数，支持不同领域
    
    Args:
        query: 查询文本
        domain: 领域类型 ("financial", "technical", "general")
    
    Returns:
        关键词列表
    """
    keywords = []
    
    if domain == "financial":
        # 金融领域关键词
        # 提取股票代码
        stock_pattern = r'[A-Z]{2}\d{4}|[A-Z]{2}\d{6}|\d{6}'
        stock_matches = re.findall(stock_pattern, query)
        keywords.extend(stock_matches)
        
        # 提取公司名称
        company_pattern = r'([A-Za-z\u4e00-\u9fff]+)(?:公司|集团|股份|有限)'
        company_matches = re.findall(company_pattern, query)
        keywords.extend(company_matches)
        
        # 提取年份
        year_pattern = r'20\d{2}年'
        year_matches = re.findall(year_pattern, query)
        keywords.extend(year_matches)
        
        # 金融关键概念
        key_concepts = ['利润', '营收', '增长', '业绩', '预测', '原因', '主要', '持续', '股价', '市值', '财务', '报告']
        
    elif domain == "technical":
        # 技术领域关键词
        # 提取技术术语
        tech_pattern = r'[A-Z][a-z]+(?:[A-Z][a-z]+)*'  # 驼峰命名
        tech_matches = re.findall(tech_pattern, query)
        keywords.extend(tech_matches)
        
        # 提取版本号
        version_pattern = r'\d+\.\d+(?:\.\d+)?'
        version_matches = re.findall(version_pattern, query)
        keywords.extend(version_matches)
        
        # 技术关键概念
        key_concepts = ['性能', '优化', '算法', '架构', '系统', '开发', '测试', '部署', '安全', '效率']
        
    else:
        # 通用领域关键词
        # 提取数字
        number_pattern = r'\d+'
        number_matches = re.findall(number_pattern, query)
        keywords.extend(number_matches)
        
        # 提取英文单词
        english_pattern = r'[A-Za-z]+'
        english_matches = re.findall(english_pattern, query)
        keywords.extend(english_matches)
        
        # 通用关键概念
        key_concepts = ['如何', '什么', '为什么', '怎么', '方法', '步骤', '原因', '结果', '影响', '建议']
    
    # 添加领域特定的关键概念
    for concept in key_concepts:
        if concept in query:
            keywords.append(concept)
    
    return list(set(keywords))

def extract_relevant_sentences_general(content: str, keywords: list, max_chars_per_doc: int = 800) -> list:
    """
    通用句子提取函数
    
    Args:
        content: 文档内容
        keywords: 关键词列表
        max_chars_per_doc: 每个文档最大字符数
    
    Returns:
        相关句子列表
    """
    if not content or not keywords:
        return []
    
    # 按句子分割（支持中英文）
    sentences = re.split(r'[。！？\n\.\!\?]+', content)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    # 计算每个句子的相关性分数
    sentence_scores = []
    for sentence in sentences:
        score = 0
        for keyword in keywords:
            if keyword.lower() in sentence.lower():  # 不区分大小写
                score += 1
        # 考虑句子长度，避免过长的句子
        if len(sentence) > 200:
            score *= 0.5
        sentence_scores.append((sentence, score))
    
    # 按分数排序
    sentence_scores.sort(key=lambda x: x[1], reverse=True)
    
    # 选择最相关的句子
    selected_sentences = []
    total_chars = 0
    
    for sentence, score in sentence_scores:
        if score > 0 and total_chars + len(sentence) <= max_chars_per_doc:
            selected_sentences.append(sentence)
            total_chars += len(sentence)
    
    return selected_sentences

def test_different_domains():
    """测试不同领域的查询优化效果"""
    
    print("🧪 测试上下文优化在不同领域的通用性")
    print("=" * 60)
    
    # 测试查询集合
    test_queries = {
        "financial": [
            "德赛电池（000049）2021年利润持续增长的主要原因是什么？",
            "000049的业绩表现如何？",
            "德赛电池的财务数据怎么样？"
        ],
        "technical": [
            "如何优化Python代码的性能？",
            "Docker容器化部署的最佳实践是什么？",
            "机器学习模型的训练过程包括哪些步骤？"
        ],
        "general": [
            "如何学习一门新的编程语言？",
            "提高工作效率的方法有哪些？",
            "健康的生活方式包括哪些方面？"
        ]
    }
    
    # 模拟文档内容
    sample_documents = {
        "financial": [
            "德赛电池（000049）的业绩预告超出预期，主要得益于iPhone 12 Pro Max需求强劲和新品盈利能力提升。预计2021年利润将持续增长，源于A客户的业务成长、非手机业务的增长以及并表比例的增加。",
            "公司2021年营收193.9亿元，同比增长5%；归母净利润63.69亿元，同比增长25.5%。产品结构优化，盈利能力提升。",
            "德赛电池在储能业务方面进展顺利，正在加大SIP领域的投入，预计公司成长速度加快。"
        ],
        "technical": [
            "Python性能优化的关键方法包括使用适当的数据结构、避免不必要的循环、利用内置函数和库。代码优化应该从算法层面开始，然后考虑语言特定的优化技巧。",
            "Docker容器化部署的最佳实践包括使用多阶段构建、优化镜像大小、合理设置资源限制、使用健康检查、实现自动化部署流程。",
            "机器学习模型的训练过程包括数据预处理、特征工程、模型选择、超参数调优、交叉验证、模型评估和部署等步骤。"
        ],
        "general": [
            "学习新编程语言的有效方法包括理解基础概念、动手实践项目、阅读优秀代码、参与开源项目、持续学习和实践。",
            "提高工作效率的方法包括时间管理、任务优先级排序、使用工具自动化、减少干扰、保持专注、定期休息和反思。",
            "健康的生活方式包括均衡饮食、规律运动、充足睡眠、心理健康、社交活动、避免不良习惯等多个方面。"
        ]
    }
    
    for domain, queries in test_queries.items():
        print(f"\n📊 测试领域: {domain.upper()}")
        print("-" * 40)
        
        for i, query in enumerate(queries, 1):
            print(f"\n🔍 查询 {i}: {query}")
            
            # 提取关键词
            keywords = extract_keywords_general(query, domain)
            print(f"   关键词: {keywords}")
            
            # 模拟检索到的文档
            docs = sample_documents[domain]
            
            # 提取相关句子
            all_relevant_sentences = []
            total_chars = 0
            max_chars = 2000
            
            for doc in docs[:3]:  # 只处理前3个文档
                relevant_sentences = extract_relevant_sentences_general(doc, keywords, max_chars_per_doc=800)
                
                for sentence in relevant_sentences:
                    if total_chars + len(sentence) <= max_chars:
                        all_relevant_sentences.append(sentence)
                        total_chars += len(sentence)
                    else:
                        break
                
                if total_chars >= max_chars:
                    break
            
            # 拼接上下文
            context = "\n\n".join(all_relevant_sentences)
            
            print(f"   上下文长度: {len(context)} 字符")
            print(f"   句子数量: {len(all_relevant_sentences)}")
            print(f"   前100字符: {context[:100]}...")
            
            # 计算优化效果
            original_length = sum(len(doc) for doc in docs)
            compression_ratio = (1 - len(context) / original_length) * 100
            print(f"   压缩比例: {compression_ratio:.1f}%")

def test_metadata_extraction_general():
    """测试通用元数据提取"""
    
    print(f"\n🔧 测试通用元数据提取")
    print("=" * 60)
    
    test_queries = [
        # 金融查询
        "德赛电池（000049）2021年利润增长原因",
        "000049的业绩表现",
        "德赛电池财务数据",
        
        # 技术查询
        "Python 3.9性能优化方法",
        "Docker容器部署最佳实践",
        "机器学习模型训练步骤",
        
        # 通用查询
        "如何学习编程语言",
        "提高工作效率的方法",
        "健康生活方式建议"
    ]
    
    for query in test_queries:
        print(f"\n📋 查询: {query}")
        
        # 尝试提取不同类型的元数据
        metadata = {}
        
        # 提取数字（可能是版本号、股票代码等）
        numbers = re.findall(r'\d+', query)
        if numbers:
            metadata['numbers'] = numbers
        
        # 提取英文单词（可能是技术术语、公司名等）
        english_words = re.findall(r'[A-Za-z]+', query)
        if english_words:
            metadata['english_words'] = english_words
        
        # 提取中文实体
        chinese_entities = re.findall(r'[\u4e00-\u9fff]+', query)
        if chinese_entities:
            metadata['chinese_entities'] = chinese_entities
        
        print(f"   提取的元数据: {metadata}")

if __name__ == "__main__":
    test_different_domains()
    test_metadata_extraction_general() 