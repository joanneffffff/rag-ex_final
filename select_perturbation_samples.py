#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
选择适合扰动实验的中文样本
为trend、year、term三种扰动类型选择总共20个样本
"""

import json
import re
from typing import List, Dict, Set, Tuple
from collections import defaultdict

class PerturbationSampleSelector:
    def __init__(self):
        # 定义三种扰动类型的关键词
        self.trend_keywords = {
            '上升', '下降', '上涨', '下跌', '增长', '减少', '提升', '降低', '增加', '减少',
            '好转', '恶化', '改善', '积极', '消极', '盈利', '亏损', '扩张', '收缩',
            '持续增长', '持续下滑', '稳步增长', '显著下降', '强劲', '疲软', '高于', '低于',
            '优于', '劣于', '领先', '落后', '增加率', '减少率', '上升趋势', '下降趋势',
            '增长趋势', '减少趋势'
        }
        
        # 年份关键词使用正则表达式动态检测，与YearPerturber保持一致
        self.year_pattern = re.compile(r'\b(20\d{2})(?:年|年度)?\b')
        
        self.term_keywords = {
            '市盈率', '净利润', '市净率', '市销率', '营收', '收入', '营业收入', '营业利润',
            '营业利润', '总资产', '净资产', '负债', '资产', '利润', '成本', '市值', '估值',
            '股息', '分红', '配股', '增发', '回购', '交易量', '成交额', '换手率'
        }
    
    def load_dataset(self, file_path: str) -> List[Dict]:
        """加载评测数据集 - 支持JSONL格式"""
        samples = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if line:
                    try:
                        samples.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        print(f"⚠️ 第{line_num}行JSON解析失败: {e}")
                        continue
        return samples
    
    def extract_keywords(self, text: str, keyword_set: Set[str]) -> Set[str]:
        """从文本中提取关键词 - 使用正则表达式全词匹配"""
        found_keywords = set()
        for keyword in keyword_set:
            # 使用正则表达式进行全词匹配，与扰动器保持一致
            pattern = r'\b' + re.escape(keyword) + r'\b'
            if re.search(pattern, text, re.IGNORECASE):
                found_keywords.add(keyword)
        return found_keywords
    
    def analyze_sample(self, sample: Dict) -> Dict:
        """分析单个样本的关键词分布"""
        summary = sample.get('summary', '')
        content = sample.get('content', '')
        generated_question = sample.get('generated_question', '')
        
        # 主要关注context字段（summary和content），因为这是扰动器作用的对象
        context_text = f"{summary} {content}"
        question_text = generated_question
        
        # 分别分析context和question中的关键词
        context_trend_found = self.extract_keywords(context_text, self.trend_keywords)
        context_year_found = set()
        context_year_matches = self.year_pattern.findall(context_text)
        for match in context_year_matches:
            context_year_found.add(match)
        context_term_found = self.extract_keywords(context_text, self.term_keywords)
        
        question_trend_found = self.extract_keywords(question_text, self.trend_keywords)
        question_year_found = set()
        question_year_matches = self.year_pattern.findall(question_text)
        for match in question_year_matches:
            question_year_found.add(match)
        question_term_found = self.extract_keywords(question_text, self.term_keywords)
        
        # 合并所有关键词（但主要权重给context）
        trend_found = context_trend_found | question_trend_found
        year_found = context_year_found | question_year_found
        term_found = context_term_found | question_term_found
        
        return {
            'sample_id': sample.get('id', 'unknown'),
            'summary': summary,
            'content': content,
            'generated_question': generated_question,
            'trend_keywords': trend_found,
            'year_keywords': year_found,
            'term_keywords': term_found,
            'context_trend_score': len(context_trend_found),
            'context_year_score': len(context_year_found),
            'context_term_score': len(context_term_found),
            'question_trend_score': len(question_trend_found),
            'question_year_score': len(question_year_found),
            'question_term_score': len(question_term_found),
            'trend_score': len(trend_found),
            'year_score': len(year_found),
            'term_score': len(term_found),
            'total_score': len(trend_found) + len(year_found) + len(term_found),
            'context_score': len(context_trend_found) + len(context_year_found) + len(context_term_found)
        }
    
    def select_samples(self, samples: List[Dict], target_count: int = 20) -> Dict[str, List[Dict]]:
        """选择适合的样本 - 使用多样性选择策略"""
        analyzed_samples = [self.analyze_sample(sample) for sample in samples]
        
        # 第一轮：按context_score排序，优先选择context中有关键词的样本
        context_samples = [s for s in analyzed_samples if s['context_score'] > 0]
        context_samples.sort(key=lambda x: x['context_score'], reverse=True)
        
        # 第二轮：多样性选择，确保覆盖不同的问题类型和上下文类型
        selected_samples = []
        remaining_samples = context_samples.copy()
        
        # 确保覆盖不同的问题类型和上下文类型
        type_coverage = {
            'question_types': set(),
            'context_types': set()
        }
        
        # 首先选择context_score最高的样本
        for _ in range(min(target_count, len(context_samples))):
            if not remaining_samples:
                break
            
            # 计算每个样本的多样性分数
            for sample in remaining_samples:
                # 简单的多样性计算：与已选样本的差异
                diversity_score = 0.0
                for selected in selected_samples:
                    # 问题类型差异
                    if sample.get('question_type') != selected.get('question_type'):
                        diversity_score += 1.0
                    # 上下文类型差异
                    if sample.get('context_type') != selected.get('context_type'):
                        diversity_score += 1.0
                
                sample['diversity_score'] = diversity_score
            
            # 选择最佳样本（平衡context_score和多样性）
            best_sample = max(remaining_samples, key=lambda s: s['context_score'] + s.get('diversity_score', 0))
            
            selected_samples.append(best_sample)
            remaining_samples.remove(best_sample)
        
        # 如果还不够，从剩余样本中补充
        if len(selected_samples) < target_count:
            remaining_all = [s for s in analyzed_samples if s not in selected_samples]
            remaining_all.sort(key=lambda x: x['total_score'], reverse=True)
            selected_samples.extend(remaining_all[:target_count - len(selected_samples)])
        
        # 按扰动类型分类
        categorized_samples = {
            'trend': [],
            'year': [],
            'term': []
        }
        
        for sample in selected_samples:
            # 每个样本可以用于多种扰动类型
            if sample['trend_score'] > 0:
                categorized_samples['trend'].append(sample)
            if sample['year_score'] > 0:
                categorized_samples['year'].append(sample)
            if sample['term_score'] > 0:
                categorized_samples['term'].append(sample)
        
        return categorized_samples
    
    def print_analysis(self, categorized_samples: Dict[str, List[Dict]]):
        """打印分析结果"""
        print("=" * 80)
        print("📊 扰动样本选择分析结果")
        print("=" * 80)
        
        for perturber_type, samples in categorized_samples.items():
            print(f"\n🔍 {perturber_type.upper()} 扰动器样本 ({len(samples)}个):")
            print("-" * 60)
            
            for i, sample in enumerate(samples, 1):
                print(f"{i:2d}. 样本ID: {sample['sample_id']}")
                print(f"    趋势关键词: {sample['trend_keywords']}")
                print(f"    年份关键词: {sample['year_keywords']}")
                print(f"    术语关键词: {sample['term_keywords']}")
                print(f"    总分: {sample['total_score']}")
                print(f"    问题: {sample['generated_question'][:100]}...")
                print()
        
        # 统计信息
        print("📈 统计信息:")
        print("-" * 40)
        total_samples = len(set([s['sample_id'] for samples in categorized_samples.values() for s in samples]))
        print(f"总样本数: {total_samples}")
        for perturber_type, samples in categorized_samples.items():
            print(f"{perturber_type} 扰动器可用样本: {len(samples)}")
    
    def save_selected_samples(self, categorized_samples: Dict[str, List[Dict]], output_file: str):
        """保存选中的样本"""
        # 去重并保存
        unique_samples = {}
        for samples in categorized_samples.values():
            for sample in samples:
                # 转换set为list以便JSON序列化
                sample_copy = sample.copy()
                sample_copy['trend_keywords'] = list(sample['trend_keywords'])
                sample_copy['year_keywords'] = list(sample['year_keywords'])
                sample_copy['term_keywords'] = list(sample['term_keywords'])
                unique_samples[sample['sample_id']] = sample_copy
        
        selected_data = {
            'total_samples': len(unique_samples),
            'categorized_samples': {
                k: [s.copy() for s in v] for k, v in categorized_samples.items()
            },
            'unique_samples': list(unique_samples.values())
        }
        
        # 转换所有set为list
        for perturber_type in selected_data['categorized_samples']:
            for sample in selected_data['categorized_samples'][perturber_type]:
                sample['trend_keywords'] = list(sample['trend_keywords'])
                sample['year_keywords'] = list(sample['year_keywords'])
                sample['term_keywords'] = list(sample['term_keywords'])
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(selected_data, f, ensure_ascii=False, indent=2)
        
        print(f"✅ 选中样本已保存到: {output_file}")

def main():
    """主函数"""
    selector = PerturbationSampleSelector()
    
    # 加载数据集
    print("📂 加载评测数据集...")
    samples = selector.load_dataset('data/alphafin/alphafin_eval_samples_updated.jsonl')
    print(f"✅ 加载了 {len(samples)} 个样本")
    
    # 选择样本
    print("\n🔍 分析样本关键词分布...")
    categorized_samples = selector.select_samples(samples, target_count=20)
    
    # 打印分析结果
    selector.print_analysis(categorized_samples)
    
    # 保存结果
    output_file = 'selected_perturbation_samples.json'
    selector.save_selected_samples(categorized_samples, output_file)
    
    print(f"\n🎯 完成！已为三种扰动类型选择了20个样本")
    print(f"📁 结果文件: {output_file}")

if __name__ == "__main__":
    main() 