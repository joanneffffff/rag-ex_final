#!/usr/bin/env python3
"""
扰动实验结果分析工具
提供详细的结果分析和可视化
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any
from dataclasses import asdict
import argparse

class PerturbationResultAnalyzer:
    """扰动实验结果分析器"""
    
    def __init__(self, results_file: str):
        """
        初始化分析器
        
        Args:
            results_file: 结果文件路径
        """
        self.results_file = results_file
        self.results = self.load_results()
        self.df = self.create_dataframe()
    
    def load_results(self) -> List[Dict[str, Any]]:
        """加载结果文件"""
        try:
            with open(self.results_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"❌ 加载结果文件失败: {e}")
            return []
    
    def create_dataframe(self) -> pd.DataFrame:
        """创建DataFrame用于分析"""
        if not self.results:
            return pd.DataFrame()
        
        # 转换为DataFrame
        df = pd.DataFrame(self.results)
        
        # 添加计算列
        df['f1_drop'] = df['f1_original_vs_expected'] - df['f1_perturbed_vs_expected']
        df['f1_ratio'] = df['f1_perturbed_vs_expected'] / df['f1_original_vs_expected'].replace(0, 1)
        
        return df
    
    def basic_statistics(self) -> Dict[str, Any]:
        """基础统计分析"""
        if self.df.empty:
            return {}
        
        stats = {
            'total_experiments': len(self.df),
            'perturber_counts': self.df['perturber_name'].value_counts().to_dict(),
            'avg_f1_scores': {
                'original_vs_expected': self.df['f1_original_vs_expected'].mean(),
                'perturbed_vs_expected': self.df['f1_perturbed_vs_expected'].mean(),
                'perturbed_vs_original': self.df['f1_perturbed_vs_original'].mean()
            },
            'avg_f1_drop': self.df['f1_drop'].mean(),
            'avg_f1_ratio': self.df['f1_ratio'].mean()
        }
        
        return stats
    
    def perturber_analysis(self) -> Dict[str, Any]:
        """扰动器分析"""
        if self.df.empty:
            return {}
        
        analysis = {}
        
        for perturber in self.df['perturber_name'].unique():
            perturber_data = self.df[self.df['perturber_name'] == perturber]
            
            analysis[perturber] = {
                'count': len(perturber_data),
                'avg_f1_original': perturber_data['f1_original_vs_expected'].mean(),
                'avg_f1_perturbed': perturber_data['f1_perturbed_vs_expected'].mean(),
                'avg_f1_drop': perturber_data['f1_drop'].mean(),
                'avg_f1_ratio': perturber_data['f1_ratio'].mean(),
                'success_rate': (perturber_data['f1_perturbed_vs_expected'] > 0.5).mean()
            }
        
        return analysis
    
    def llm_judge_analysis(self) -> Dict[str, Any]:
        """LLM Judge分析"""
        if self.df.empty:
            return {}
        
        # 过滤有LLM Judge分数的数据
        judge_data = self.df.dropna(subset=['llm_judge_score_accuracy'])
        
        if judge_data.empty:
            return {'error': '没有LLM Judge数据'}
        
        analysis = {
            'total_judged': len(judge_data),
            'avg_scores': {
                'accuracy': judge_data['llm_judge_score_accuracy'].mean(),
                'completeness': judge_data['llm_judge_score_completeness'].mean(),
                'professionalism': judge_data['llm_judge_score_professionalism'].mean()
            },
            'perturber_judge_scores': {}
        }
        
        # 按扰动器分组分析
        for perturber in judge_data['perturber_name'].unique():
            perturber_data = judge_data[judge_data['perturber_name'] == perturber]
            analysis['perturber_judge_scores'][perturber] = {
                'count': len(perturber_data),
                'avg_accuracy': perturber_data['llm_judge_score_accuracy'].mean(),
                'avg_completeness': perturber_data['llm_judge_score_completeness'].mean(),
                'avg_professionalism': perturber_data['llm_judge_score_professionalism'].mean()
            }
        
        return analysis
    
    def generate_visualizations(self, output_dir: str = "analysis_plots"):
        """生成可视化图表"""
        if self.df.empty:
            print("❌ 没有数据可以可视化")
            return
        
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 1. F1分数对比图
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # F1分数分布
        axes[0, 0].hist(self.df['f1_original_vs_expected'], alpha=0.7, label='原始答案', bins=20)
        axes[0, 0].hist(self.df['f1_perturbed_vs_expected'], alpha=0.7, label='扰动答案', bins=20)
        axes[0, 0].set_title('F1分数分布对比')
        axes[0, 0].legend()
        
        # 扰动器F1分数对比
        perturber_f1 = self.df.groupby('perturber_name')['f1_perturbed_vs_expected'].mean()
        perturber_f1.plot(kind='bar', ax=axes[0, 1])
        axes[0, 1].set_title('各扰动器平均F1分数')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # F1下降幅度
        axes[1, 0].hist(self.df['f1_drop'], bins=20)
        axes[1, 0].set_title('F1分数下降幅度分布')
        axes[1, 0].axvline(x=0, color='red', linestyle='--', label='无变化')
        axes[1, 0].legend()
        
        # LLM Judge分数（如果有）
        if 'llm_judge_score_accuracy' in self.df.columns:
            judge_data = self.df.dropna(subset=['llm_judge_score_accuracy'])
            if not judge_data.empty:
                judge_scores = ['llm_judge_score_accuracy', 'llm_judge_score_completeness', 'llm_judge_score_professionalism']
                judge_means = [judge_data[col].mean() for col in judge_scores]
                axes[1, 1].bar(['准确性', '完整性', '专业性'], judge_means)
                axes[1, 1].set_title('LLM Judge平均分数')
                axes[1, 1].set_ylim(0, 10)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/f1_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. 扰动器效果对比
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # 扰动器成功率
        success_rates = {}
        for perturber in self.df['perturber_name'].unique():
            perturber_data = self.df[self.df['perturber_name'] == perturber]
            success_rate = (perturber_data['f1_perturbed_vs_expected'] > 0.5).mean()
            success_rates[perturber] = success_rate
        
        axes[0].bar(success_rates.keys(), success_rates.values())
        axes[0].set_title('各扰动器成功率 (F1 > 0.5)')
        axes[0].tick_params(axis='x', rotation=45)
        axes[0].set_ylim(0, 1)
        
        # 扰动器F1比率
        f1_ratios = self.df.groupby('perturber_name')['f1_ratio'].mean()
        axes[1].bar(f1_ratios.index, f1_ratios.values)
        axes[1].set_title('各扰动器F1比率 (扰动/原始)')
        axes[1].tick_params(axis='x', rotation=45)
        axes[1].axhline(y=1, color='red', linestyle='--', label='无变化')
        axes[1].legend()
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/perturber_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 可视化图表已保存到 {output_dir}/")
    
    def export_summary_report(self, output_file: str = "perturbation_analysis_report.json"):
        """导出分析报告"""
        report = {
            'basic_statistics': self.basic_statistics(),
            'perturber_analysis': self.perturber_analysis(),
            'llm_judge_analysis': self.llm_judge_analysis(),
            'metadata': {
                'results_file': self.results_file,
                'total_experiments': len(self.df) if not self.df.empty else 0,
                'analysis_timestamp': pd.Timestamp.now().isoformat()
            }
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        print(f"📋 分析报告已保存到: {output_file}")
        return report
    
    def print_summary(self):
        """打印分析摘要"""
        print("📊 扰动实验结果分析摘要")
        print("=" * 50)
        
        # 基础统计
        stats = self.basic_statistics()
        if stats:
            print(f"总实验数: {stats['total_experiments']}")
            print(f"扰动器分布: {stats['perturber_counts']}")
            print(f"平均F1分数:")
            print(f"  原始答案 vs 期望答案: {stats['avg_f1_scores']['original_vs_expected']:.3f}")
            print(f"  扰动答案 vs 期望答案: {stats['avg_f1_scores']['perturbed_vs_expected']:.3f}")
            print(f"  扰动答案 vs 原始答案: {stats['avg_f1_scores']['perturbed_vs_original']:.3f}")
            print(f"平均F1下降: {stats['avg_f1_drop']:.3f}")
            print(f"平均F1比率: {stats['avg_f1_ratio']:.3f}")
        
        # 扰动器分析
        perturber_analysis = self.perturber_analysis()
        if perturber_analysis:
            print(f"\n扰动器详细分析:")
            for perturber, analysis in perturber_analysis.items():
                print(f"  {perturber}:")
                print(f"    实验数: {analysis['count']}")
                print(f"    平均F1分数: {analysis['avg_f1_perturbed']:.3f}")
                print(f"    平均F1下降: {analysis['avg_f1_drop']:.3f}")
                print(f"    成功率: {analysis['success_rate']:.1%}")
        
        # LLM Judge分析
        judge_analysis = self.llm_judge_analysis()
        if judge_analysis and 'error' not in judge_analysis:
            print(f"\nLLM Judge分析:")
            print(f"  评估样本数: {judge_analysis['total_judged']}")
            print(f"  平均分数:")
            print(f"    准确性: {judge_analysis['avg_scores']['accuracy']:.2f}")
            print(f"    完整性: {judge_analysis['avg_scores']['completeness']:.2f}")
            print(f"    专业性: {judge_analysis['avg_scores']['professionalism']:.2f}")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='扰动实验结果分析工具')
    parser.add_argument('results_file', help='结果文件路径')
    parser.add_argument('--output', type=str, default='analysis_report.json', 
                       help='分析报告输出文件')
    parser.add_argument('--plots', action='store_true', help='生成可视化图表')
    parser.add_argument('--plots_dir', type=str, default='analysis_plots', 
                       help='图表输出目录')
    
    args = parser.parse_args()
    
    # 创建分析器
    analyzer = PerturbationResultAnalyzer(args.results_file)
    
    # 打印摘要
    analyzer.print_summary()
    
    # 导出报告
    analyzer.export_summary_report(args.output)
    
    # 生成可视化
    if args.plots:
        analyzer.generate_visualizations(args.plots_dir)
    
    print(f"\n✅ 分析完成！")

if __name__ == "__main__":
    main() 