import re
import jieba
from typing import List, Optional, Dict
from enum import Enum

class Granularity(Enum):
    """特征提取粒度枚举"""
    WORD = "word"
    SENTENCE = "sentence"
    PHRASE = "phrase"
    ENTITY = "entity"
    NUMBER = "number"
    YEAR = "year"

class FeatureExtractor:
    """特征提取器，支持多种粒度的特征提取"""
    
    def __init__(self, language: str = "zh"):
        """
        初始化特征提取器
        
        Args:
            language: 语言类型 ("zh" 或 "en")
        """
        self.language = language
        self._setup_jieba()
    
    def _setup_jieba(self):
        """设置jieba分词器"""
        if self.language == "zh":
            # 添加金融领域词汇
            financial_terms = [
                "市盈率", "市净率", "市销率", "净利润", "营业收入", "营业利润",
                "总资产", "净资产", "负债", "现金流", "毛利率", "净利率",
                "营收", "利润", "资产", "负债", "权益", "现金流"
            ]
            for term in financial_terms:
                jieba.add_word(term)
    
    def extract_features(self, text: str, granularity: Granularity = Granularity.WORD) -> List[str]:
        """
        根据指定粒度提取特征
        
        Args:
            text: 输入文本
            granularity: 特征粒度
            
        Returns:
            特征列表
        """
        if granularity == Granularity.WORD:
            return self._extract_words(text)
        elif granularity == Granularity.SENTENCE:
            return self._extract_sentences(text)
        elif granularity == Granularity.PHRASE:
            return self._extract_phrases(text)
        elif granularity == Granularity.ENTITY:
            return self._extract_entities(text)
        elif granularity == Granularity.NUMBER:
            return self._extract_numbers(text)
        elif granularity == Granularity.YEAR:
            return self._extract_years(text)
        else:
            raise ValueError(f"Unsupported granularity: {granularity}")
    
    def _extract_words(self, text: str) -> List[str]:
        """提取词汇特征"""
        if self.language == "zh":
            # 使用jieba分词
            words = list(jieba.cut(text))
            # 过滤掉停用词和标点符号
            filtered_words = []
            for word in words:
                if len(word.strip()) > 1 and not re.match(r'^[^\w\u4e00-\u9fff]+$', word):
                    filtered_words.append(word.strip())
            return filtered_words
        else:
            # 英文分词
            words = re.findall(r'\b\w+\b', text.lower())
            return [word for word in words if len(word) > 2]
    
    def _extract_sentences(self, text: str) -> List[str]:
        """提取句子特征"""
        if self.language == "zh":
            # 中文句子分割
            sentences = re.split(r'[。！？；]', text)
        else:
            # 英文句子分割
            sentences = re.split(r'[.!?;]', text)
        
        return [s.strip() for s in sentences if len(s.strip()) > 5]
    
    def _extract_phrases(self, text: str) -> List[str]:
        """提取短语特征"""
        # 简单的短语提取：3-5个词的组合
        words = self._extract_words(text)
        phrases = []
        
        for i in range(len(words) - 2):
            for j in range(3, min(6, len(words) - i + 1)):
                phrase = "".join(words[i:i+j]) if self.language == "zh" else " ".join(words[i:i+j])
                if len(phrase) > 3:
                    phrases.append(phrase)
        
        return phrases[:20]  # 限制短语数量
    
    def _extract_entities(self, text: str) -> List[str]:
        """提取实体特征（公司名、数字等）"""
        entities = []
        
        # 提取公司名（简单模式）
        company_patterns = [
            r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',  # 英文公司名
            r'([\u4e00-\u9fff]+(?:股份|集团|公司|有限|科技|投资))',  # 中文公司名
        ]
        
        for pattern in company_patterns:
            matches = re.findall(pattern, text)
            entities.extend(matches)
        
        # 提取数字
        numbers = re.findall(r'\d+(?:\.\d+)?', text)
        entities.extend(numbers)
        
        return entities
    
    def _extract_numbers(self, text: str) -> List[str]:
        """提取数字特征"""
        numbers = re.findall(r'\d+(?:\.\d+)?(?:%|万|亿|千)?', text)
        return numbers
    
    def _extract_years(self, text: str) -> List[str]:
        """提取年份特征"""
        if self.language == "zh":
            years = re.findall(r'(20\d{2})年', text)
        else:
            years = re.findall(r'(20\d{2})', text)
        return years
    
    def extract_all_features(self, text: str) -> Dict[str, List[str]]:
        """
        提取所有类型的特征
        
        Args:
            text: 输入文本
            
        Returns:
            包含所有特征类型的字典
        """
        return {
            'words': self._extract_words(text),
            'sentences': self._extract_sentences(text),
            'phrases': self._extract_phrases(text),
            'entities': self._extract_entities(text),
            'numbers': self._extract_numbers(text),
            'years': self._extract_years(text)
        } 