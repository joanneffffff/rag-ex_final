# rag-ex

1.检索评估分支：
比较原始检索文档和扰动后的检索文档
评估检索器的鲁棒性
Ground truth是原始检索到的文档
2.生成评估分支：
比较原始生成答案和扰动后的生成答案
评估生成器的鲁棒性
Ground truth是原始prompt生成的答案

这样的设计更合理，因为：
检索扰动主要关注文档相似度和检索质量
生成扰动主要关注答案的语义一致性和生成质量
两个分支使用不同的评估指标，更适合各自的评估目标

------------------
multimodal_encoder2.py - 专门的表格和时间序列编码器

multimodal_encoder.py可以选择是否调用multimodal_encoder2.py
# 使用基础编码器
encoder = MultiModalEncoder(config, use_enhanced_encoders=False)

# 使用增强编码器
encoder = MultiModalEncoder(config, use_enhanced_encoders=True)

-----------------------------
1. 第一个：sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
这是编码器，用于把文本（或其它数据）转成向量（embedding）。
在你的代码里，主要是数据加载/预处理阶段用的，比如 UnifiedDataLoader 里用它对数据集做初步 embedding 或索引。
2. 第二个：sentence-transformers/all-MiniLM-L6-v2
这也是编码器，同样用于把文本等转成向量。
这是你在主配置里指定的主检索/问答用的编码器，比如 RAG 检索、UI 问答等。

------------------------
优化方向：
1. 集成 FinBERT（金融专用编码器/融合）
from xlm.components.encoder.finbert_sbert import FinBertSBERT
finbert_sbert_encoder = FinBertSBERT()
encoder = MultiModalEncoder(
    config=config,
    text_encoder=finbert_sbert_encoder,
    use_enhanced_encoders=True
)
2. Prompt Engineering（优化提示词）
self.prompt_template = (
    "You are a financial analyst assistant. "
    "Given the following context, answer the user's question as accurately and concisely as possible. "
    "If the answer requires calculation, show the calculation steps. "
    "If the answer is not found, say 'Not found in context.'\n\n"
    "Context:\n{context}\n\n"
    "Question: {question}\n"
    "Answer:"
)
3. Generator Model Selection（更强生成器模型）
generator = load_generator(
    generator_model_name="llama2-7b-chat",
    use_local_llm=True
)
4. 后处理
5. 检索-生成协同优化：rerank （TODO in future）