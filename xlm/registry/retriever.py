from xlm.components.retriever.sbert_retriever import SBERTRetriever
from xlm.components.encoder.encoder import Encoder
from xlm.dto.dto import DocumentWithMetadata, DocumentMetadata


def load_retriever(encoder_model_name: str, data_path: str, encoder=None):
    """
    加载检索器
    Args:
        encoder_model_name: 编码器模型名称
        data_path: 数据文件路径
        encoder: 可选的编码器实例
    Returns:
        检索器实例
    """
    # 如果提供了编码器，直接使用
    if encoder is None:
        # 加载默认编码器
        encoder = Encoder(
            model_name=encoder_model_name,
            cache_dir="D:/AI/huggingface"
        )
    
    # 读取文档
    with open(data_path, encoding="utf-8") as f:
        lines = f.readlines()
    
    # 转换为文档对象
    corpus_documents = []
    for line in lines:
        if line.strip():
            doc = DocumentWithMetadata(
                content=line.strip(),
                metadata=DocumentMetadata(
                    source=data_path,
                    created_at="",
                    author=""
                )
            )
            corpus_documents.append(doc)
    
    # 创建检索器
    return SBERTRetriever(encoder=encoder, corpus_documents=corpus_documents)
