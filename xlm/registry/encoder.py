from xlm.components.encoder.encoder import Encoder


def load_encoder(model_name: str = "all-MiniLM-L6-v2", cache_dir: str = "D:/AI/huggingface"):
    """
    加载编码器
    Args:
        model_name: 模型名称
        cache_dir: 模型缓存目录
    Returns:
        编码器实例
    """
    return Encoder(
        model_name=model_name,
        cache_dir=cache_dir
    )
