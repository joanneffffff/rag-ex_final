from xlm.components.generator.llm_generator import LLMGenerator
from xlm.components.generator.local_llm_generator import LocalLLMGenerator
from xlm.registry import DEFAULT_LMS_ENDPOINT


def load_generator(
    generator_model_name: str,
    split_lines: bool = False,
    use_local_llm: bool = False,
    lms_endpoint: str = DEFAULT_LMS_ENDPOINT,
):
    """
    加载生成器
    Args:
        generator_model_name: 模型名称
        split_lines: 是否按行分割
        use_local_llm: 是否使用本地LLM
        lms_endpoint: LMS服务端点
    Returns:
        Generator实例
    """
    if use_local_llm:
        return LocalLLMGenerator(
            model_name=generator_model_name,
            cache_dir="D:/AI/huggingface",
            temperature=0.1,  # 降低温度以获得更确定性的答案
            max_new_tokens=50,  # 限制生成长度
            top_p=0.9
        )
    else:
        return LLMGenerator(
            model_name=generator_model_name,
            endpoint=lms_endpoint,
            split_lines=split_lines,
        )
