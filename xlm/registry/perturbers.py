from xlm.modules.perturber.leave_one_out_perturber import LeaveOneOutPerturber
from xlm.modules.perturber.llm_based_perturber import LLMBasedPerturber
from xlm.modules.perturber.random_word_perturber import RandomWordPerturber
from xlm.modules.perturber.reorder_perturber import ReorderPerturber
from xlm.components.generator.llm_generator import LLMGenerator
from xlm.registry import DEFAULT_LMS_ENDPOINT
from xlm.modules.perturber.trend_perturber import TrendPerturber
from xlm.modules.perturber.term_perturber import TermPerturber
from xlm.modules.perturber.year_perturber import YearPerturber

# 初始化基本perturbers
leave_one_out_perturber = LeaveOneOutPerturber()
random_word_perturber = RandomWordPerturber()
reorder_perturber = ReorderPerturber()

# 初始化LLM-based perturbers
antonym_perturber = LLMBasedPerturber(
    generator=LLMGenerator(
        model_name="gpt2",
        endpoint=DEFAULT_LMS_ENDPOINT,
        split_lines=True
    ),
    prompt_template="Generate an antonym for: {text}\nAntonym:",
)

synonym_perturber = LLMBasedPerturber(
    generator=LLMGenerator(
        model_name="gpt2",
        endpoint=DEFAULT_LMS_ENDPOINT,
        split_lines=True
    ),
    prompt_template="Generate a synonym for: {text}\nSynonym:",
)

paraphrase_perturber = LLMBasedPerturber(
    generator=LLMGenerator(
        model_name="gpt2",
        endpoint=DEFAULT_LMS_ENDPOINT,
        split_lines=True
    ),
    prompt_template="Paraphrase this text: {text}\nParaphrase:",
)

# 注册所有perturbers
PERTURBERS = {
    "leave_one_out": leave_one_out_perturber,
    "random_word_perturber": random_word_perturber,
    "reorder_perturber": reorder_perturber,
    "antonym_perturber": antonym_perturber,
    "synonym_perturber": synonym_perturber,
    "paraphrase_perturber": paraphrase_perturber,
    "trend_perturber": TrendPerturber(),
    "term_perturber": TermPerturber(),
    "year_perturber": YearPerturber(),
}


def load_perturber(perturber_name: str):
    if perturber_name not in PERTURBERS.keys():
        raise Exception(
            f"The entered perturber name is not found! Available "
            f"perturbers are: {list(PERTURBERS.keys())}"
        )

    return PERTURBERS.get(perturber_name)
