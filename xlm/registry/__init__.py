from enum import Enum

DEFAULT_LMS_ENDPOINT = "http://localhost:9985"

class ExplanationGranularity(str, Enum):
    WORD_LEVEL = "WORD_LEVEL"
    SENTENCE_LEVEL = "SENTENCE_LEVEL"
    PARAGRAPH_LEVEL = "PARAGRAPH_LEVEL"
