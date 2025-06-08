from abc import ABC, abstractmethod
from typing import List, Tuple, Union

from xlm.dto.dto import DocumentWithMetadata


class Retriever(ABC):
    @abstractmethod
    def retrieve(
        self,
        text: str,
        top_k: int = 3,
        return_scores: bool = False,
    ) -> Union[List[DocumentWithMetadata], Tuple[List[DocumentWithMetadata], List[float]]]: ...
