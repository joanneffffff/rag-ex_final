from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional

class BasePerturber(ABC):
    """Abstract base class for all perturbers."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.name = self.__class__.__name__.replace('Perturber', '').lower()

    @abstractmethod
    def perturb(self, original_text: str, features: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Applies perturbation to the original text.
        
        Args:
            original_text: The original text to perturb
            features: Optional list of features to target for perturbation
            
        Returns:
            List of dictionaries, each containing:
            - 'perturbed_text': The perturbed text
            - 'perturbation_detail': Description of what was changed
            - 'original_feature': The original feature that was perturbed (if applicable)
        """
        pass

    @property
    def perturber_name(self) -> str:
        """Returns the name of the perturber."""
        return self.name

    def get_config(self) -> Dict[str, Any]:
        """Returns the configuration of the perturber."""
        return self.config

    def set_config(self, config: Dict[str, Any]) -> None:
        """Sets the configuration of the perturber."""
        self.config.update(config) 