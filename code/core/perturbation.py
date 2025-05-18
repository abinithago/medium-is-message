"""Base classes for text perturbation."""
from abc import ABC, abstractmethod
from typing import List, Optional

from .types import TextSample, PerturbationType

class BasePerturbation(ABC):
    """Abstract base class for all perturbation types."""
    
    def __init__(self, perturbation_type: PerturbationType):
        """Initialize the perturbation.
        
        Args:
            perturbation_type: The type of perturbation being applied
        """
        self.perturbation_type = perturbation_type
    
    @abstractmethod
    def perturb(self, sample: TextSample) -> List[TextSample]:
        """Apply the perturbation to a text sample.
        
        Args:
            sample: The text sample to perturb
            
        Returns:
            A list of perturbed text samples
        """
        pass
    
    @abstractmethod
    def validate(self, original: TextSample, perturbed: TextSample) -> bool:
        """Validate that the perturbation was applied correctly.
        
        Args:
            original: The original text sample
            perturbed: The perturbed text sample
            
        Returns:
            True if the perturbation is valid, False otherwise
        """
        pass 