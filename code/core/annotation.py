"""Base classes for response annotation."""
from abc import ABC, abstractmethod
from typing import Any, Dict

from .types import LLMResponse, AnnotationType

class BaseAnnotator(ABC):
    """Abstract base class for all annotator types."""
    
    def __init__(self, annotation_type: AnnotationType):
        """Initialize the annotator.
        
        Args:
            annotation_type: The type of annotation being applied
        """
        self.annotation_type = annotation_type
    
    @abstractmethod
    def annotate(self, response: LLMResponse) -> Dict[str, Any]:
        """Annotate an LLM response.
        
        Args:
            response: The LLM response to annotate
            
        Returns:
            A dictionary containing the annotation results
        """
        pass
    
    @abstractmethod
    def validate(self, response: LLMResponse, annotation: Dict[str, Any]) -> bool:
        """Validate that the annotation is correct.
        
        Args:
            response: The original LLM response
            annotation: The annotation results
            
        Returns:
            True if the annotation is valid, False otherwise
        """
        pass 