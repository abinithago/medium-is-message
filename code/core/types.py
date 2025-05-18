"""Type definitions for the MIM package."""
from dataclasses import dataclass
from typing import Dict, List, Optional, Union
from enum import Enum

class PerturbationType(Enum):
    """Types of perturbations that can be applied to text."""
    GENDER = "gender"
    LANGUAGE = "language"
    STRUCTURE = "structure"

class AnnotationType(Enum):
    """Types of annotations that can be applied to responses."""
    TREATMENT = "treatment"
    PROFESSION = "profession"

@dataclass
class TextSample:
    """Represents a text sample with its metadata."""
    text: str
    id: str
    metadata: Dict[str, Union[str, int, float]]
    perturbations: Optional[List[PerturbationType]] = None

@dataclass
class LLMResponse:
    """Represents a response from an LLM."""
    text: str
    model: str
    sample_id: str
    confidence: float
    metadata: Dict[str, Union[str, int, float]] 