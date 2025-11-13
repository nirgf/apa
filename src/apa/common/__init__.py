"""
Common API components for APA modules.

This package provides standardized interfaces, base classes, data structures,
validators, and exception handling for all APA modules.
"""

from apa.common.data_structures import (
    DataContainer,
    ProcessingResult,
    ModelResult,
    PipelineResult,
)
from apa.common.base_classes import (
    BaseModule,
    BaseDataProcessor,
    BaseModel,
    BasePipeline,
)
from apa.common.exceptions import (
    APAException,
    ValidationError,
    ProcessingError,
    ModelError,
    DataError,
    ConfigurationError,
    PipelineError,
)

__all__ = [
    # Data structures
    "DataContainer",
    "ProcessingResult",
    "ModelResult",
    "PipelineResult",
    # Base classes
    "BaseModule",
    "BaseDataProcessor",
    "BaseModel",
    "BasePipeline",
    # Exceptions
    "APAException",
    "ValidationError",
    "ProcessingError",
    "ModelError",
    "DataError",
    "ConfigurationError",
    "PipelineError",
]
