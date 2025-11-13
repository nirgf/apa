"""
APA - Advanced Pavement Analytics

A geospatial AI pipeline for predicting Pavement Condition Index (PCI) 
from satellite imagery.
"""

__version__ = "0.1.0"
__author__ = "APA Team"

# Main exports
from apa.common import (
    DataContainer,
    ProcessingResult,
    ModelResult,
    PipelineResult,
    BaseModule,
    BaseDataProcessor,
    BaseModel,
    BasePipeline,
)

__all__ = [
    "DataContainer",
    "ProcessingResult",
    "ModelResult",
    "PipelineResult",
    "BaseModule",
    "BaseDataProcessor",
    "BaseModel",
    "BasePipeline",
]

