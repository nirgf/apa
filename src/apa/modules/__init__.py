"""
Concrete module implementations for APA.

This package contains ready-to-use implementations of data importers,
processors, models, and pipelines that follow the common API.
"""

from apa.modules.data_importers import (
    HyperspectralDataImporter,
    GroundTruthDataImporter,
    RoadDataImporter,
)
from apa.modules.processors import (
    ROIProcessor,
    RoadExtractor,
    PCISegmenter,
    DataPreprocessor,
)
from apa.modules.models import (
    UNetModel,
    CNNModel,
    ModelManager,
)
from apa.modules.pipelines import (
    APAPipeline,
    ModularPipeline,
)

__all__ = [
    # Data importers
    "HyperspectralDataImporter",
    "GroundTruthDataImporter",
    "RoadDataImporter",
    # Processors
    "ROIProcessor",
    "RoadExtractor",
    "PCISegmenter",
    "DataPreprocessor",
    # Models
    "UNetModel",
    "CNNModel",
    "ModelManager",
    # Pipelines
    "APAPipeline",
    "ModularPipeline",
]

