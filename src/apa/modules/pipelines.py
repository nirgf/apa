"""
Pipeline orchestration modules for APA.

Provides implementations for complete APA pipelines and modular pipeline creation.
"""

from typing import Any, Dict, Optional, List

from apa.common import (
    BasePipeline,
    DataContainer,
    PipelineResult,
)
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
)


class APAPipeline(BasePipeline):
    """
    Complete APA pipeline with all standard stages.
    
    Stages:
    1. data_import - Import hyperspectral imagery and ground truth data
    2. roi_processing - Process regions of interest
    3. road_extraction - Extract road networks from imagery
    4. pci_segmentation - Perform PCI segmentation
    5. data_preparation - Prepare data for neural network training
    6. model_training - Train machine learning models
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize APA pipeline.
        
        Args:
            config: Pipeline configuration
        """
        super().__init__("apa_pipeline", config)
        self._setup_stages()
    
    def _setup_stages(self):
        """Set up all pipeline stages."""
        # Stage 1: Data import
        self.add_stage(
            'data_import',
            HyperspectralDataImporter(self.config.get('data_import', {})),
            dependencies=[]
        )
        
        # Stage 2: ROI processing
        self.add_stage(
            'roi_processing',
            ROIProcessor(self.config.get('roi_processing', {})),
            dependencies=['data_import']
        )
        
        # Stage 3: Road extraction
        self.add_stage(
            'road_extraction',
            RoadExtractor(self.config.get('road_extraction', {})),
            dependencies=['roi_processing']
        )
        
        # Stage 4: PCI segmentation
        self.add_stage(
            'pci_segmentation',
            PCISegmenter(self.config.get('pci_segmentation', {})),
            dependencies=['road_extraction']
        )
        
        # Stage 5: Data preparation
        self.add_stage(
            'data_preparation',
            DataPreprocessor(self.config.get('data_preparation', {})),
            dependencies=['pci_segmentation']
        )
        
        # Stage 6: Model training
        model_config = self.config.get('model_training', {})
        model_type = model_config.get('model_type', 'unet')
        
        if model_type == 'unet':
            model = UNetModel(model_config)
        else:
            model = CNNModel(model_config)
        
        self.add_stage(
            'model_training',
            model,
            dependencies=['data_preparation']
        )


class ModularPipeline(BasePipeline):
    """
    Modular pipeline for custom stage composition.
    
    Allows users to create custom pipelines by adding stages manually.
    """
    
    def __init__(self, name: str = "modular_pipeline", config: Optional[Dict[str, Any]] = None):
        """
        Initialize modular pipeline.
        
        Args:
            name: Pipeline name
            config: Pipeline configuration
        """
        super().__init__(name, config)
    
    def add_custom_stage(self, name: str, stage: Any, dependencies: Optional[List[str]] = None):
        """
        Add a custom stage to the pipeline.
        
        Args:
            name: Stage name
            stage: Stage implementation (must implement appropriate interface)
            dependencies: List of stage names this stage depends on
        """
        self.add_stage(name, stage, dependencies)

