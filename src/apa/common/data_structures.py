"""
Standardized data structures for APA.

Provides unified data containers and result objects that all modules
use for input/output, ensuring consistency across the pipeline.
"""

from typing import Any, Dict, Optional, List
from dataclasses import dataclass, field
import numpy as np


@dataclass
class DataContainer:
    """
    Standardized data container for all APA modules.
    
    This container holds data in a consistent format, making it easy
    to pass data between modules in the pipeline.
    """
    
    data: Any = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    data_type: str = "unknown"
    shape: Optional[tuple] = None
    dtype: Optional[str] = None
    
    def __post_init__(self):
        """Validate and set default values after initialization."""
        if self.data is not None:
            if isinstance(self.data, np.ndarray):
                self.shape = self.data.shape
                self.dtype = str(self.data.dtype)
            elif isinstance(self.data, dict):
                # For dictionary data (e.g., {'image': ..., 'lon_mat': ..., 'lat_mat': ...})
                # Set shape from the 'image' key if it exists
                if 'image' in self.data and isinstance(self.data['image'], np.ndarray):
                    self.shape = self.data['image'].shape
                    self.dtype = str(self.data['image'].dtype)
            elif hasattr(self.data, 'shape'):
                self.shape = self.data.shape
            if 'data_type' not in self.metadata:
                self.metadata['data_type'] = self.data_type
    
    def validate(self) -> bool:
        """
        Validate the data container.
        
        Returns:
            True if valid, raises ValidationError otherwise
        """
        if self.data is None:
            raise ValueError("Data container cannot have None data")
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert container to dictionary."""
        return {
            'data': self.data,
            'metadata': self.metadata,
            'data_type': self.data_type,
            'shape': self.shape,
            'dtype': self.dtype,
        }


@dataclass
class ProcessingResult:
    """
    Result object for processing operations.
    
    Contains the processed data and metadata about the processing operation.
    """
    
    data: DataContainer
    success: bool = True
    message: str = ""
    metrics: Dict[str, Any] = field(default_factory=dict)
    processing_time: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            'data': self.data.to_dict(),
            'success': self.success,
            'message': self.message,
            'metrics': self.metrics,
            'processing_time': self.processing_time,
        }


@dataclass
class ModelResult:
    """
    Result object for model operations (training, prediction).
    
    Contains model outputs, predictions, and training metrics.
    """
    
    predictions: Optional[Any] = None
    probabilities: Optional[Any] = None
    success: bool = True
    message: str = ""
    metrics: Dict[str, Any] = field(default_factory=dict)
    model_info: Dict[str, Any] = field(default_factory=dict)
    training_history: Optional[Dict[str, List[float]]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            'predictions': self.predictions,
            'probabilities': self.probabilities,
            'success': self.success,
            'message': self.message,
            'metrics': self.metrics,
            'model_info': self.model_info,
            'training_history': self.training_history,
        }


@dataclass
class PipelineResult:
    """
    Result object for pipeline execution.
    
    Contains results from all pipeline stages and overall execution status.
    """
    
    stage_results: Dict[str, Any] = field(default_factory=dict)
    success: bool = True
    message: str = ""
    execution_time: Optional[float] = None
    errors: List[str] = field(default_factory=list)
    
    def add_stage_result(self, stage_name: str, result: Any):
        """Add a stage result to the pipeline result."""
        self.stage_results[stage_name] = result
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            'stage_results': {
                k: v.to_dict() if hasattr(v, 'to_dict') else v
                for k, v in self.stage_results.items()
            },
            'success': self.success,
            'message': self.message,
            'execution_time': self.execution_time,
            'errors': self.errors,
        }

