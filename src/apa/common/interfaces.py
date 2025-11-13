"""
Standardized interfaces for APA modules.

Defines the contract that all modules must follow, ensuring
consistency and interoperability across the pipeline.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from apa.common.data_structures import DataContainer, ProcessingResult, ModelResult


class DataInterface(ABC):
    """Interface for data import and validation modules."""
    
    @abstractmethod
    def load_data(self, config: Dict[str, Any]) -> DataContainer:
        """
        Load data from source.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            DataContainer with loaded data
        """
        pass
    
    @abstractmethod
    def validate_data(self, data: DataContainer) -> bool:
        """
        Validate loaded data.
        
        Args:
            data: DataContainer to validate
            
        Returns:
            True if valid, raises ValidationError otherwise
        """
        pass


class ProcessingInterface(ABC):
    """Interface for data processing modules."""
    
    @abstractmethod
    def process_data(self, data: DataContainer, config: Dict[str, Any]) -> ProcessingResult:
        """
        Process input data.
        
        Args:
            data: Input data container
            config: Processing configuration
            
        Returns:
            ProcessingResult with processed data
        """
        pass
    
    @abstractmethod
    def validate_input(self, data: DataContainer) -> bool:
        """
        Validate input data.
        
        Args:
            data: Input data container
            
        Returns:
            True if valid, raises ValidationError otherwise
        """
        pass


class ModelInterface(ABC):
    """Interface for machine learning models."""
    
    @abstractmethod
    def train(self, data: DataContainer, config: Dict[str, Any]) -> ModelResult:
        """
        Train the model.
        
        Args:
            data: Training data container
            config: Training configuration
            
        Returns:
            ModelResult with training metrics
        """
        pass
    
    @abstractmethod
    def predict(self, data: DataContainer) -> ModelResult:
        """
        Make predictions.
        
        Args:
            data: Input data container
            
        Returns:
            ModelResult with predictions
        """
        pass
    
    @abstractmethod
    def save_model(self, path: str) -> bool:
        """
        Save model to disk.
        
        Args:
            path: Path to save model
            
        Returns:
            True if successful
        """
        pass
    
    @abstractmethod
    def load_model(self, path: str) -> bool:
        """
        Load model from disk.
        
        Args:
            path: Path to load model from
            
        Returns:
            True if successful
        """
        pass


class PipelineInterface(ABC):
    """Interface for pipeline orchestration."""
    
    @abstractmethod
    def run_pipeline(self, data: Optional[DataContainer], config: Dict[str, Any]) -> Any:
        """
        Run the complete pipeline.
        
        Args:
            data: Optional input data container
            config: Pipeline configuration
            
        Returns:
            Pipeline result
        """
        pass
    
    @abstractmethod
    def add_stage(self, name: str, stage: Any, dependencies: Optional[list] = None):
        """
        Add a stage to the pipeline.
        
        Args:
            name: Stage name
            stage: Stage implementation
            dependencies: List of stage names this stage depends on
        """
        pass

