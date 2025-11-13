"""
Abstract base classes for APA modules.

Provides base implementations that modules can inherit from,
reducing boilerplate and ensuring consistent behavior.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, List
import time
import logging

from apa.common.data_structures import (
    DataContainer,
    ProcessingResult,
    ModelResult,
    PipelineResult,
)
from apa.common.exceptions import (
    ValidationError,
    ProcessingError,
    ConfigurationError,
)
from apa.common.interfaces import (
    DataInterface,
    ProcessingInterface,
    ModelInterface,
    PipelineInterface,
)


logger = logging.getLogger(__name__)


class BaseModule(ABC):
    """
    Base class for all APA modules.
    
    Provides common functionality like configuration management,
    logging, and validation.
    """
    
    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        """
        Initialize base module.
        
        Args:
            name: Module name
            config: Module configuration
        """
        self.name = name
        self.config = config or {}
        self.logger = logging.getLogger(f"apa.{name}")
        self._validate_config()
    
    def _validate_config(self):
        """Validate module configuration."""
        required_keys = getattr(self, 'required_config_keys', [])
        missing_keys = [key for key in required_keys if key not in self.config]
        if missing_keys:
            raise ConfigurationError(
                f"Missing required configuration keys: {missing_keys}",
                {'module': self.name, 'missing_keys': missing_keys}
            )
    
    def get_config(self, key: str, default: Any = None) -> Any:
        """Get configuration value."""
        return self.config.get(key, default)
    
    def log_info(self, message: str):
        """Log info message."""
        self.logger.info(message)
    
    def log_error(self, message: str, exc_info: bool = False):
        """Log error message."""
        self.logger.error(message, exc_info=exc_info)
    
    def log_warning(self, message: str):
        """Log warning message."""
        self.logger.warning(message)


class BaseDataProcessor(BaseModule, ProcessingInterface):
    """
    Base class for data processing modules.
    
    Provides common functionality for data processing operations.
    """
    
    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        """
        Initialize data processor.
        
        Args:
            name: Processor name
            config: Processor configuration
        """
        super().__init__(name, config)
        self.supported_data_types = getattr(self, 'supported_data_types', [])
    
    def process_data(self, data: DataContainer, config: Optional[Dict[str, Any]] = None) -> ProcessingResult:
        """
        Process input data (public interface).
        
        Args:
            data: Input data container
            config: Optional processing configuration (overrides instance config)
            
        Returns:
            ProcessingResult with processed data
        """
        start_time = time.time()
        
        try:
            # Validate input
            self.validate_input(data)
            
            # Merge configs
            processing_config = {**self.config, **(config or {})}
            
            # Process data
            processed_data = self._process_impl(data, processing_config)
            
            # Validate output
            if not isinstance(processed_data, DataContainer):
                processed_data = DataContainer(
                    data=processed_data,
                    metadata={'processor': self.name},
                    data_type=data.data_type
                )
            
            processing_time = time.time() - start_time
            
            return ProcessingResult(
                data=processed_data,
                success=True,
                message=f"Processing completed successfully",
                processing_time=processing_time
            )
        
        except Exception as e:
            processing_time = time.time() - start_time
            self.log_error(f"Processing failed: {str(e)}", exc_info=True)
            raise ProcessingError(
                f"Processing failed in {self.name}: {str(e)}",
                {'module': self.name, 'processing_time': processing_time}
            ) from e
    
    def validate_input(self, data: DataContainer) -> bool:
        """
        Validate input data.
        
        Args:
            data: Input data container
            
        Returns:
            True if valid
            
        Raises:
            ValidationError if invalid
        """
        if not isinstance(data, DataContainer):
            raise ValidationError(f"Expected DataContainer, got {type(data)}")
        
        if data.data is None:
            raise ValidationError("Input data cannot be None")
        
        if self.supported_data_types and data.data_type not in self.supported_data_types:
            raise ValidationError(
                f"Unsupported data type: {data.data_type}. "
                f"Supported types: {self.supported_data_types}"
            )
        
        return True
    
    @abstractmethod
    def _process_impl(self, data: DataContainer, config: Dict[str, Any]) -> DataContainer:
        """
        Implementation of processing logic (to be overridden by subclasses).
        
        Args:
            data: Input data container
            config: Processing configuration
            
        Returns:
            Processed data container
        """
        pass


class BaseModel(BaseModule, ModelInterface):
    """
    Base class for machine learning models.
    
    Provides common functionality for model operations.
    """
    
    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        """
        Initialize model.
        
        Args:
            name: Model name
            config: Model configuration
        """
        super().__init__(name, config)
        self.model = None
        self.is_trained = False
    
    def train(self, data: DataContainer, config: Optional[Dict[str, Any]] = None) -> ModelResult:
        """
        Train the model (public interface).
        
        Args:
            data: Training data container
            config: Optional training configuration
            
        Returns:
            ModelResult with training metrics
        """
        start_time = time.time()
        
        try:
            training_config = {**self.config, **(config or {})}
            metrics, history = self._train_impl(data, training_config)
            
            self.is_trained = True
            training_time = time.time() - start_time
            
            return ModelResult(
                success=True,
                message="Training completed successfully",
                metrics=metrics,
                training_history=history,
                model_info={'training_time': training_time}
            )
        
        except Exception as e:
            self.log_error(f"Training failed: {str(e)}", exc_info=True)
            raise ProcessingError(f"Training failed in {self.name}: {str(e)}") from e
    
    def predict(self, data: DataContainer) -> ModelResult:
        """
        Make predictions (public interface).
        
        Args:
            data: Input data container
            
        Returns:
            ModelResult with predictions
        """
        if not self.is_trained and self.model is None:
            raise ProcessingError("Model must be trained or loaded before prediction")
        
        try:
            predictions, probabilities = self._predict_impl(data)
            
            return ModelResult(
                predictions=predictions,
                probabilities=probabilities,
                success=True,
                message="Prediction completed successfully"
            )
        
        except Exception as e:
            self.log_error(f"Prediction failed: {str(e)}", exc_info=True)
            raise ProcessingError(f"Prediction failed in {self.name}: {str(e)}") from e
    
    def save_model(self, path: str) -> bool:
        """Save model to disk."""
        # To be implemented by subclasses
        raise NotImplementedError("Subclasses must implement save_model")
    
    def load_model(self, path: str) -> bool:
        """Load model from disk."""
        # To be implemented by subclasses
        raise NotImplementedError("Subclasses must implement load_model")
    
    @abstractmethod
    def _train_impl(self, data: DataContainer, config: Dict[str, Any]) -> tuple:
        """
        Implementation of training logic (to be overridden by subclasses).
        
        Args:
            data: Training data container
            config: Training configuration
            
        Returns:
            Tuple of (metrics dict, training history dict)
        """
        pass
    
    @abstractmethod
    def _predict_impl(self, data: DataContainer) -> tuple:
        """
        Implementation of prediction logic (to be overridden by subclasses).
        
        Args:
            data: Input data container
            
        Returns:
            Tuple of (predictions, probabilities)
        """
        pass


class BasePipeline(BaseModule, PipelineInterface):
    """
    Base class for pipeline orchestration.
    
    Provides common functionality for pipeline execution.
    """
    
    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        """
        Initialize pipeline.
        
        Args:
            name: Pipeline name
            config: Pipeline configuration
        """
        super().__init__(name, config)
        self.stages: Dict[str, Any] = {}
        self.stage_dependencies: Dict[str, List[str]] = {}
    
    def add_stage(self, name: str, stage: Any, dependencies: Optional[List[str]] = None):
        """
        Add a stage to the pipeline.
        
        Args:
            name: Stage name
            stage: Stage implementation
            dependencies: List of stage names this stage depends on
        """
        self.stages[name] = stage
        self.stage_dependencies[name] = dependencies or []
        self.log_info(f"Added stage '{name}' with dependencies: {dependencies or []}")
    
    def run_pipeline(self, data: Optional[DataContainer], config: Optional[Dict[str, Any]] = None) -> PipelineResult:
        """
        Run the complete pipeline (public interface).
        
        Args:
            data: Optional input data container
            config: Optional pipeline configuration
            
        Returns:
            PipelineResult with all stage results
        """
        start_time = time.time()
        pipeline_config = {**self.config, **(config or {})}
        result = PipelineResult()
        
        try:
            # Resolve execution order based on dependencies
            execution_order = self._resolve_execution_order()
            
            current_data = data
            
            # Execute stages in order
            for stage_name in execution_order:
                self.log_info(f"Executing stage: {stage_name}")
                stage = self.stages[stage_name]
                
                try:
                    if hasattr(stage, 'process_data'):
                        stage_result = stage.process_data(current_data, pipeline_config)
                        if hasattr(stage_result, 'data'):
                            current_data = stage_result.data
                        result.add_stage_result(stage_name, stage_result)
                    elif hasattr(stage, 'load_data'):
                        current_data = stage.load_data(pipeline_config)
                        result.add_stage_result(stage_name, ProcessingResult(data=current_data))
                    else:
                        self.log_warning(f"Stage {stage_name} has no recognized interface")
                
                except Exception as e:
                    error_msg = f"Stage {stage_name} failed: {str(e)}"
                    result.errors.append(error_msg)
                    self.log_error(error_msg, exc_info=True)
                    if pipeline_config.get('stop_on_error', True):
                        raise ProcessingError(error_msg) from e
            
            result.success = len(result.errors) == 0
            result.message = "Pipeline completed successfully" if result.success else "Pipeline completed with errors"
            result.execution_time = time.time() - start_time
            
            return result
        
        except Exception as e:
            result.success = False
            result.message = f"Pipeline failed: {str(e)}"
            result.execution_time = time.time() - start_time
            self.log_error(f"Pipeline execution failed: {str(e)}", exc_info=True)
            raise ProcessingError(f"Pipeline {self.name} failed: {str(e)}") from e
    
    def _resolve_execution_order(self) -> List[str]:
        """
        Resolve execution order based on dependencies (topological sort).
        
        Returns:
            List of stage names in execution order
        """
        # Simple topological sort
        visited = set()
        order = []
        
        def visit(stage_name: str):
            if stage_name in visited:
                return
            visited.add(stage_name)
            for dep in self.stage_dependencies.get(stage_name, []):
                if dep in self.stages:
                    visit(dep)
            order.append(stage_name)
        
        for stage_name in self.stages:
            visit(stage_name)
        
        return order

