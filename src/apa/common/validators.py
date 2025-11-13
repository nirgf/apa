"""
Validation utilities for APA modules.

Provides input/output validation and configuration validation
to ensure data integrity throughout the pipeline.
"""

from typing import Any, Dict, List, Optional
import numpy as np

from apa.common.data_structures import DataContainer
from apa.common.exceptions import ValidationError


class InputValidator:
    """Validates input data for modules."""
    
    @staticmethod
    def validate_data_container(data: Any) -> bool:
        """
        Validate that input is a DataContainer.
        
        Args:
            data: Data to validate
            
        Returns:
            True if valid
            
        Raises:
            ValidationError if invalid
        """
        if not isinstance(data, DataContainer):
            raise ValidationError(
                f"Expected DataContainer, got {type(data).__name__}",
                {'expected': 'DataContainer', 'got': type(data).__name__}
            )
        return True
    
    @staticmethod
    def validate_not_none(data: Any, name: str = "data") -> bool:
        """
        Validate that data is not None.
        
        Args:
            data: Data to validate
            name: Name of the data (for error messages)
            
        Returns:
            True if valid
            
        Raises:
            ValidationError if invalid
        """
        if data is None:
            raise ValidationError(f"{name} cannot be None")
        return True
    
    @staticmethod
    def validate_array_shape(array: np.ndarray, expected_shape: tuple, name: str = "array") -> bool:
        """
        Validate array shape.
        
        Args:
            array: Array to validate
            expected_shape: Expected shape (use -1 for any dimension)
            name: Name of the array (for error messages)
            
        Returns:
            True if valid
            
        Raises:
            ValidationError if invalid
        """
        if not isinstance(array, np.ndarray):
            raise ValidationError(f"{name} must be a numpy array")
        
        if len(array.shape) != len(expected_shape):
            raise ValidationError(
                f"{name} has wrong number of dimensions: "
                f"expected {len(expected_shape)}, got {len(array.shape)}"
            )
        
        for i, (expected, actual) in enumerate(zip(expected_shape, array.shape)):
            if expected != -1 and expected != actual:
                raise ValidationError(
                    f"{name} dimension {i} mismatch: expected {expected}, got {actual}"
                )
        
        return True


class OutputValidator:
    """Validates output data from modules."""
    
    @staticmethod
    def validate_processing_result(result: Any) -> bool:
        """
        Validate processing result.
        
        Args:
            result: Result to validate
            
        Returns:
            True if valid
            
        Raises:
            ValidationError if invalid
        """
        from apa.common.data_structures import ProcessingResult
        
        if not isinstance(result, ProcessingResult):
            raise ValidationError(
                f"Expected ProcessingResult, got {type(result).__name__}"
            )
        
        if result.data is None:
            raise ValidationError("ProcessingResult.data cannot be None")
        
        return True
    
    @staticmethod
    def validate_model_result(result: Any) -> bool:
        """
        Validate model result.
        
        Args:
            result: Result to validate
            
        Returns:
            True if valid
            
        Raises:
            ValidationError if invalid
        """
        from apa.common.data_structures import ModelResult
        
        if not isinstance(result, ModelResult):
            raise ValidationError(
                f"Expected ModelResult, got {type(result).__name__}"
            )
        
        return True


class ConfigValidator:
    """Validates configuration dictionaries."""
    
    @staticmethod
    def validate_required_keys(config: Dict[str, Any], required_keys: List[str], context: str = "config") -> bool:
        """
        Validate that all required keys are present.
        
        Args:
            config: Configuration dictionary
            required_keys: List of required keys
            context: Context for error messages
            
        Returns:
            True if valid
            
        Raises:
            ValidationError if invalid
        """
        missing_keys = [key for key in required_keys if key not in config]
        if missing_keys:
            raise ValidationError(
                f"Missing required {context} keys: {missing_keys}",
                {'missing_keys': missing_keys, 'context': context}
            )
        return True
    
    @staticmethod
    def validate_key_type(config: Dict[str, Any], key: str, expected_type: type, context: str = "config") -> bool:
        """
        Validate that a key has the expected type.
        
        Args:
            config: Configuration dictionary
            key: Key to validate
            expected_type: Expected type
            context: Context for error messages
            
        Returns:
            True if valid
            
        Raises:
            ValidationError if invalid
        """
        if key not in config:
            return True  # Optional key
        
        if not isinstance(config[key], expected_type):
            raise ValidationError(
                f"{context}[{key}] must be of type {expected_type.__name__}, "
                f"got {type(config[key]).__name__}"
            )
        
        return True
    
    @staticmethod
    def validate_value_range(value: Any, min_val: Optional[float] = None, 
                            max_val: Optional[float] = None, name: str = "value") -> bool:
        """
        Validate that a value is within a range.
        
        Args:
            value: Value to validate
            min_val: Minimum value (inclusive)
            max_val: Maximum value (inclusive)
            name: Name of the value (for error messages)
            
        Returns:
            True if valid
            
        Raises:
            ValidationError if invalid
        """
        if not isinstance(value, (int, float)):
            raise ValidationError(f"{name} must be numeric")
        
        if min_val is not None and value < min_val:
            raise ValidationError(f"{name} must be >= {min_val}, got {value}")
        
        if max_val is not None and value > max_val:
            raise ValidationError(f"{name} must be <= {max_val}, got {value}")
        
        return True

