"""
Custom exception classes for APA.

Provides specific exception types for different error scenarios
to enable better error handling and debugging.
"""


class APAException(Exception):
    """Base exception class for all APA-related errors."""
    
    def __init__(self, message: str, details: dict = None):
        """
        Initialize APA exception.
        
        Args:
            message: Error message
            details: Additional error details
        """
        super().__init__(message)
        self.message = message
        self.details = details or {}
    
    def __str__(self):
        if self.details:
            return f"{self.message} | Details: {self.details}"
        return self.message


class ValidationError(APAException):
    """Raised when data or configuration validation fails."""
    pass


class ProcessingError(APAException):
    """Raised when data processing operations fail."""
    pass


class ModelError(APAException):
    """Raised when model operations (training, prediction) fail."""
    pass


class DataError(APAException):
    """Raised when data operations (import, export) fail."""
    pass


class ConfigurationError(APAException):
    """Raised when configuration is invalid or missing."""
    pass


class PipelineError(APAException):
    """Raised when pipeline execution fails."""
    pass

