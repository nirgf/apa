"""
Metrics calculation utilities for APA.

Provides performance metrics calculation for models and predictions.
"""

from typing import Any, Dict, Optional, List
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    confusion_matrix,
    classification_report,
)


class MetricsCalculator:
    """Utilities for calculating performance metrics."""
    
    @staticmethod
    def calculate_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Calculate classification metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            Dictionary of metrics
        """
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'f1_score': f1_score(y_true, y_pred, average='weighted', zero_division=0),
        }
        
        return metrics
    
    @staticmethod
    def calculate_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Calculate regression metrics.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            Dictionary of metrics
        """
        metrics = {
            'mae': mean_absolute_error(y_true, y_pred),
            'mse': mean_squared_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'r2': r2_score(y_true, y_pred),
        }
        
        return metrics
    
    @staticmethod
    def calculate_pci_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
        """
        Calculate PCI-specific metrics.
        
        Args:
            y_true: True PCI values
            y_pred: Predicted PCI values
            
        Returns:
            Dictionary of PCI metrics
        """
        # Flatten arrays for calculation
        y_true_flat = y_true.flatten()
        y_pred_flat = y_pred.flatten()
        
        # Classification metrics (treating PCI as classes)
        classification_metrics = MetricsCalculator.calculate_classification_metrics(
            y_true_flat, y_pred_flat
        )
        
        # Regression metrics (treating PCI as continuous)
        regression_metrics = MetricsCalculator.calculate_regression_metrics(
            y_true_flat, y_pred_flat
        )
        
        # PCI-specific: percentage within tolerance
        tolerance = 1
        within_tolerance = np.abs(y_true_flat - y_pred_flat) <= tolerance
        pci_accuracy = np.mean(within_tolerance)
        
        metrics = {
            **classification_metrics,
            **regression_metrics,
            'pci_accuracy_within_1': pci_accuracy,
        }
        
        return metrics
    
    @staticmethod
    def calculate_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray,
                                  labels: Optional[List[int]] = None) -> np.ndarray:
        """
        Calculate confusion matrix.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            labels: Optional list of label values
            
        Returns:
            Confusion matrix
        """
        return confusion_matrix(y_true.flatten(), y_pred.flatten(), labels=labels)
    
    @staticmethod
    def generate_classification_report(y_true: np.ndarray, y_pred: np.ndarray,
                                      labels: Optional[List[int]] = None) -> str:
        """
        Generate classification report.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            labels: Optional list of label values
            
        Returns:
            Classification report string
        """
        return classification_report(
            y_true.flatten(),
            y_pred.flatten(),
            labels=labels,
            zero_division=0
        )

