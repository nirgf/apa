"""
Visualization utilities for APA.

Provides plotting and visualization tools for data and results.
"""

from typing import Any, Dict, Optional, List
import matplotlib.pyplot as plt
import numpy as np


class VisualizationUtils:
    """Utilities for visualization and plotting."""
    
    @staticmethod
    def plot_roi_overview(data: np.ndarray, roi_bounds: Optional[List[float]] = None,
                          title: str = "ROI Overview", save_path: Optional[str] = None):
        """
        Plot ROI overview.
        
        Args:
            data: Image data
            roi_bounds: Optional ROI bounds
            title: Plot title
            save_path: Optional path to save plot
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        if len(data.shape) == 3:
            # RGB or multispectral - show first 3 bands
            display_data = data[:, :, :3] if data.shape[2] >= 3 else data[:, :, 0]
            ax.imshow(display_data)
        else:
            ax.imshow(data, cmap='gray')
        
        ax.set_title(title)
        ax.axis('off')
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        else:
            plt.show()
        
        plt.close()
    
    @staticmethod
    def plot_training_history(history: Dict[str, List[float]], save_path: Optional[str] = None):
        """
        Plot training history.
        
        Args:
            history: Training history dictionary
            save_path: Optional path to save plot
        """
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # Plot loss
        if 'loss' in history:
            axes[0].plot(history['loss'], label='Training Loss')
        if 'val_loss' in history:
            axes[0].plot(history['val_loss'], label='Validation Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training Loss')
        axes[0].legend()
        axes[0].grid(True)
        
        # Plot accuracy
        if 'accuracy' in history:
            axes[1].plot(history['accuracy'], label='Training Accuracy')
        if 'val_accuracy' in history:
            axes[1].plot(history['val_accuracy'], label='Validation Accuracy')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].set_title('Training Accuracy')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        else:
            plt.show()
        
        plt.close()
    
    @staticmethod
    def plot_predictions(ground_truth: np.ndarray, predictions: np.ndarray,
                        save_path: Optional[str] = None):
        """
        Plot predictions vs ground truth.
        
        Args:
            ground_truth: Ground truth data
            predictions: Predicted data
            save_path: Optional path to save plot
        """
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        axes[0].imshow(ground_truth, cmap='viridis')
        axes[0].set_title('Ground Truth')
        axes[0].axis('off')
        
        axes[1].imshow(predictions, cmap='viridis')
        axes[1].set_title('Predictions')
        axes[1].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        else:
            plt.show()
        
        plt.close()

