# Utilities API

The `apa.utils` module provides essential utility functions for I/O operations, visualization, and metrics calculation, supporting the APA pipeline with comprehensive helper tools.

## 📦 Module Overview

```python
from apa.utils import IOUtils, VisualizationUtils, MetricsCalculator
```

The utilities module handles:
- Multi-format file I/O operations (YAML, JSON, HDF5, Pickle)
- Comprehensive plotting and visualization tools
- PCI-specific metrics calculation and model evaluation
- Data validation and quality assessment

## 🔧 Classes

### IOUtils

Handles file I/O operations for various data formats used in the APA pipeline.

#### Constructor

```python
IOUtils()
```

**Example:**
```python
from apa.utils.io import IOUtils

# Initialize I/O utilities
io_utils = IOUtils()
```

#### Methods

##### `read_yaml_config(config_path: str) -> Dict[str, Any]`

Read a YAML configuration file.

**Parameters:**
- `config_path` (str): Path to the YAML configuration file

**Returns:**
- `Dict[str, Any]`: Configuration dictionary

**Raises:**
- `FileNotFoundError`: If config file doesn't exist
- `yaml.YAMLError`: If config file is invalid YAML

**Example:**
```python
# Read YAML configuration
config = io_utils.read_yaml_config('configs/detroit.yaml')
print(f"Data source: {config['config']['data']['enum_data_source']}")
```

##### `write_yaml_config(config: Dict[str, Any], output_path: str) -> None`

Write a configuration dictionary to a YAML file.

**Parameters:**
- `config` (Dict[str, Any]): Configuration dictionary
- `output_path` (str): Path where to save the YAML file

**Example:**
```python
# Write YAML configuration
config = {'data': {'input_path': 'data/'}}
io_utils.write_yaml_config(config, 'output/config.yaml')
```

##### `read_json_file(json_path: str) -> Dict[str, Any]`

Read a JSON file.

**Parameters:**
- `json_path` (str): Path to the JSON file

**Returns:**
- `Dict[str, Any]`: JSON data as dictionary

**Example:**
```python
# Read JSON file
metadata = io_utils.read_json_file('data/metadata.json')
print(f"Metadata keys: {list(metadata.keys())}")
```

##### `write_json_file(data: Dict[str, Any], output_path: str) -> None`

Write data to a JSON file.

**Parameters:**
- `data` (Dict[str, Any]): Data dictionary to write
- `output_path` (str): Path where to save the JSON file

**Example:**
```python
# Write JSON file
data = {'results': {'accuracy': 0.95}}
io_utils.write_json_file(data, 'results.json')
```

##### `read_h5_file(h5_path: str, dataset_name: Optional[str] = None) -> Any`

Read data from an HDF5 file.

**Parameters:**
- `h5_path` (str): Path to the HDF5 file
- `dataset_name` (Optional[str]): Name of the dataset to read (if None, reads all datasets)

**Returns:**
- `Any`: Data from the HDF5 file

**Example:**
```python
# Read specific dataset
data = io_utils.read_h5_file('data.h5', 'images')

# Read all datasets
all_data = io_utils.read_h5_file('data.h5')
```

##### `write_h5_file(data: Any, output_path: str, dataset_name: str = "data") -> None`

Write data to an HDF5 file.

**Parameters:**
- `data` (Any): Data to write
- `output_path` (str): Path where to save the HDF5 file
- `dataset_name` (str): Name of the dataset in the HDF5 file

**Example:**
```python
# Write data to HDF5
import numpy as np
data = np.random.rand(100, 100, 12)
io_utils.write_h5_file(data, 'output.h5', 'hyperspectral_data')
```

##### `save_pickle_data(data: Any, output_path: str) -> None`

Save data using pickle.

**Parameters:**
- `data` (Any): Data to save
- `output_path` (str): Path where to save the pickle file

**Example:**
```python
# Save complex data structure
complex_data = {'model': model, 'results': results}
io_utils.save_pickle_data(complex_data, 'model_data.pkl')
```

##### `load_pickle_data(pickle_path: str) -> Any`

Load data from a pickle file.

**Parameters:**
- `pickle_path` (str): Path to the pickle file

**Returns:**
- `Any`: Loaded data

**Example:**
```python
# Load pickled data
loaded_data = io_utils.load_pickle_data('model_data.pkl')
model = loaded_data['model']
```

### VisualizationUtils

Provides comprehensive plotting and visualization tools for APA results and data analysis.

#### Constructor

```python
VisualizationUtils()
```

**Example:**
```python
from apa.utils.visualization import VisualizationUtils

# Initialize visualization utilities
viz_utils = VisualizationUtils()
```

#### Methods

##### `plot_spectral_curves(wavelengths: np.ndarray, stats: Dict[str, Any], title: Optional[str] = None, stat_type: str = 'mean') -> None`

Plot spectral curves for different segments.

**Parameters:**
- `wavelengths` (np.ndarray): Array of wavelengths
- `stats` (Dict[str, Any]): Dictionary containing statistics for different segments
- `title` (Optional[str]): Optional title for the plot
- `stat_type` (str): Type of statistic to plot ('mean', 'std', etc.)

**Example:**
```python
import numpy as np

# Plot spectral curves
wavelengths = np.linspace(400, 1000, 12)
stats = {'segment_1': {'mean': np.random.rand(12)}}
viz_utils.plot_spectral_curves(wavelengths, stats, 'Spectral Analysis')
```

##### `plot_roi_overview(roi_data: Dict[str, Any], save_path: Optional[str] = None) -> None`

Create an overview plot of a region of interest.

**Parameters:**
- `roi_data` (Dict[str, Any]): Dictionary containing ROI data
- `save_path` (Optional[str]): Optional path to save the plot

**Example:**
```python
# Create ROI overview plot
roi_data = {
    'roi': [-83.14294, -83.00007, 42.34429, 42.39170],
    'cropped_msp_img': msp_image,
    'segment_mask': segment_mask,
    'segID_PCI_LUT': pci_lut
}

viz_utils.plot_roi_overview(roi_data, save_path='roi_overview.png')
```

##### `plot_training_history(history: Dict[str, List[float]], save_path: Optional[str] = None) -> None`

Plot training history for model training.

**Parameters:**
- `history` (Dict[str, List[float]]): Dictionary containing training history
- `save_path` (Optional[str]): Optional path to save the plot

**Example:**
```python
# Plot training history
history = {
    'loss': [0.5, 0.3, 0.2, 0.1],
    'val_loss': [0.6, 0.4, 0.3, 0.2],
    'accuracy': [0.8, 0.9, 0.95, 0.98],
    'val_accuracy': [0.75, 0.85, 0.9, 0.95]
}

viz_utils.plot_training_history(history, save_path='training_history.png')
```

##### `create_results_report(results: Dict[str, Any], output_path: str) -> None`

Create a comprehensive results report.

**Parameters:**
- `results` (Dict[str, Any]): Dictionary containing pipeline results
- `output_path` (str): Path where to save the report

**Example:**
```python
# Create results report
results = {
    'processed_rois': roi_data_list,
    'config': config,
    'stages_completed': ['data_import', 'roi_processing']
}

viz_utils.create_results_report(results, 'apa_results_report.txt')
```

##### `plot_prediction_comparison(true_values: np.ndarray, predicted_values: np.ndarray, save_path: Optional[str] = None) -> None`

Plot comparison between true and predicted values.

**Parameters:**
- `true_values` (np.ndarray): Array of true values
- `predicted_values` (np.ndarray): Array of predicted values
- `save_path` (Optional[str]): Optional path to save the plot

**Example:**
```python
import numpy as np

# Plot prediction comparison
true_pci = np.array([85, 70, 55, 40, 25])
pred_pci = np.array([82, 68, 58, 42, 28])

viz_utils.plot_prediction_comparison(true_pci, pred_pci, save_path='prediction_comparison.png')
```

### MetricsCalculator

Calculates various metrics used in model evaluation and performance assessment.

#### Constructor

```python
MetricsCalculator()
```

**Example:**
```python
from apa.utils.metrics import MetricsCalculator

# Initialize metrics calculator
metrics_calc = MetricsCalculator()
```

#### Methods

##### `calculate_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, average: str = 'weighted') -> Dict[str, float]`

Calculate classification metrics.

**Parameters:**
- `y_true` (np.ndarray): True labels
- `y_pred` (np.ndarray): Predicted labels
- `average` (str): Averaging method for multi-class metrics

**Returns:**
- `Dict[str, float]`: Dictionary containing classification metrics

**Example:**
```python
import numpy as np

# Calculate classification metrics
y_true = np.array([0, 1, 2, 0, 1])
y_pred = np.array([0, 1, 1, 0, 1])

metrics = metrics_calc.calculate_classification_metrics(y_true, y_pred)
print(f"Accuracy: {metrics['accuracy']:.3f}")
print(f"F1 Score: {metrics['f1_score']:.3f}")
```

##### `calculate_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]`

Calculate regression metrics.

**Parameters:**
- `y_true` (np.ndarray): True values
- `y_pred` (np.ndarray): Predicted values

**Returns:**
- `Dict[str, float]`: Dictionary containing regression metrics

**Example:**
```python
# Calculate regression metrics
y_true = np.array([85.0, 70.0, 55.0, 40.0])
y_pred = np.array([82.0, 68.0, 58.0, 42.0])

metrics = metrics_calc.calculate_regression_metrics(y_true, y_pred)
print(f"MAE: {metrics['mae']:.3f}")
print(f"R² Score: {metrics['r2_score']:.3f}")
```

##### `calculate_pci_metrics(true_pci: np.ndarray, pred_pci: np.ndarray) -> Dict[str, float]`

Calculate PCI-specific metrics.

**Parameters:**
- `true_pci` (np.ndarray): True PCI values
- `pred_pci` (np.ndarray): Predicted PCI values

**Returns:**
- `Dict[str, float]`: Dictionary containing PCI-specific metrics

**Example:**
```python
# Calculate PCI-specific metrics
true_pci = np.array([85, 70, 55, 40, 25])
pred_pci = np.array([82, 68, 58, 42, 28])

pci_metrics = metrics_calc.calculate_pci_metrics(true_pci, pred_pci)
print(f"PCI MAE: {pci_metrics['pci_mae']:.3f}")
print(f"Accuracy within 5: {pci_metrics['pci_accuracy_within_5']:.1f}%")
```

##### `calculate_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, labels: Optional[List[str]] = None) -> np.ndarray`

Calculate confusion matrix.

**Parameters:**
- `y_true` (np.ndarray): True labels
- `y_pred` (np.ndarray): Predicted labels
- `labels` (Optional[List[str]]): Optional list of label names

**Returns:**
- `np.ndarray`: Confusion matrix

**Example:**
```python
# Calculate confusion matrix
y_true = np.array([0, 1, 2, 0, 1])
y_pred = np.array([0, 1, 1, 0, 1])
labels = ['Excellent', 'Good', 'Fair']

cm = metrics_calc.calculate_confusion_matrix(y_true, y_pred, labels)
print("Confusion Matrix:")
print(cm)
```

##### `generate_classification_report(y_true: np.ndarray, y_pred: np.ndarray, target_names: Optional[List[str]] = None) -> str`

Generate a detailed classification report.

**Parameters:**
- `y_true` (np.ndarray): True labels
- `y_pred` (np.ndarray): Predicted labels
- `target_names` (Optional[List[str]]): Optional list of target class names

**Returns:**
- `str`: Classification report as string

**Example:**
```python
# Generate classification report
y_true = np.array([0, 1, 2, 0, 1])
y_pred = np.array([0, 1, 1, 0, 1])
target_names = ['Excellent', 'Good', 'Fair']

report = metrics_calc.generate_classification_report(y_true, y_pred, target_names)
print(report)
```

## 🎯 Usage Examples

### Complete Data I/O Workflow

```python
from apa.utils.io import IOUtils
import numpy as np

def data_io_workflow():
    """Complete data I/O workflow example."""
    
    io_utils = IOUtils()
    
    # Read configuration
    config = io_utils.read_yaml_config('configs/detroit.yaml')
    
    # Read metadata
    metadata = io_utils.read_json_file('data/metadata.json')
    
    # Read hyperspectral data
    h5_data = io_utils.read_h5_file('data/hyperspectral.h5', 'images')
    
    # Process data
    processed_data = process_hyperspectral_data(h5_data)
    
    # Save processed data
    io_utils.write_h5_file(processed_data, 'output/processed_data.h5', 'processed_images')
    
    # Save results
    results = {'processed_shape': processed_data.shape, 'metadata': metadata}
    io_utils.write_json_file(results, 'output/results.json')
    
    return results

# Usage
results = data_io_workflow()
```

### Comprehensive Visualization

```python
from apa.utils.visualization import VisualizationUtils
import numpy as np

def comprehensive_visualization():
    """Create comprehensive visualizations."""
    
    viz_utils = VisualizationUtils()
    
    # Simulate ROI data
    roi_data = {
        'roi': [-83.14294, -83.00007, 42.34429, 42.39170],
        'cropped_msp_img': np.random.rand(100, 100, 12),
        'segment_mask': np.random.randint(0, 5, (100, 100)),
        'segID_PCI_LUT': {1: 85, 2: 70, 3: 55, 4: 40}
    }
    
    # Create ROI overview
    viz_utils.plot_roi_overview(roi_data, save_path='roi_overview.png')
    
    # Simulate training history
    history = {
        'loss': np.linspace(1.0, 0.1, 50),
        'val_loss': np.linspace(1.1, 0.15, 50),
        'accuracy': np.linspace(0.5, 0.95, 50),
        'val_accuracy': np.linspace(0.45, 0.9, 50)
    }
    
    # Plot training history
    viz_utils.plot_training_history(history, save_path='training_history.png')
    
    # Simulate prediction comparison
    true_pci = np.random.uniform(20, 100, 100)
    pred_pci = true_pci + np.random.normal(0, 5, 100)
    
    viz_utils.plot_prediction_comparison(true_pci, pred_pci, save_path='prediction_comparison.png')
    
    # Create results report
    results = {
        'processed_rois': [roi_data],
        'config': {'data': {'enum_data_source': 'VENUS_DETROIT'}},
        'stages_completed': ['data_import', 'roi_processing']
    }
    
    viz_utils.create_results_report(results, 'comprehensive_report.txt')

# Usage
comprehensive_visualization()
```

### Complete Metrics Analysis

```python
from apa.utils.metrics import MetricsCalculator
import numpy as np

def complete_metrics_analysis():
    """Perform complete metrics analysis."""
    
    metrics_calc = MetricsCalculator()
    
    # Simulate data
    np.random.seed(42)
    n_samples = 1000
    
    # True and predicted PCI values
    true_pci = np.random.uniform(20, 100, n_samples)
    pred_pci = true_pci + np.random.normal(0, 8, n_samples)
    
    # True and predicted condition classes
    true_conditions = np.digitize(true_pci, [25, 55, 70, 85])
    pred_conditions = np.digitize(pred_pci, [25, 55, 70, 85])
    
    # Calculate PCI-specific metrics
    pci_metrics = metrics_calc.calculate_pci_metrics(true_pci, pred_pci)
    print("PCI Metrics:")
    for metric, value in pci_metrics.items():
        print(f"  {metric}: {value:.3f}")
    
    # Calculate classification metrics
    class_metrics = metrics_calc.calculate_classification_metrics(true_conditions, pred_conditions)
    print("\nClassification Metrics:")
    for metric, value in class_metrics.items():
        print(f"  {metric}: {value:.3f}")
    
    # Calculate regression metrics
    reg_metrics = metrics_calc.calculate_regression_metrics(true_pci, pred_pci)
    print("\nRegression Metrics:")
    for metric, value in reg_metrics.items():
        print(f"  {metric}: {value:.3f}")
    
    # Generate classification report
    target_names = ['Failed', 'Poor', 'Fair', 'Good', 'Excellent']
    report = metrics_calc.generate_classification_report(true_conditions, pred_conditions, target_names)
    print(f"\nClassification Report:\n{report}")
    
    # Calculate confusion matrix
    cm = metrics_calc.calculate_confusion_matrix(true_conditions, pred_conditions, target_names)
    print(f"\nConfusion Matrix:\n{cm}")
    
    return {
        'pci_metrics': pci_metrics,
        'class_metrics': class_metrics,
        'reg_metrics': reg_metrics,
        'confusion_matrix': cm
    }

# Usage
metrics_results = complete_metrics_analysis()
```

## 🔧 Advanced Usage

### Custom Visualization Functions

```python
import matplotlib.pyplot as plt
from apa.utils.visualization import VisualizationUtils

class CustomVisualizationUtils(VisualizationUtils):
    """Extended visualization utilities with custom functions."""
    
    def plot_pci_distribution(self, pci_values: np.ndarray, save_path: Optional[str] = None):
        """Plot PCI value distribution."""
        plt.figure(figsize=(10, 6))
        
        # Histogram
        plt.subplot(1, 2, 1)
        plt.hist(pci_values, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        plt.title('PCI Distribution')
        plt.xlabel('PCI Score')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)
        
        # Box plot
        plt.subplot(1, 2, 2)
        plt.boxplot(pci_values)
        plt.title('PCI Box Plot')
        plt.ylabel('PCI Score')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_spectral_signatures(self, wavelengths: np.ndarray, signatures: Dict[str, np.ndarray], save_path: Optional[str] = None):
        """Plot spectral signatures for different materials."""
        plt.figure(figsize=(12, 8))
        
        colors = ['red', 'blue', 'green', 'orange', 'purple']
        
        for i, (material, spectrum) in enumerate(signatures.items()):
            plt.plot(wavelengths, spectrum, label=material, color=colors[i % len(colors)], linewidth=2)
        
        plt.xlabel('Wavelength (nm)')
        plt.ylabel('Reflectance')
        plt.title('Spectral Signatures')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()

# Usage
custom_viz = CustomVisualizationUtils()

# Plot PCI distribution
pci_values = np.random.uniform(20, 100, 500)
custom_viz.plot_pci_distribution(pci_values, save_path='pci_distribution.png')

# Plot spectral signatures
wavelengths = np.linspace(400, 1000, 12)
signatures = {
    'Asphalt': np.random.rand(12) * 0.3 + 0.1,
    'Concrete': np.random.rand(12) * 0.4 + 0.2,
    'Vegetation': np.random.rand(12) * 0.5 + 0.3
}
custom_viz.plot_spectral_signatures(wavelengths, signatures, save_path='spectral_signatures.png')
```

### Batch Metrics Calculation

```python
def batch_metrics_calculation(true_values_list: List[np.ndarray], pred_values_list: List[np.ndarray]):
    """Calculate metrics for multiple datasets."""
    
    metrics_calc = MetricsCalculator()
    all_metrics = []
    
    for i, (true_vals, pred_vals) in enumerate(zip(true_values_list, pred_values_list)):
        print(f"Calculating metrics for dataset {i+1}...")
        
        # Calculate all metrics
        pci_metrics = metrics_calc.calculate_pci_metrics(true_vals, pred_vals)
        reg_metrics = metrics_calc.calculate_regression_metrics(true_vals, pred_vals)
        
        # Combine metrics
        combined_metrics = {**pci_metrics, **reg_metrics}
        combined_metrics['dataset_id'] = i + 1
        
        all_metrics.append(combined_metrics)
    
    return all_metrics

# Usage
true_values_list = [
    np.random.uniform(20, 100, 100),
    np.random.uniform(20, 100, 150),
    np.random.uniform(20, 100, 200)
]

pred_values_list = [
    true_values_list[0] + np.random.normal(0, 5, 100),
    true_values_list[1] + np.random.normal(0, 5, 150),
    true_values_list[2] + np.random.normal(0, 5, 200)
]

batch_metrics = batch_metrics_calculation(true_values_list, pred_values_list)

# Print summary
for metrics in batch_metrics:
    print(f"Dataset {metrics['dataset_id']}: MAE={metrics['mae']:.3f}, R²={metrics['r2_score']:.3f}")
```

## 🚨 Error Handling

### Common Utility Errors

1. **File I/O Errors**
   ```python
   # Error: File not found
   FileNotFoundError: Configuration file not found: configs/missing.yaml
   ```

2. **Data Format Errors**
   ```python
   # Error: Invalid data format
   ValueError: Invalid YAML syntax in config.yaml: while parsing a block mapping
   ```

3. **Visualization Errors**
   ```python
   # Error: Invalid data for plotting
   ValueError: Data arrays must have the same length for plotting
   ```

### Error Handling Best Practices

```python
def safe_utility_operations():
    """Safely perform utility operations with error handling."""
    
    io_utils = IOUtils()
    viz_utils = VisualizationUtils()
    metrics_calc = MetricsCalculator()
    
    try:
        # Safe file operations
        config = io_utils.read_yaml_config('configs/detroit.yaml')
        
        # Safe visualization
        roi_data = load_roi_data()
        viz_utils.plot_roi_overview(roi_data)
        
        # Safe metrics calculation
        true_pci, pred_pci = load_pci_data()
        metrics = metrics_calc.calculate_pci_metrics(true_pci, pred_pci)
        
        return config, metrics
        
    except FileNotFoundError as e:
        print(f"✗ File not found: {e}")
        return None, None
    except ValueError as e:
        print(f"✗ Data error: {e}")
        return None, None
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        return None, None

# Usage
config, metrics = safe_utility_operations()
```

## 🔗 Related Documentation

- [Main API Documentation](../index.md)
- [Configuration API](../config/)
- [Data Management API](../data/)
- [Pipeline API](../pipeline/)
