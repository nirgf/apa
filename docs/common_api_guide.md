# APA Common API Guide

This guide explains how to use the APA Common API to create modular, independent pipeline stages that can be easily composed and reused.

## Overview

The APA Common API provides standardized interfaces and base classes that ensure consistency across all APA modules. This enables:

- **Modularity**: Each stage is independent and can be used separately
- **Composability**: Stages can be easily combined into custom pipelines
- **Consistency**: All modules follow the same interface patterns
- **Reusability**: Modules can be reused across different projects
- **Validation**: Built-in input/output validation and error handling

## Architecture

### Core Components

1. **Interfaces** (`apa.common.interfaces`)
   - `DataInterface`: For data import and validation modules
   - `ProcessingInterface`: For data processing modules
   - `ModelInterface`: For machine learning models
   - `PipelineInterface`: For pipeline orchestration

2. **Base Classes** (`apa.common.base_classes`)
   - `BaseModule`: Base class for all modules
   - `BaseDataProcessor`: For data processing modules
   - `BaseModel`: For machine learning models
   - `BasePipeline`: For pipeline orchestration

3. **Data Structures** (`apa.common.data_structures`)
   - `DataContainer`: Standardized data container
   - `ProcessingResult`: Processing operation results
   - `ModelResult`: Model operation results
   - `PipelineResult`: Pipeline execution results

4. **Validation** (`apa.common.validators`)
   - `InputValidator`: Input data validation
   - `OutputValidator`: Output data validation
   - `ConfigValidator`: Configuration validation

5. **Error Handling** (`apa.common.exceptions`)
   - Custom exception classes for different error types
   - Consistent error reporting across modules

## Quick Start

### Basic Usage

```python
from apa.common import DataContainer
from apa.modules import HyperspectralDataImporter, ROIProcessor, APAPipeline

# Create a data importer
importer = HyperspectralDataImporter({
    'input_path': 'data/hyperspectral',
    'filename_NED': 'NED.h5',
    'filename_RGB': 'RGB.h5'
})

# Load data
data = importer.load_data({
    'input_path': 'data/hyperspectral',
    'filename_NED': 'NED.h5',
    'filename_RGB': 'RGB.h5'
})

# Process data
processor = ROIProcessor()
result = processor.process_data(data, {
    'roi_bounds': [42.3, 42.4, -83.0, -82.9]
})

# Use in pipeline
pipeline = APAPipeline()
pipeline_result = pipeline.run_pipeline(data, config)
```

### Creating Custom Modules

```python
from apa.common.base_classes import BaseDataProcessor
from apa.common.data_structures import DataContainer, ProcessingResult

class CustomProcessor(BaseDataProcessor):
    def __init__(self, config=None):
        super().__init__("custom_processor", config)
        self.supported_data_types = ['hyperspectral']
        self.required_config_keys = ['parameter1']
    
    def load_data(self, source):
        # Implementation for loading data
        pass
    
    def validate_data(self, data):
        # Implementation for validating data
        return True
    
    def get_data_info(self, data):
        # Implementation for getting data info
        return data.get_info()
    
    def _process_impl(self, data, config):
        # Implementation of actual processing
        # Return processed DataContainer
        pass
```

## Data Flow

### Standardized Data Container

All data in the APA pipeline uses the `DataContainer` class:

```python
data = DataContainer(
    data={
        'hyperspectral_image': np.array(...),
        'longitude_matrix': np.array(...),
        'latitude_matrix': np.array(...)
    },
    data_type='hyperspectral',
    metadata={'source': 'file_path', 'roi_bounds': [...]},
    source='data/source.h5'
)
```

### Processing Results

All processing operations return `ProcessingResult`:

```python
result = processor.process_data(data, config)

if result.success:
    processed_data = result.processed_data
    processing_time = result.processing_time
    processing_info = result.processing_info
else:
    error_message = result.error_message
```

## Module Types

### 1. Data Importers

Import data from various sources:

```python
from apa.modules import HyperspectralDataImporter, GroundTruthDataImporter, RoadDataImporter

# Hyperspectral imagery
importer = HyperspectralDataImporter(config)
data = importer.load_data(source)

# Ground truth PCI data
gt_importer = GroundTruthDataImporter(config)
gt_data = gt_importer.load_data(excel_path)

# Road network data
road_importer = RoadDataImporter(config)
road_data = road_importer.load_data(bbox)
```

### 2. Data Processors

Process and transform data:

```python
from apa.modules import ROIProcessor, RoadExtractor, PCISegmenter, DataPreprocessor

# ROI processing
roi_processor = ROIProcessor()
result = roi_processor.process_data(data, {
    'roi_bounds': [lat_min, lat_max, lon_min, lon_max]
})

# Road extraction
road_extractor = RoadExtractor()
result = road_extractor.process_data(data, {})

# PCI segmentation
pci_segmenter = PCISegmenter()
result = pci_segmenter.process_data(data, {})

# Data preprocessing
preprocessor = DataPreprocessor()
result = preprocessor.process_data(data, {
    'crop_size': 32,
    'overlap': 0.1
})
```

### 3. Machine Learning Models

Train and use ML models:

```python
from apa.modules import UNetModel, CNNModel, ModelManager

# U-Net model
unet = UNetModel({
    'input_size': (32, 32, 12),
    'n_classes': 4,
    'epochs': 100,
    'batch_size': 32
})

# Train model
train_result = unet.train(data, config)

# Make predictions
pred_result = unet.predict(data)

# Model manager
manager = ModelManager()
manager.add_model('unet', unet)
manager.set_active_model('unet')
result = manager.predict(data)
```

### 4. Pipelines

Orchestrate multiple stages:

```python
from apa.modules import APAPipeline, ModularPipeline

# Pre-built APA pipeline
pipeline = APAPipeline(config)
result = pipeline.run_pipeline(data, config)

# Custom modular pipeline
modular_pipeline = ModularPipeline("custom_pipeline")

# Add stages
modular_pipeline.add_custom_stage('roi', ROIProcessor())
modular_pipeline.add_custom_stage('road', RoadExtractor(), dependencies=['roi'])

# Run pipeline
result = modular_pipeline.run_pipeline(data, config)
```

## Configuration

### Standard Configuration Format

All modules use a standardized configuration format:

```python
config = {
    'data_import': {
        'input_path': 'data/hyperspectral',
        'filename_NED': 'NED.h5',
        'filename_RGB': 'RGB.h5'
    },
    'roi_processing': {
        'roi_bounds': [42.3, 42.4, -83.0, -82.9]
    },
    'road_extraction': {},
    'pci_segmentation': {},
    'data_preparation': {
        'crop_size': 32,
        'overlap': 0.1
    },
    'model_training': {
        'input_size': (32, 32, 12),
        'n_classes': 4,
        'epochs': 100,
        'batch_size': 32,
        'learning_rate': 1e-3
    }
}
```

### Configuration Validation

All modules validate their configuration:

```python
from apa.common.validators import ConfigValidator

# Validate specific configuration
ConfigValidator.validate_data_config(config['data_import'])
ConfigValidator.validate_model_config(config['model_training'])
ConfigValidator.validate_processing_config(config['data_preparation'])
```

## Error Handling

### Exception Types

The API provides specific exception types:

```python
from apa.common.exceptions import (
    ValidationError, ProcessingError, ModelError, 
    DataError, ConfigurationError, PipelineError
)

try:
    result = processor.process_data(data, config)
except ValidationError as e:
    print(f"Validation failed: {e}")
except ProcessingError as e:
    print(f"Processing failed: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
```

### Error Handling in Results

All operations return result objects with success flags:

```python
result = processor.process_data(data, config)

if result.success:
    # Process successful result
    processed_data = result.processed_data
else:
    # Handle error
    print(f"Error: {result.error_message}")
```

## Advanced Usage

### Custom Pipeline Creation

```python
# Create a custom pipeline
pipeline = ModularPipeline("my_custom_pipeline")

# Add stages with dependencies
pipeline.add_custom_stage('data_import', HyperspectralDataImporter())
pipeline.add_custom_stage('roi_processing', ROIProcessor(), dependencies=['data_import'])
pipeline.add_custom_stage('road_extraction', RoadExtractor(), dependencies=['roi_processing'])
pipeline.add_custom_stage('model_training', UNetModel(), dependencies=['road_extraction'])

# Run pipeline
result = pipeline.run_pipeline(data, config)
```

### Stage Management

```python
# Get pipeline information
info = pipeline.get_pipeline_info()
print(f"Stages: {info['stages']}")
print(f"Execution order: {info['execution_order']}")
print(f"Dependencies: {info['dependencies']}")

# Remove a stage
pipeline.remove_stage('model_training')

# Create stage from type
new_stage = pipeline.create_stage_from_type('roi_processing', 'new_roi', config)
```

### Data Validation

```python
from apa.common.validators import InputValidator

# Validate data container
InputValidator.validate_data_container(data)

# Validate numpy array
InputValidator.validate_numpy_array(array, expected_shape=(100, 100, 12))

# Validate hyperspectral data
InputValidator.validate_hyperspectral_data(hyperspectral_array)

# Validate ground truth data
InputValidator.validate_ground_truth_data(pci_array)
```

## Best Practices

### 1. Module Design

- Always inherit from appropriate base classes
- Implement all required interface methods
- Validate inputs and outputs
- Provide meaningful error messages
- Document configuration parameters

### 2. Pipeline Design

- Design stages to be independent
- Use clear dependency relationships
- Handle errors gracefully
- Provide progress feedback
- Save intermediate results

### 3. Configuration

- Use descriptive configuration keys
- Provide default values where possible
- Validate configuration parameters
- Document all configuration options
- Use consistent naming conventions

### 4. Error Handling

- Use specific exception types
- Provide detailed error messages
- Include context information
- Handle errors at appropriate levels
- Log errors for debugging

### 5. Testing

- Test each module independently
- Test pipeline compositions
- Test error conditions
- Validate all inputs and outputs
- Test configuration validation

## Examples

See the `examples/` directory for complete working examples:

- `common_api_usage.py`: Basic API usage examples
- `modular_pipeline_example.py`: Custom pipeline creation
- Additional examples for specific use cases

## Migration Guide

### From Legacy Code

To migrate existing APA code to use the common API:

1. **Identify module types**: Determine if modules are data importers, processors, models, or pipelines
2. **Inherit from base classes**: Change module classes to inherit from appropriate base classes
3. **Implement interfaces**: Ensure all required interface methods are implemented
4. **Update data handling**: Use `DataContainer` for all data
5. **Standardize results**: Return appropriate result objects
6. **Add validation**: Include input/output validation
7. **Update configuration**: Use standardized configuration format

### Example Migration

```python
# Before (legacy)
class MyProcessor:
    def process(self, data, config):
        # Processing logic
        return processed_data

# After (common API)
class MyProcessor(BaseDataProcessor):
    def __init__(self, config=None):
        super().__init__("my_processor", config)
        self.supported_data_types = ['hyperspectral']
        self.required_config_keys = ['param1']
    
    def load_data(self, source):
        # Implementation
        pass
    
    def validate_data(self, data):
        # Implementation
        return True
    
    def get_data_info(self, data):
        # Implementation
        return data.get_info()
    
    def _process_impl(self, data, config):
        # Processing logic
        return processed_data_container
```

## Conclusion

The APA Common API provides a robust, standardized foundation for building modular, reusable pipeline components. By following the patterns and practices outlined in this guide, you can create maintainable, testable, and composable modules that integrate seamlessly with the APA ecosystem.




