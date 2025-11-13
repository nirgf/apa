# APA API Reference

This directory contains comprehensive API documentation for all modules in the APA (Advanced Pavement Analytics) package. The API is organized into logical modules, each handling specific aspects of the pavement analytics pipeline.

## 📚 API Structure

```
docs/api/
├── index.md              # Main API overview and quick start
├── config/               # Configuration management
│   └── index.md         # ConfigManager, ConfigSchema
├── data/                 # Data import and processing
│   └── index.md         # DataImporter, DataPreprocessor, DataValidator
├── pipeline/             # Pipeline orchestration
│   └── index.md         # APAPipeline, PipelineStage
├── utils/                # Utilities and helpers
│   └── index.md         # IOUtils, VisualizationUtils, MetricsCalculator
├── models/               # Machine learning models
│   └── index.md         # CNNModule, UNetModule, ModelManager
└── processing/           # Image processing
    └── index.md         # ImageProcessor, Georeferencer, RoadExtractor
```

## 🚀 Quick Start

### Basic Usage

```python
import apa
from apa.config.manager import ConfigManager
from apa.pipeline.runner import APAPipeline

# Load configuration
config_manager = ConfigManager()
config = config_manager.load_config('configs/detroit.yaml')

# Run pipeline
pipeline = APAPipeline(config['config'])
pipeline.run()

# Get results
results = pipeline.get_results()
```

### CLI Usage

```bash
# Run the complete pipeline
apa run --config configs/detroit.yaml

# Validate configuration
apa validate --config configs/detroit.yaml

# Create new configuration
apa create-config --template detroit --output my_config.yaml
```

## 📖 Module Documentation

### [Configuration Management](config/)
- **ConfigManager**: Load, validate, and manage configuration files
- **ConfigSchema**: Schema validation for configuration files
- **Features**: YAML loading, template system, default merging

### [Data Management](data/)
- **DataImporter**: Import data from various sources
- **DataPreprocessor**: Data preprocessing operations
- **DataValidator**: Data validation and quality checks
- **Features**: Multi-source support, validation, preprocessing

### [Pipeline Orchestration](pipeline/)
- **APAPipeline**: Main pipeline orchestrator
- **PipelineStage**: Individual pipeline stages
- **Features**: Stage-based execution, error handling, progress tracking

### [Utilities](utils/)
- **IOUtils**: File I/O operations
- **VisualizationUtils**: Plotting and visualization
- **MetricsCalculator**: Performance metrics
- **Features**: Multi-format I/O, comprehensive plotting, PCI metrics

### [Models](models/)
- **CNNModule**: CNN model implementations
- **UNetModule**: U-Net model implementations
- **ModelManager**: Model training and management
- **Features**: Deep learning models, training, evaluation

### [Processing](processing/)
- **ImageProcessor**: Image processing operations
- **Georeferencer**: Geospatial data processing
- **RoadExtractor**: Road network extraction
- **Features**: Image processing, georeferencing, road extraction

## 🎯 Common Patterns

### Configuration Loading
```python
from apa.config.manager import ConfigManager

config_manager = ConfigManager()
config = config_manager.load_config('path/to/config.yaml')
```

### Data Import
```python
from apa.data.importers import DataImporter

importer = DataImporter(config)
data = importer.import_hyperspectral_data(data_dir, filenames, metadata)
```

### Pipeline Execution
```python
from apa.pipeline.runner import APAPipeline

pipeline = APAPipeline(config)
pipeline.run()
results = pipeline.get_results()
```

### Visualization
```python
from apa.utils.visualization import VisualizationUtils

viz_utils = VisualizationUtils()
viz_utils.plot_roi_overview(roi_data, save_path='overview.png')
```

### Metrics Calculation
```python
from apa.utils.metrics import MetricsCalculator

metrics_calc = MetricsCalculator()
metrics = metrics_calc.calculate_pci_metrics(true_pci, pred_pci)
```

## 🔧 Development Guidelines

### Code Style
- Follow PEP 8 guidelines
- Use type hints for all functions
- Include comprehensive docstrings
- Write unit tests for all functions

### Error Handling
- Use specific exception types
- Provide meaningful error messages
- Include context in error information
- Log errors appropriately

### Documentation
- Document all public APIs
- Include usage examples
- Provide parameter descriptions
- Explain return values

## 📞 Support

For API-related questions:
- Check the specific module documentation
- Review the example notebooks
- Open an issue on GitHub
- Contact the development team

## 🔗 Related Documentation

- [Installation Guide](../installation.md)
- [Quick Start Guide](../quickstart.md)
- [Configuration Reference](../configuration.md)
- [Tutorials](../tutorials/)
- [Examples](../../examples/)
