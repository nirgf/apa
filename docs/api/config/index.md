# Configuration Management API

The `apa.config` module provides comprehensive configuration management for the APA pipeline, including loading, validation, and template-based configuration creation.

## 📦 Module Overview

```python
from apa.config import ConfigManager, ConfigSchema
```

The configuration module handles:
- YAML configuration file loading and validation
- Template-based configuration creation
- Default value merging and validation
- Schema-based configuration validation

## 🔧 Classes

### ConfigManager

The main configuration manager class that handles loading, saving, and managing configuration files.

#### Constructor

```python
ConfigManager(config_dir: Optional[str] = None)
```

**Parameters:**
- `config_dir` (Optional[str]): Directory containing configuration files. If None, uses default configs directory.

**Example:**
```python
from apa.config.manager import ConfigManager

# Initialize with default config directory
manager = ConfigManager()

# Initialize with custom config directory
manager = ConfigManager('/path/to/configs')
```

#### Methods

##### `load_config(config_path: str) -> Dict[str, Any]`

Load and validate a configuration file.

**Parameters:**
- `config_path` (str): Path to the configuration file

**Returns:**
- `Dict[str, Any]`: Validated configuration dictionary

**Raises:**
- `FileNotFoundError`: If config file doesn't exist
- `yaml.YAMLError`: If config file is invalid YAML
- `ValueError`: If config doesn't match schema

**Example:**
```python
# Load configuration
config = manager.load_config('configs/detroit.yaml')
print(f"Data source: {config['config']['data']['enum_data_source']}")
```

##### `save_config(config: Dict[str, Any], output_path: str) -> None`

Save a configuration dictionary to a file.

**Parameters:**
- `config` (Dict[str, Any]): Configuration dictionary
- `output_path` (str): Path where to save the configuration

**Example:**
```python
# Save configuration
manager.save_config(config, 'output/config.yaml')
```

##### `create_config_from_template(template_name: str, output_path: str) -> None`

Create a new configuration file from a template.

**Parameters:**
- `template_name` (str): Name of the template to use
- `output_path` (str): Path where to save the new configuration

**Raises:**
- `ValueError`: If template doesn't exist

**Example:**
```python
# Create configuration from template
manager.create_config_from_template('detroit', 'my_config.yaml')
```

##### `list_templates() -> List[str]`

List available configuration templates.

**Returns:**
- `List[str]`: List of template names

**Example:**
```python
# List available templates
templates = manager.list_templates()
print(f"Available templates: {templates}")
```

##### `get_default_config() -> Dict[str, Any]`

Get the default configuration.

**Returns:**
- `Dict[str, Any]`: Default configuration dictionary

**Example:**
```python
# Get default configuration
default_config = manager.get_default_config()
```

##### `merge_with_defaults(config: Dict[str, Any]) -> Dict[str, Any]`

Merge a configuration with default values.

**Parameters:**
- `config` (Dict[str, Any]): Configuration to merge

**Returns:**
- `Dict[str, Any]`: Merged configuration

**Example:**
```python
# Merge with defaults
custom_config = {'config': {'data': {'input_path': 'custom/path'}}}
merged_config = manager.merge_with_defaults(custom_config)
```

### ConfigSchema

Configuration schema validator that ensures configuration files match the expected structure.

#### Constructor

```python
ConfigSchema()
```

**Example:**
```python
from apa.config.schemas import ConfigSchema

schema = ConfigSchema()
```

#### Methods

##### `validate(config: Dict[str, Any]) -> None`

Validate a configuration dictionary.

**Parameters:**
- `config` (Dict[str, Any]): Configuration dictionary to validate

**Raises:**
- `ValueError`: If configuration is invalid

**Example:**
```python
# Validate configuration
try:
    schema.validate(config)
    print("Configuration is valid")
except ValueError as e:
    print(f"Configuration error: {e}")
```

## 📋 Configuration Structure

### Main Configuration Sections

The APA configuration follows this structure:

```yaml
config:
  data:                    # Data source and paths
    input_path: "data/hyperspectral_images/"
    output_path: "results/"
    enum_data_source: "VENUS_DETROIT"
    rois: [[-83.14294, -83.00007, 42.34429, 42.39170]]
    
  preprocessing:           # Data preprocessing settings
    normalization: true
    normalization_type: "min-max"
    georeferencing:
      merge_threshold: [0.05]
      merge_method: "mean_min"
      
  cnn_model:              # Model architecture settings
    architecture: "unet_categorical"
    input_shape: "(64, 64, 12)"
    num_classes: 4
    
  training:               # Training parameters
    batch_size: 16
    epochs: 100
    learning_rate_ini: 0.001
    
  output:                 # Output settings
    save_model: true
    model_save_path: "models/trained_model.h5"
```

### Required Fields

#### Data Section
- `input_path`: Path to input data directory
- `output_path`: Path for output results
- `rois`: List of regions of interest (each ROI is [xmin, xmax, ymin, ymax])

#### Preprocessing Section
- `normalization`: Whether to normalize pixel values (boolean)
- `normalization_type`: Type of normalization ("min-max", "z-score", "robust")

#### CNN Model Section
- `architecture`: Model architecture ("unet_categorical", "3d_cnn", "2d_cnn")
- `input_shape`: Input tensor shape (string format like "(64, 64, 12)")
- `num_classes`: Number of target classes (integer)

#### Training Section
- `batch_size`: Training batch size (positive integer)
- `epochs`: Number of training epochs (positive integer)

## 🎯 Usage Examples

### Basic Configuration Loading

```python
from apa.config.manager import ConfigManager

# Initialize manager
config_manager = ConfigManager()

# Load configuration
config = config_manager.load_config('configs/detroit.yaml')

# Access configuration values
data_config = config['config']['data']
print(f"Input path: {data_config['input_path']}")
print(f"ROIs: {data_config['rois']}")
```

### Configuration Validation

```python
from apa.config.schemas import ConfigSchema

# Initialize schema
schema = ConfigSchema()

# Validate configuration
try:
    schema.validate(config)
    print("✓ Configuration is valid")
except ValueError as e:
    print(f"✗ Configuration error: {e}")
```

### Template-Based Configuration Creation

```python
# List available templates
templates = config_manager.list_templates()
print(f"Available templates: {templates}")

# Create new configuration from template
config_manager.create_config_from_template('detroit', 'my_detroit_config.yaml')
```

### Custom Configuration with Defaults

```python
# Create custom configuration
custom_config = {
    'config': {
        'data': {
            'input_path': '/custom/data/path',
            'rois': [[-83.0, -83.1, 42.3, 42.4]]
        }
    }
}

# Merge with defaults
merged_config = config_manager.merge_with_defaults(custom_config)

# Save merged configuration
config_manager.save_config(merged_config, 'custom_config.yaml')
```

### Configuration Validation with Error Handling

```python
def validate_config_file(config_path):
    """Validate a configuration file with detailed error reporting."""
    try:
        config_manager = ConfigManager()
        config = config_manager.load_config(config_path)
        print(f"✓ Configuration '{config_path}' is valid")
        return config
    except FileNotFoundError:
        print(f"✗ Configuration file not found: {config_path}")
    except yaml.YAMLError as e:
        print(f"✗ Invalid YAML in configuration: {e}")
    except ValueError as e:
        print(f"✗ Configuration validation error: {e}")
    return None

# Usage
config = validate_config_file('configs/detroit.yaml')
```

## 🔧 Advanced Usage

### Custom Configuration Directory

```python
# Use custom configuration directory
custom_manager = ConfigManager('/path/to/custom/configs')
config = custom_manager.load_config('my_config.yaml')
```

### Batch Configuration Processing

```python
import glob
from pathlib import Path

def process_all_configs(config_dir):
    """Process all YAML files in a directory."""
    config_manager = ConfigManager()
    schema = ConfigSchema()
    
    config_files = glob.glob(f"{config_dir}/*.yaml")
    
    for config_file in config_files:
        try:
            config = config_manager.load_config(config_file)
            print(f"✓ {config_file}: Valid")
        except Exception as e:
            print(f"✗ {config_file}: {e}")

# Usage
process_all_configs('configs/')
```

### Configuration Comparison

```python
def compare_configs(config1_path, config2_path):
    """Compare two configuration files."""
    config_manager = ConfigManager()
    
    config1 = config_manager.load_config(config1_path)
    config2 = config_manager.load_config(config2_path)
    
    # Compare specific sections
    data1 = config1['config']['data']
    data2 = config2['config']['data']
    
    differences = []
    for key in set(data1.keys()) | set(data2.keys()):
        if data1.get(key) != data2.get(key):
            differences.append(f"{key}: {data1.get(key)} vs {data2.get(key)}")
    
    return differences

# Usage
diffs = compare_configs('configs/detroit.yaml', 'configs/kiryat_ata.yaml')
for diff in diffs:
    print(diff)
```

## 🚨 Error Handling

### Common Configuration Errors

1. **Missing Required Fields**
   ```python
   # Error: Missing required field
   ValueError: Missing required data field: rois
   ```

2. **Invalid ROI Format**
   ```python
   # Error: Invalid ROI format
   ValueError: Each ROI must be a list of 4 coordinates [xmin, xmax, ymin, ymax]
   ```

3. **Invalid Data Types**
   ```python
   # Error: Invalid data type
   ValueError: batch_size must be a positive integer
   ```

### Error Handling Best Practices

```python
def safe_config_load(config_path):
    """Safely load configuration with comprehensive error handling."""
    try:
        config_manager = ConfigManager()
        config = config_manager.load_config(config_path)
        return config
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML syntax in {config_path}: {e}")
    except ValueError as e:
        raise ValueError(f"Configuration validation failed for {config_path}: {e}")
    except Exception as e:
        raise RuntimeError(f"Unexpected error loading {config_path}: {e}")
```

## 🔗 Related Documentation

- [Main API Documentation](../index.md)
- [Data Management API](../data/)
- [Pipeline API](../pipeline/)
- [Configuration Reference](../../configuration.md)
