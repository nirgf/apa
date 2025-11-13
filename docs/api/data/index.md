# Data Management API

The `apa.data` module provides comprehensive data import, preprocessing, and validation capabilities for the APA pipeline, supporting various satellite imagery sources and ground truth data formats.

## 📦 Module Overview

```python
from apa.data import DataImporter, DataPreprocessor, DataValidator
```

The data module handles:
- Hyperspectral imagery import from multiple sources
- Ground truth PCI data loading and processing
- Road network data from OpenStreetMap
- Data validation and quality assessment
- Data preprocessing and normalization

## 🔧 Classes

### DataImporter

Handles importing data from various sources including satellite imagery, ground truth data, and configuration files.

#### Constructor

```python
DataImporter(config: Dict[str, Any])
```

**Parameters:**
- `config` (Dict[str, Any]): Configuration dictionary containing data source settings

**Example:**
```python
from apa.data.importers import DataImporter

# Initialize with configuration
importer = DataImporter(config)
```

#### Methods

##### `import_hyperspectral_data(data_dirname: str, data_filename: List[str], metadata_filename: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List]`

Import hyperspectral imagery data from various satellite sources.

**Parameters:**
- `data_dirname` (str): Directory containing the data files
- `data_filename` (List[str]): List of data filenames
- `metadata_filename` (str): Path to metadata file

**Returns:**
- `Tuple[np.ndarray, np.ndarray, np.ndarray, List]`: Tuple of (lon_mat, lat_mat, Msp_Image, rois)

**Raises:**
- `Exception`: If data import fails

**Example:**
```python
# Import hyperspectral data
lon_mat, lat_mat, msp_image, rois = importer.import_hyperspectral_data(
    'data/satellite_images/',
    ['image_1.tif', 'image_2.tif'],
    'data/metadata.json'
)

print(f"Image shape: {msp_image.shape}")
print(f"ROIs: {rois}")
```

##### `import_ground_truth_data(excel_path: str) -> Dict[str, Any]`

Import ground truth PCI data from Excel/CSV files.

**Parameters:**
- `excel_path` (str): Path to the Excel/CSV file containing PCI data

**Returns:**
- `Dict[str, Any]`: Dictionary containing ground truth data

**Raises:**
- `Exception`: If ground truth import fails

**Example:**
```python
# Import ground truth data
ground_truth = importer.import_ground_truth_data('data/Detroit/Pavement_Condition.csv')

print(f"Ground truth file: {ground_truth['file_path']}")
print(f"Has data: {ground_truth['has_data']}")
```

##### `import_road_network_data(config: Dict[str, Any]) -> Dict[str, Any]`

Import road network data from OpenStreetMap.

**Parameters:**
- `config` (Dict[str, Any]): Configuration dictionary containing road network settings

**Returns:**
- `Dict[str, Any]`: Dictionary containing road network data

**Raises:**
- `Exception`: If road network import fails

**Example:**
```python
# Import road network data
road_network = importer.import_road_network_data(config)

print(f"Road network source: {road_network['metadata']['source']}")
```

##### `validate_imported_data(data: Dict[str, Any]) -> bool`

Validate imported data for consistency and quality.

**Parameters:**
- `data` (Dict[str, Any]): Dictionary containing imported data

**Returns:**
- `bool`: True if data is valid, False otherwise

**Example:**
```python
# Validate imported data
data_dict = {
    'hyperspectral_data': {
        'lon_mat': lon_mat,
        'lat_mat': lat_mat,
        'Msp_Image': msp_image,
        'rois': rois
    },
    'ground_truth_data': ground_truth
}

is_valid = importer.validate_imported_data(data_dict)
print(f"Data validation: {'✓ Valid' if is_valid else '✗ Invalid'}")
```

##### `get_data_summary(data: Dict[str, Any]) -> Dict[str, Any]`

Get a summary of imported data.

**Parameters:**
- `data` (Dict[str, Any]): Dictionary containing imported data

**Returns:**
- `Dict[str, Any]`: Dictionary containing data summary

**Example:**
```python
# Get data summary
summary = importer.get_data_summary(data_dict)

print(f"Hyperspectral image shape: {summary['hyperspectral']['image_shape']}")
print(f"Number of ROIs: {summary['hyperspectral']['num_rois']}")
```

### DataPreprocessor

Handles data preprocessing operations including normalization, filtering, and augmentation.

#### Constructor

```python
DataPreprocessor(config: Dict[str, Any])
```

**Parameters:**
- `config` (Dict[str, Any]): Configuration dictionary containing preprocessing settings

**Example:**
```python
from apa.data.preprocessors import DataPreprocessor

# Initialize preprocessor
preprocessor = DataPreprocessor(config)
```

#### Methods

##### `normalize_data(data: np.ndarray, method: str = 'min-max') -> np.ndarray`

Normalize data using specified method.

**Parameters:**
- `data` (np.ndarray): Input data array
- `method` (str): Normalization method ('min-max', 'z-score', 'robust')

**Returns:**
- `np.ndarray`: Normalized data array

**Example:**
```python
# Normalize hyperspectral data
normalized_data = preprocessor.normalize_data(msp_image, method='min-max')
print(f"Normalized data range: {normalized_data.min():.3f} to {normalized_data.max():.3f}")
```

##### `apply_spectral_smoothing(data: np.ndarray, window_size: int = 3) -> np.ndarray`

Apply spectral smoothing to reduce noise.

**Parameters:**
- `data` (np.ndarray): Input data array
- `window_size` (int): Size of smoothing window

**Returns:**
- `np.ndarray`: Smoothed data array

**Example:**
```python
# Apply spectral smoothing
smoothed_data = preprocessor.apply_spectral_smoothing(msp_image, window_size=5)
```

##### `crop_to_roi(data: np.ndarray, roi: List[float]) -> np.ndarray`

Crop data to specified region of interest.

**Parameters:**
- `data` (np.ndarray): Input data array
- `roi` (List[float]): Region of interest [xmin, xmax, ymin, ymax]

**Returns:**
- `np.ndarray`: Cropped data array

**Example:**
```python
# Crop to ROI
roi = [-83.14294, -83.00007, 42.34429, 42.39170]
cropped_data = preprocessor.crop_to_roi(msp_image, roi)
print(f"Cropped data shape: {cropped_data.shape}")
```

### DataValidator

Validates data quality and consistency.

#### Constructor

```python
DataValidator()
```

**Example:**
```python
from apa.data.validators import DataValidator

# Initialize validator
validator = DataValidator()
```

#### Methods

##### `validate_data_shapes(data: Dict[str, Any]) -> bool`

Validate that data arrays have consistent shapes.

**Parameters:**
- `data` (Dict[str, Any]): Dictionary containing data arrays

**Returns:**
- `bool`: True if shapes are consistent, False otherwise

**Example:**
```python
# Validate data shapes
data_dict = {
    'lon_mat': lon_mat,
    'lat_mat': lat_mat,
    'msp_image': msp_image
}

shapes_valid = validator.validate_data_shapes(data_dict)
print(f"Shape validation: {'✓ Valid' if shapes_valid else '✗ Invalid'}")
```

##### `validate_data_ranges(data: np.ndarray, expected_range: Tuple[float, float]) -> bool`

Validate that data values are within expected range.

**Parameters:**
- `data` (np.ndarray): Input data array
- `expected_range` (Tuple[float, float]): Expected (min, max) range

**Returns:**
- `bool`: True if data is within range, False otherwise

**Example:**
```python
# Validate data ranges
range_valid = validator.validate_data_ranges(msp_image, (0.0, 1.0))
print(f"Range validation: {'✓ Valid' if range_valid else '✗ Invalid'}")
```

## 📊 Supported Data Sources

### Satellite Imagery Sources

#### VENUS (Israel - Kiryat Ata)
- **Format**: GeoTIFF files
- **Bands**: 12 spectral bands
- **Resolution**: 5.3m spatial resolution
- **Coverage**: Kiryat Ata region, Israel

#### VENUS (Detroit)
- **Format**: GeoTIFF files
- **Bands**: 12 spectral bands
- **Resolution**: 5.3m spatial resolution
- **Coverage**: Detroit metropolitan area

#### Airbus (Detroit)
- **Format**: GeoTIFF files
- **Bands**: Multispectral (4 bands) or Panchromatic (1 band)
- **Resolution**: Variable (1-5m)
- **Coverage**: Detroit metropolitan area

### Ground Truth Data

#### Detroit Pavement Condition
- **Format**: CSV files
- **Fields**: Road segments with PCI scores
- **Coverage**: Detroit road network
- **Source**: Municipal data

#### Custom PCI Data
- **Format**: Excel/CSV files
- **Fields**: Configurable field mapping
- **Coverage**: Any geographic region
- **Source**: User-provided data

## 🎯 Usage Examples

### Complete Data Import Workflow

```python
from apa.data.importers import DataImporter
from apa.data.preprocessors import DataPreprocessor
from apa.data.validators import DataValidator

def import_and_process_data(config):
    """Complete data import and processing workflow."""
    
    # Initialize components
    importer = DataImporter(config)
    preprocessor = DataPreprocessor(config)
    validator = DataValidator()
    
    # Import hyperspectral data
    print("Importing hyperspectral data...")
    lon_mat, lat_mat, msp_image, rois = importer.import_hyperspectral_data(
        config['data']['input_path'],
        [config['data']['filename_NED'], config['data']['filename_RGB']],
        'data/dummy_metadata.json'
    )
    
    # Import ground truth data
    print("Importing ground truth data...")
    ground_truth = importer.import_ground_truth_data(
        'data/Detroit/Pavement_Condition.csv'
    )
    
    # Import road network data
    print("Importing road network data...")
    road_network = importer.import_road_network_data(config)
    
    # Prepare data dictionary
    data_dict = {
        'hyperspectral_data': {
            'lon_mat': lon_mat,
            'lat_mat': lat_mat,
            'Msp_Image': msp_image,
            'rois': rois
        },
        'ground_truth_data': ground_truth,
        'road_network_data': road_network
    }
    
    # Validate imported data
    print("Validating data...")
    if not importer.validate_imported_data(data_dict):
        raise ValueError("Data validation failed")
    
    # Preprocess data
    print("Preprocessing data...")
    if config['preprocessing']['normalization']:
        msp_image = preprocessor.normalize_data(
            msp_image, 
            config['preprocessing']['normalization_type']
        )
    
    if config['preprocessing']['spectral_smoothing']:
        msp_image = preprocessor.apply_spectral_smoothing(msp_image)
    
    # Get data summary
    summary = importer.get_data_summary(data_dict)
    print(f"Data summary: {summary}")
    
    return data_dict, summary

# Usage
config = load_config('configs/detroit.yaml')
data_dict, summary = import_and_process_data(config['config'])
```

### ROI-Based Data Processing

```python
def process_roi_data(data_dict, roi_index=0):
    """Process data for a specific ROI."""
    
    preprocessor = DataPreprocessor(config)
    
    # Get ROI data
    rois = data_dict['hyperspectral_data']['rois']
    msp_image = data_dict['hyperspectral_data']['Msp_Image']
    
    if roi_index >= len(rois):
        raise ValueError(f"ROI index {roi_index} out of range")
    
    roi = rois[roi_index]
    print(f"Processing ROI {roi_index}: {roi}")
    
    # Crop to ROI
    cropped_image = preprocessor.crop_to_roi(msp_image, roi)
    
    # Apply preprocessing
    if config['preprocessing']['normalization']:
        cropped_image = preprocessor.normalize_data(cropped_image)
    
    return cropped_image, roi

# Usage
cropped_image, roi = process_roi_data(data_dict, roi_index=0)
print(f"Cropped image shape: {cropped_image.shape}")
```

### Data Quality Assessment

```python
def assess_data_quality(data_dict):
    """Assess the quality of imported data."""
    
    validator = DataValidator()
    
    # Validate shapes
    shapes_valid = validator.validate_data_shapes(data_dict['hyperspectral_data'])
    print(f"Shape validation: {'✓' if shapes_valid else '✗'}")
    
    # Validate ranges
    msp_image = data_dict['hyperspectral_data']['Msp_Image']
    range_valid = validator.validate_data_ranges(msp_image, (0.0, 1.0))
    print(f"Range validation: {'✓' if range_valid else '✗'}")
    
    # Check for missing values
    has_nans = np.isnan(msp_image).any()
    print(f"Missing values: {'✗ Found' if has_nans else '✓ None'}")
    
    # Check for infinite values
    has_infs = np.isinf(msp_image).any()
    print(f"Infinite values: {'✗ Found' if has_infs else '✓ None'}")
    
    return {
        'shapes_valid': shapes_valid,
        'range_valid': range_valid,
        'has_nans': has_nans,
        'has_infs': has_infs
    }

# Usage
quality_report = assess_data_quality(data_dict)
```

## 🔧 Advanced Usage

### Custom Data Source Integration

```python
class CustomDataImporter(DataImporter):
    """Custom data importer for new data sources."""
    
    def import_custom_data(self, data_path: str) -> Dict[str, Any]:
        """Import data from custom source."""
        # Implement custom import logic
        pass
    
    def validate_custom_data(self, data: Dict[str, Any]) -> bool:
        """Validate custom data format."""
        # Implement custom validation logic
        pass

# Usage
custom_importer = CustomDataImporter(config)
custom_data = custom_importer.import_custom_data('path/to/custom/data')
```

### Batch Data Processing

```python
def process_multiple_datasets(configs):
    """Process multiple datasets in batch."""
    
    results = []
    
    for config in configs:
        try:
            importer = DataImporter(config)
            data_dict = importer.import_hyperspectral_data(
                config['data']['input_path'],
                config['data']['filename'],
                config['data']['metadata_path']
            )
            
            # Validate data
            if importer.validate_imported_data(data_dict):
                results.append({
                    'config': config,
                    'data': data_dict,
                    'status': 'success'
                })
            else:
                results.append({
                    'config': config,
                    'data': None,
                    'status': 'validation_failed'
                })
                
        except Exception as e:
            results.append({
                'config': config,
                'data': None,
                'status': f'error: {str(e)}'
            })
    
    return results

# Usage
configs = [load_config(f'configs/{name}.yaml') for name in ['detroit', 'kiryat_ata']]
results = process_multiple_datasets(configs)
```

## 🚨 Error Handling

### Common Data Import Errors

1. **File Not Found**
   ```python
   # Error: Data file not found
   FileNotFoundError: Data file not found: data/satellite_images/image.tif
   ```

2. **Invalid Data Format**
   ```python
   # Error: Invalid data format
   ValueError: Unsupported data format. Expected GeoTIFF, got JPEG
   ```

3. **Data Validation Failure**
   ```python
   # Error: Data validation failed
   ValueError: Hyperspectral image spatial dimensions must match coordinate matrices
   ```

### Error Handling Best Practices

```python
def safe_data_import(config):
    """Safely import data with comprehensive error handling."""
    try:
        importer = DataImporter(config)
        
        # Import data
        data = importer.import_hyperspectral_data(
            config['data']['input_path'],
            config['data']['filename'],
            config['data']['metadata_path']
        )
        
        # Validate data
        if not importer.validate_imported_data(data):
            raise ValueError("Data validation failed")
        
        return data
        
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Data file not found: {e}")
    except ValueError as e:
        raise ValueError(f"Data validation error: {e}")
    except Exception as e:
        raise RuntimeError(f"Unexpected data import error: {e}")
```

## 🔗 Related Documentation

- [Main API Documentation](../index.md)
- [Configuration API](../config/)
- [Pipeline API](../pipeline/)
- [Processing API](../processing/)
