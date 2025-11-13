# APA Examples

This directory contains example scripts demonstrating how to use the APA API.

## Basic Usage

The `basic_usage.py` script demonstrates:

1. **Data Import**: Loading hyperspectral imagery data
2. **Data Processing**: Processing data through multiple stages (ROI, road extraction, preprocessing)
3. **Model Training**: Training a U-Net model and making predictions
4. **Custom Pipeline**: Creating a custom modular pipeline

### Running the Example

```bash
# From the project root
python examples/basic_usage.py
```

## Plug-and-Play Modules

All APA modules follow a consistent API, making them easy to use:

```python
from apa.modules import HyperspectralDataImporter, ROIProcessor

# Create a module with configuration
importer = HyperspectralDataImporter({
    'input_path': 'data/',
    'dataset': 1,
})

# Use the module
data = importer.load_data()

# Process with another module
processor = ROIProcessor({'roi_bounds': [42.3, 42.4, -83.0, -82.9]})
result = processor.process_data(data)
```

## Adding Your Own Logic

To add your own logic to a module:

1. Find the module in `src/apa/modules/`
2. Look for `# TODO:` comments
3. Replace the placeholder code with your implementation
4. The API handles all input/output parsing automatically

Example:

```python
# In src/apa/modules/data_importers.py
def _load_venus_detroit(self, input_path: str, config: Dict[str, Any]) -> np.ndarray:
    """Load VENUS Detroit data."""
    # TODO: Replace this with your actual loading logic
    filename = config.get('filename_NED', 'NED.h5')
    filepath = os.path.join(input_path, filename)
    
    with h5py.File(filepath, 'r') as f:
        data = f['your_data_key'][:]  # Replace with actual key
    
    return data
```

The module will automatically:
- Validate inputs
- Parse outputs into DataContainer format
- Handle errors
- Log operations

