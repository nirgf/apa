# Processing API

The `apa.processing` module provides image processing, filtering, and geospatial operations for the APA pipeline, handling hyperspectral image processing, georeferencing, and road extraction.

## 📦 Module Overview

```python
from apa.processing import ImageProcessor, Georeferencer, RoadExtractor
```

The processing module handles:
- Hyperspectral image processing and filtering
- Georeferencing and coordinate transformations
- Road extraction from satellite imagery
- Image enhancement and preprocessing

## 🔧 Classes

### ImageProcessor

Handles image processing operations for hyperspectral data.

#### Constructor

```python
ImageProcessor(config: Dict[str, Any])
```

**Parameters:**
- `config` (Dict[str, Any]): Configuration dictionary containing processing settings

**Example:**
```python
from apa.processing.filters import ImageProcessor

# Initialize image processor
processor = ImageProcessor(config)
```

#### Methods

##### `apply_filters(image: np.ndarray, filter_type: str) -> np.ndarray`

Apply specified filters to the image.

**Parameters:**
- `image` (np.ndarray): Input image array
- `filter_type` (str): Type of filter to apply

**Returns:**
- `np.ndarray`: Filtered image array

**Example:**
```python
# Apply filters
filtered_image = processor.apply_filters(hyperspectral_image, 'gaussian')
```

##### `enhance_image(image: np.ndarray) -> np.ndarray`

Enhance image quality and contrast.

**Parameters:**
- `image` (np.ndarray): Input image array

**Returns:**
- `np.ndarray`: Enhanced image array

**Example:**
```python
# Enhance image
enhanced_image = processor.enhance_image(hyperspectral_image)
```

### Georeferencer

Handles georeferencing and coordinate transformations.

#### Constructor

```python
Georeferencer(config: Dict[str, Any])
```

**Parameters:**
- `config` (Dict[str, Any]): Configuration dictionary containing georeferencing settings

**Example:**
```python
from apa.processing.georeferencing import Georeferencer

# Initialize georeferencer
georeferencer = Georeferencer(config)
```

#### Methods

##### `transform_coordinates(coords: np.ndarray, from_crs: str, to_crs: str) -> np.ndarray`

Transform coordinates between different coordinate reference systems.

**Parameters:**
- `coords` (np.ndarray): Input coordinates
- `from_crs` (str): Source coordinate reference system
- `to_crs` (str): Target coordinate reference system

**Returns:**
- `np.ndarray`: Transformed coordinates

**Example:**
```python
# Transform coordinates
transformed_coords = georeferencer.transform_coordinates(
    coordinates, 'EPSG:4326', 'EPSG:3857'
)
```

##### `georeference_image(image: np.ndarray, bounds: Tuple[float, float, float, float]) -> np.ndarray`

Georeference an image with specified bounds.

**Parameters:**
- `image` (np.ndarray): Input image array
- `bounds` (Tuple[float, float, float, float]): Geographic bounds (xmin, ymin, xmax, ymax)

**Returns:**
- `np.ndarray`: Georeferenced image

**Example:**
```python
# Georeference image
bounds = (-83.14294, 42.34429, -83.00007, 42.39170)
georeferenced_image = georeferencer.georeference_image(image, bounds)
```

### RoadExtractor

Extracts road networks from satellite imagery.

#### Constructor

```python
RoadExtractor(config: Dict[str, Any])
```

**Parameters:**
- `config` (Dict[str, Any]): Configuration dictionary containing extraction settings

**Example:**
```python
from apa.processing.segmentation import RoadExtractor

# Initialize road extractor
extractor = RoadExtractor(config)
```

#### Methods

##### `extract_roads(image: np.ndarray) -> np.ndarray`

Extract road network from satellite image.

**Parameters:**
- `image` (np.ndarray): Input satellite image

**Returns:**
- `np.ndarray`: Binary road mask

**Example:**
```python
# Extract roads
road_mask = extractor.extract_roads(satellite_image)
print(f"Road mask shape: {road_mask.shape}")
```

##### `segment_roads(road_mask: np.ndarray) -> np.ndarray`

Segment roads into individual road segments.

**Parameters:**
- `road_mask` (np.ndarray): Binary road mask

**Returns:**
- `np.ndarray`: Segmented road network

**Example:**
```python
# Segment roads
segmented_roads = extractor.segment_roads(road_mask)
print(f"Number of road segments: {len(np.unique(segmented_roads))}")
```

## 🎯 Usage Examples

### Complete Image Processing Pipeline

```python
from apa.processing.filters import ImageProcessor
from apa.processing.georeferencing import Georeferencer
from apa.processing.segmentation import RoadExtractor

def process_satellite_image():
    """Complete satellite image processing pipeline."""
    
    # Initialize processors
    image_processor = ImageProcessor(config)
    georeferencer = Georeferencer(config)
    road_extractor = RoadExtractor(config)
    
    # Load satellite image
    satellite_image = load_satellite_image()
    
    # Apply image processing
    filtered_image = image_processor.apply_filters(satellite_image, 'gaussian')
    enhanced_image = image_processor.enhance_image(filtered_image)
    
    # Georeference image
    bounds = (-83.14294, 42.34429, -83.00007, 42.39170)
    georeferenced_image = georeferencer.georeference_image(enhanced_image, bounds)
    
    # Extract roads
    road_mask = road_extractor.extract_roads(georeferenced_image)
    segmented_roads = road_extractor.segment_roads(road_mask)
    
    return {
        'original_image': satellite_image,
        'processed_image': georeferenced_image,
        'road_mask': road_mask,
        'segmented_roads': segmented_roads
    }

# Usage
results = process_satellite_image()
```

### Batch Processing

```python
def batch_process_images(image_paths: List[str]):
    """Process multiple satellite images in batch."""
    
    processor = ImageProcessor(config)
    extractor = RoadExtractor(config)
    
    results = []
    
    for image_path in image_paths:
        try:
            # Load image
            image = load_image(image_path)
            
            # Process image
            processed_image = processor.apply_filters(image, 'gaussian')
            
            # Extract roads
            road_mask = extractor.extract_roads(processed_image)
            
            results.append({
                'image_path': image_path,
                'road_mask': road_mask,
                'status': 'success'
            })
            
        except Exception as e:
            results.append({
                'image_path': image_path,
                'road_mask': None,
                'status': f'error: {str(e)}'
            })
    
    return results

# Usage
image_paths = ['image1.tif', 'image2.tif', 'image3.tif']
results = batch_process_images(image_paths)
```

## 🔗 Related Documentation

- [Main API Documentation](../index.md)
- [Data Management API](../data/)
- [Models API](../models/)
- [Utilities API](../utils/)
