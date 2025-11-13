# Models API

The `apa.models` module provides machine learning models and architectures for pavement condition analysis, including CNN and U-Net implementations for road segmentation and PCI prediction.

## 📦 Module Overview

```python
from apa.models import CNNModule, UNetModule, ModelManager
```

The models module handles:
- CNN architectures for pavement classification
- U-Net models for road segmentation
- Model training and evaluation
- Model management and persistence

## 🔧 Classes

### CNNModule

Convolutional Neural Network implementations for pavement condition analysis.

#### Constructor

```python
CNNModule(config: Dict[str, Any])
```

**Parameters:**
- `config` (Dict[str, Any]): Configuration dictionary containing model settings

**Example:**
```python
from apa.models.cnn import CNNModule

# Initialize CNN module
cnn_module = CNNModule(config)
```

#### Methods

##### `create_model() -> tf.keras.Model`

Create a CNN model based on configuration.

**Returns:**
- `tf.keras.Model`: Compiled CNN model

**Example:**
```python
# Create CNN model
model = cnn_module.create_model()
print(f"Model created with {model.count_params()} parameters")
```

##### `train_model(X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray) -> tf.keras.callbacks.History`

Train the CNN model.

**Parameters:**
- `X_train` (np.ndarray): Training features
- `y_train` (np.ndarray): Training labels
- `X_val` (np.ndarray): Validation features
- `y_val` (np.ndarray): Validation labels

**Returns:**
- `tf.keras.callbacks.History`: Training history

**Example:**
```python
# Train model
history = cnn_module.train_model(X_train, y_train, X_val, y_val)
print("Training completed")
```

### UNetModule

U-Net implementation for road segmentation from satellite imagery.

#### Constructor

```python
UNetModule(config: Dict[str, Any])
```

**Parameters:**
- `config` (Dict[str, Any]): Configuration dictionary containing model settings

**Example:**
```python
from apa.models.unet import UNetModule

# Initialize U-Net module
unet_module = UNetModule(config)
```

#### Methods

##### `create_model() -> tf.keras.Model`

Create a U-Net model for segmentation.

**Returns:**
- `tf.keras.Model`: Compiled U-Net model

**Example:**
```python
# Create U-Net model
model = unet_module.create_model()
print(f"U-Net model created with {model.count_params()} parameters")
```

##### `predict_segmentation(images: np.ndarray) -> np.ndarray`

Predict road segmentation from satellite images.

**Parameters:**
- `images` (np.ndarray): Input satellite images

**Returns:**
- `np.ndarray`: Segmentation masks

**Example:**
```python
# Predict segmentation
masks = unet_module.predict_segmentation(satellite_images)
print(f"Generated {masks.shape[0]} segmentation masks")
```

### ModelManager

Manages model training, evaluation, and persistence.

#### Constructor

```python
ModelManager(config: Dict[str, Any])
```

**Parameters:**
- `config` (Dict[str, Any]): Configuration dictionary containing model settings

**Example:**
```python
from apa.models.base import ModelManager

# Initialize model manager
model_manager = ModelManager(config)
```

#### Methods

##### `train_model(model_type: str, X_train: np.ndarray, y_train: np.ndarray) -> Dict[str, Any]`

Train a model of specified type.

**Parameters:**
- `model_type` (str): Type of model to train ('cnn', 'unet')
- `X_train` (np.ndarray): Training features
- `y_train` (np.ndarray): Training labels

**Returns:**
- `Dict[str, Any]`: Training results

**Example:**
```python
# Train model
results = model_manager.train_model('cnn', X_train, y_train)
print(f"Training completed with accuracy: {results['accuracy']:.3f}")
```

##### `save_model(model: tf.keras.Model, path: str) -> None`

Save a trained model.

**Parameters:**
- `model` (tf.keras.Model): Model to save
- `path` (str): Path where to save the model

**Example:**
```python
# Save model
model_manager.save_model(trained_model, 'models/pavement_model.h5')
```

##### `load_model(path: str) -> tf.keras.Model`

Load a trained model.

**Parameters:**
- `path` (str): Path to the saved model

**Returns:**
- `tf.keras.Model`: Loaded model

**Example:**
```python
# Load model
model = model_manager.load_model('models/pavement_model.h5')
```

## 🎯 Usage Examples

### CNN Model Training

```python
from apa.models.cnn import CNNModule
import numpy as np

def train_cnn_model():
    """Train a CNN model for pavement classification."""
    
    # Initialize CNN module
    cnn_module = CNNModule(config)
    
    # Create model
    model = cnn_module.create_model()
    
    # Prepare training data
    X_train = np.random.rand(1000, 64, 64, 12)
    y_train = np.random.randint(0, 4, 1000)
    X_val = np.random.rand(200, 64, 64, 12)
    y_val = np.random.randint(0, 4, 200)
    
    # Train model
    history = cnn_module.train_model(X_train, y_train, X_val, y_val)
    
    return model, history

# Usage
model, history = train_cnn_model()
```

### U-Net Road Segmentation

```python
from apa.models.unet import UNetModule

def road_segmentation():
    """Perform road segmentation using U-Net."""
    
    # Initialize U-Net module
    unet_module = UNetModule(config)
    
    # Create model
    model = unet_module.create_model()
    
    # Load satellite images
    satellite_images = load_satellite_images()
    
    # Predict segmentation
    road_masks = unet_module.predict_segmentation(satellite_images)
    
    return road_masks

# Usage
masks = road_segmentation()
```

## 🔗 Related Documentation

- [Main API Documentation](../index.md)
- [Data Management API](../data/)
- [Pipeline API](../pipeline/)
- [Utilities API](../utils/)
