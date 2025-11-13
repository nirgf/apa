"""
Examples demonstrating the common API usage for APA modules.

This script shows how to use the standardized interfaces to create
modular, independent pipeline stages.
"""

import sys
from pathlib import Path
import numpy as np
import logging

# Add the src directory to the path
project_root = Path(__file__).parent.parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

# Import common API components
from apa.common import (
    DataContainer, ProcessingResult, ModelResult, PipelineResult,
    ValidationError, ProcessingError, ModelError
)
from apa.modules import (
    HyperspectralDataImporter, GroundTruthDataImporter,
    ROIProcessor, RoadExtractor, PCISegmenter, DataPreprocessor,
    UNetModel, CNNModel, ModelManager,
    APAPipeline, ModularPipeline
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def example_data_import():
    """Example: Using data importers with the common API."""
    print("=== Data Import Example ===")
    
    # Create hyperspectral data importer
    importer_config = {
        'input_path': 'data/sample_hyperspectral',
        'filename_NED': 'NED.h5',
        'filename_RGB': 'RGB.h5'
    }
    
    importer = HyperspectralDataImporter(importer_config)
    
    # Load data
    try:
        data = importer.load_data(importer_config)
        print(f"✓ Data loaded successfully: {data.get_shape()}")
        print(f"  Data type: {data.data_type}")
        print(f"  Metadata: {data.metadata}")
        
        # Validate data
        is_valid = importer.validate_data(data)
        print(f"✓ Data validation: {'PASSED' if is_valid else 'FAILED'}")
        
        # Get data info
        info = importer.get_data_info(data)
        print(f"✓ Data info: {info}")
        
    except Exception as e:
        print(f"✗ Data import failed: {e}")


def example_data_processing():
    """Example: Using data processors with the common API."""
    print("\n=== Data Processing Example ===")
    
    # Create dummy hyperspectral data
    dummy_image = np.random.rand(100, 100, 12).astype(np.float32)
    dummy_lon = np.random.rand(100, 100) * 0.1 - 83.0
    dummy_lat = np.random.rand(100, 100) * 0.1 + 42.3
    
    data = DataContainer(
        data={
            'hyperspectral_image': dummy_image,
            'longitude_matrix': dummy_lon,
            'latitude_matrix': dummy_lat
        },
        data_type='hyperspectral',
        metadata={'source': 'dummy_data'}
    )
    
    # ROI Processing
    roi_processor = ROIProcessor()
    roi_config = {
        'roi_bounds': [42.3, 42.4, -83.0, -82.9]  # lat_min, lat_max, lon_min, lon_max
    }
    
    try:
        result = roi_processor.process_data(data, roi_config)
        print(f"✓ ROI processing: {'SUCCESS' if result.success else 'FAILED'}")
        if result.success:
            print(f"  Processing time: {result.processing_time:.2f}s")
            print(f"  Output shape: {result.processed_data.get_shape()}")
        
    except Exception as e:
        print(f"✗ ROI processing failed: {e}")
    
    # Road Extraction
    road_extractor = RoadExtractor()
    road_config = {}
    
    try:
        result = road_extractor.process_data(data, road_config)
        print(f"✓ Road extraction: {'SUCCESS' if result.success else 'FAILED'}")
        if result.success:
            print(f"  Road coverage: {result.processed_data.metadata.get('road_coverage', 'N/A')}")
        
    except Exception as e:
        print(f"✗ Road extraction failed: {e}")


def example_model_usage():
    """Example: Using models with the common API."""
    print("\n=== Model Usage Example ===")
    
    # Create dummy training data
    dummy_segments = np.random.rand(100, 32, 32, 12).astype(np.float32)
    
    data = DataContainer(
        data={'cropped_segments': dummy_segments},
        data_type='preprocessed',
        metadata={'source': 'dummy_training_data'}
    )
    
    # U-Net Model
    unet_config = {
        'input_size': (32, 32, 12),
        'n_classes': 4,
        'epochs': 5,  # Short training for example
        'batch_size': 16,
        'learning_rate': 1e-3
    }
    
    unet_model = UNetModel(unet_config)
    
    try:
        # Train model
        train_result = unet_model.train(data, unet_config)
        print(f"✓ U-Net training: {'SUCCESS' if train_result.success else 'FAILED'}")
        if train_result.success:
            print(f"  Training accuracy: {train_result.metrics.get('accuracy', 'N/A')}")
            print(f"  Validation loss: {train_result.metrics.get('val_loss', 'N/A')}")
        
        # Make predictions
        pred_result = unet_model.predict(data)
        print(f"✓ U-Net prediction: {'SUCCESS' if pred_result.success else 'FAILED'}")
        if pred_result.success:
            print(f"  Predictions shape: {pred_result.predictions.shape}")
        
    except Exception as e:
        print(f"✗ U-Net operations failed: {e}")
    
    # Model Manager
    model_manager = ModelManager()
    model_manager.add_model('unet', unet_model)
    model_manager.set_active_model('unet')
    
    try:
        # Use model manager
        manager_result = model_manager.predict(data)
        print(f"✓ Model manager prediction: {'SUCCESS' if manager_result.success else 'FAILED'}")
        
    except Exception as e:
        print(f"✗ Model manager failed: {e}")


def example_pipeline_usage():
    """Example: Using pipelines with the common API."""
    print("\n=== Pipeline Usage Example ===")
    
    # Create dummy input data
    dummy_image = np.random.rand(50, 50, 12).astype(np.float32)
    dummy_lon = np.random.rand(50, 50) * 0.1 - 83.0
    dummy_lat = np.random.rand(50, 50) * 0.1 + 42.3
    
    data = DataContainer(
        data={
            'hyperspectral_image': dummy_image,
            'longitude_matrix': dummy_lon,
            'latitude_matrix': dummy_lat
        },
        data_type='hyperspectral',
        metadata={'source': 'dummy_pipeline_data'}
    )
    
    # APA Pipeline
    pipeline_config = {
        'roi_processing': {
            'roi_bounds': [42.3, 42.4, -83.0, -82.9]
        },
        'road_extraction': {},
        'pci_segmentation': {},
        'data_preparation': {
            'crop_size': 16,
            'overlap': 0.1
        },
        'model_training': {
            'input_size': (16, 16, 12),
            'n_classes': 4,
            'epochs': 2,  # Short training for example
            'batch_size': 8
        }
    }
    
    apa_pipeline = APAPipeline(pipeline_config)
    
    try:
        # Run pipeline
        result = apa_pipeline.run_pipeline(data, pipeline_config)
        print(f"✓ APA Pipeline: {'SUCCESS' if result.success else 'FAILED'}")
        if result.success:
            print(f"  Total time: {result.total_time:.2f}s")
            print(f"  Stages completed: {len(result.stage_results)}")
            
            # Print stage summaries
            for stage_name, stage_result in result.stage_results.items():
                print(f"    {stage_name}: {'SUCCESS' if stage_result.success else 'FAILED'}")
        
    except Exception as e:
        print(f"✗ APA Pipeline failed: {e}")
    
    # Modular Pipeline
    modular_pipeline = ModularPipeline("custom_pipeline")
    
    try:
        # Add custom stages
        roi_processor = ROIProcessor()
        road_extractor = RoadExtractor()
        
        modular_pipeline.add_custom_stage('roi', roi_processor)
        modular_pipeline.add_custom_stage('road', road_extractor, dependencies=['roi'])
        
        # Run modular pipeline
        modular_config = {
            'roi': {'roi_bounds': [42.3, 42.4, -83.0, -82.9]},
            'road': {}
        }
        
        result = modular_pipeline.run_pipeline(data, modular_config)
        print(f"✓ Modular Pipeline: {'SUCCESS' if result.success else 'FAILED'}")
        if result.success:
            print(f"  Total time: {result.total_time:.2f}s")
            print(f"  Stages completed: {len(result.stage_results)}")
        
    except Exception as e:
        print(f"✗ Modular Pipeline failed: {e}")


def example_error_handling():
    """Example: Error handling with the common API."""
    print("\n=== Error Handling Example ===")
    
    # Test validation errors
    try:
        # Create invalid data container
        invalid_data = DataContainer(
            data=None,  # Invalid data
            data_type='hyperspectral'
        )
        
        importer = HyperspectralDataImporter()
        importer.validate_data(invalid_data)
        
    except ValidationError as e:
        print(f"✓ Validation error caught: {e}")
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
    
    # Test processing errors
    try:
        # Create data with wrong type
        wrong_data = DataContainer(
            data={'wrong_key': 'wrong_value'},
            data_type='wrong_type'
        )
        
        processor = ROIProcessor()
        processor.validate_data(wrong_data)
        
    except ValidationError as e:
        print(f"✓ Processing validation error caught: {e}")
    except Exception as e:
        print(f"✗ Unexpected error: {e}")


def main():
    """Run all examples."""
    print("APA Common API Usage Examples")
    print("=" * 50)
    
    try:
        example_data_import()
        example_data_processing()
        example_model_usage()
        example_pipeline_usage()
        example_error_handling()
        
        print("\n" + "=" * 50)
        print("✓ All examples completed successfully!")
        
    except Exception as e:
        print(f"\n✗ Examples failed: {e}")
        logger.exception("Example execution failed")


if __name__ == "__main__":
    main()




