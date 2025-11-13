"""
Basic usage example for APA modules.

This example demonstrates how to use the APA API with plug-and-play modules.
"""

from apa.common import DataContainer
from apa.modules import (
    HyperspectralDataImporter,
    ROIProcessor,
    RoadExtractor,
    DataPreprocessor,
    UNetModel,
    ModularPipeline,
)


def example_data_import():
    """Example: Import hyperspectral data."""
    print("=== Example: Data Import ===")
    
    # Create importer with configuration
    importer = HyperspectralDataImporter({
        'input_path': 'data/Detroit',
        'filename_NED': 'NED.h5',
        'filename_RGB': 'RGB.h5',
        'dataset': 1,  # venus_Detroit
    })
    
    # Load data
    try:
        data = importer.load_data()
        print(f"✓ Data loaded: {data.data_type}")
        print(f"  Shape: {data.shape}")
        print(f"  Metadata: {data.metadata}")
    except Exception as e:
        print(f"✗ Error: {str(e)}")
        # For demo purposes, create dummy data
        import numpy as np
        data = DataContainer(
            data=np.random.rand(100, 100, 12),
            metadata={'source': 'demo'},
            data_type='hyperspectral'
        )
        print(f"  Using dummy data for demonstration")
    
    return data


def example_processing(data: DataContainer):
    """Example: Process data through multiple stages."""
    print("\n=== Example: Data Processing ===")
    
    # Stage 1: ROI Processing
    roi_processor = ROIProcessor({
        'roi_bounds': [42.3, 42.4, -83.0, -82.9],
    })
    
    try:
        roi_result = roi_processor.process_data(data)
        print(f"✓ ROI processing completed")
        print(f"  Processing time: {roi_result.processing_time:.2f}s")
        data = roi_result.data
    except Exception as e:
        print(f"✗ ROI processing error: {str(e)}")
    
    # Stage 2: Road Extraction
    road_extractor = RoadExtractor({
        'method': 'osm',
        'threshold': 0.5,
    })
    
    try:
        road_result = road_extractor.process_data(data)
        print(f"✓ Road extraction completed")
        data = road_result.data
    except Exception as e:
        print(f"✗ Road extraction error: {str(e)}")
    
    # Stage 3: Data Preprocessing
    preprocessor = DataPreprocessor({
        'normalize': True,
        'patch_size': (32, 32),
    })
    
    try:
        prep_result = preprocessor.process_data(data)
        print(f"✓ Data preprocessing completed")
        data = prep_result.data
    except Exception as e:
        print(f"✗ Preprocessing error: {str(e)}")
    
    return data


def example_model_training(data: DataContainer):
    """Example: Train a model."""
    print("\n=== Example: Model Training ===")
    
    # Create model
    model = UNetModel({
        'input_size': (32, 32, 12),
        'n_classes': 4,
        'epochs': 10,  # Reduced for demo
        'batch_size': 32,
    })
    
    try:
        # Train model
        train_result = model.train(data, {
            'epochs': 5,  # Override for quick demo
        })
        print(f"✓ Model training completed")
        print(f"  Metrics: {train_result.metrics}")
        
        # Make predictions
        pred_result = model.predict(data)
        print(f"✓ Predictions generated")
        print(f"  Predictions shape: {pred_result.predictions.shape if hasattr(pred_result.predictions, 'shape') else 'N/A'}")
        
    except Exception as e:
        print(f"✗ Model error: {str(e)}")
        print("  (This is expected - model implementation is a template)")


def example_custom_pipeline():
    """Example: Create a custom pipeline."""
    print("\n=== Example: Custom Pipeline ===")
    
    # Create modular pipeline
    pipeline = ModularPipeline("my_custom_pipeline")
    
    # Add stages
    pipeline.add_custom_stage(
        'import',
        HyperspectralDataImporter({'input_path': 'data/'}),
        dependencies=[]
    )
    
    pipeline.add_custom_stage(
        'roi',
        ROIProcessor({'roi_bounds': None}),
        dependencies=['import']
    )
    
    pipeline.add_custom_stage(
        'preprocess',
        DataPreprocessor({'normalize': True}),
        dependencies=['roi']
    )
    
    try:
        # Run pipeline
        result = pipeline.run_pipeline(None, {})
        print(f"✓ Pipeline executed")
        print(f"  Success: {result.success}")
        print(f"  Execution time: {result.execution_time:.2f}s")
        print(f"  Stages executed: {len(result.stage_results)}")
    except Exception as e:
        print(f"✗ Pipeline error: {str(e)}")
        print("  (This is expected - implementations are templates)")


def main():
    """Run all examples."""
    print("APA - Advanced Pavement Analytics")
    print("=" * 50)
    
    # Example 1: Data import
    data = example_data_import()
    
    # Example 2: Processing
    processed_data = example_processing(data)
    
    # Example 3: Model training
    example_model_training(processed_data)
    
    # Example 4: Custom pipeline
    example_custom_pipeline()
    
    print("\n" + "=" * 50)
    print("Examples completed!")
    print("\nNote: These are template implementations.")
    print("Replace the TODO sections with your actual logic.")


if __name__ == "__main__":
    main()

