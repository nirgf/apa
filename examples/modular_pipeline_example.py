"""
Example demonstrating how to create custom modular pipelines.

This script shows how to build custom pipelines by composing
different stages using the common API.
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
from apa.common import DataContainer
from apa.modules import (
    HyperspectralDataImporter, GroundTruthDataImporter,
    ROIProcessor, RoadExtractor, PCISegmenter, DataPreprocessor,
    UNetModel, CNNModel, ModularPipeline
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_data_import_pipeline():
    """Create a pipeline focused on data import and validation."""
    print("=== Data Import Pipeline ===")
    
    # Create modular pipeline
    pipeline = ModularPipeline("data_import_pipeline")
    
    # Add data import stages
    hyperspectral_importer = HyperspectralDataImporter({
        'input_path': 'data/sample_hyperspectral',
        'filename_NED': 'NED.h5',
        'filename_RGB': 'RGB.h5'
    })
    
    ground_truth_importer = GroundTruthDataImporter({
        'excel_path': 'data/Detroit/Pavement_Condition.csv'
    })
    
    # Add stages to pipeline
    pipeline.add_custom_stage('hyperspectral_import', hyperspectral_importer)
    pipeline.add_custom_stage('ground_truth_import', ground_truth_importer)
    
    # Create dummy data for testing
    dummy_data = DataContainer(
        data={'dummy': 'data'},
        data_type='dummy',
        metadata={'source': 'test'}
    )
    
    # Run pipeline
    config = {
        'hyperspectral_import': {},
        'ground_truth_import': {}
    }
    
    try:
        result = pipeline.run_pipeline(dummy_data, config)
        print(f"✓ Data import pipeline: {'SUCCESS' if result.success else 'FAILED'}")
        return result
        
    except Exception as e:
        print(f"✗ Data import pipeline failed: {e}")
        return None


def create_processing_pipeline():
    """Create a pipeline focused on data processing."""
    print("\n=== Data Processing Pipeline ===")
    
    # Create modular pipeline
    pipeline = ModularPipeline("processing_pipeline")
    
    # Add processing stages
    roi_processor = ROIProcessor({
        'roi_bounds': [42.3, 42.4, -83.0, -82.9]
    })
    
    road_extractor = RoadExtractor()
    pci_segmenter = PCISegmenter()
    data_preprocessor = DataPreprocessor({
        'crop_size': 32,
        'overlap': 0.1
    })
    
    # Add stages with dependencies
    pipeline.add_custom_stage('roi_processing', roi_processor)
    pipeline.add_custom_stage('road_extraction', road_extractor, dependencies=['roi_processing'])
    pipeline.add_custom_stage('pci_segmentation', pci_segmenter, dependencies=['road_extraction'])
    pipeline.add_custom_stage('data_preparation', data_preprocessor, dependencies=['pci_segmentation'])
    
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
        metadata={'source': 'dummy_processing_data'}
    )
    
    # Run pipeline
    config = {
        'roi_processing': {'roi_bounds': [42.3, 42.4, -83.0, -82.9]},
        'road_extraction': {},
        'pci_segmentation': {},
        'data_preparation': {'crop_size': 32, 'overlap': 0.1}
    }
    
    try:
        result = pipeline.run_pipeline(data, config)
        print(f"✓ Processing pipeline: {'SUCCESS' if result.success else 'FAILED'}")
        if result.success:
            print(f"  Total time: {result.total_time:.2f}s")
            print(f"  Stages completed: {len(result.stage_results)}")
        return result
        
    except Exception as e:
        print(f"✗ Processing pipeline failed: {e}")
        return None


def create_model_training_pipeline():
    """Create a pipeline focused on model training."""
    print("\n=== Model Training Pipeline ===")
    
    # Create modular pipeline
    pipeline = ModularPipeline("model_training_pipeline")
    
    # Add model stages
    unet_model = UNetModel({
        'input_size': (32, 32, 12),
        'n_classes': 4,
        'epochs': 3,  # Short training for example
        'batch_size': 16,
        'learning_rate': 1e-3
    })
    
    cnn_model = CNNModel({
        'input_size': (32, 32, 12),
        'n_classes': 4,
        'epochs': 3,  # Short training for example
        'batch_size': 16,
        'learning_rate': 1e-3
    })
    
    # Add stages
    pipeline.add_custom_stage('unet_training', unet_model)
    pipeline.add_custom_stage('cnn_training', cnn_model)
    
    # Create dummy training data
    dummy_segments = np.random.rand(50, 32, 32, 12).astype(np.float32)
    
    data = DataContainer(
        data={'cropped_segments': dummy_segments},
        data_type='preprocessed',
        metadata={'source': 'dummy_training_data'}
    )
    
    # Run pipeline
    config = {
        'unet_training': {
            'input_size': (32, 32, 12),
            'n_classes': 4,
            'epochs': 2,
            'batch_size': 16
        },
        'cnn_training': {
            'input_size': (32, 32, 12),
            'n_classes': 4,
            'epochs': 2,
            'batch_size': 16
        }
    }
    
    try:
        result = pipeline.run_pipeline(data, config)
        print(f"✓ Model training pipeline: {'SUCCESS' if result.success else 'FAILED'}")
        if result.success:
            print(f"  Total time: {result.total_time:.2f}s")
            print(f"  Models trained: {len(result.stage_results)}")
        return result
        
    except Exception as e:
        print(f"✗ Model training pipeline failed: {e}")
        return None


def create_custom_workflow():
    """Create a custom workflow combining different stages."""
    print("\n=== Custom Workflow ===")
    
    # Create modular pipeline
    pipeline = ModularPipeline("custom_workflow")
    
    # Add custom stages
    roi_processor = ROIProcessor()
    data_preprocessor = DataPreprocessor()
    unet_model = UNetModel()
    
    # Add stages with custom dependencies
    pipeline.add_custom_stage('roi', roi_processor)
    pipeline.add_custom_stage('preprocessing', data_preprocessor, dependencies=['roi'])
    pipeline.add_custom_stage('training', unet_model, dependencies=['preprocessing'])
    
    # Create dummy data
    dummy_image = np.random.rand(80, 80, 12).astype(np.float32)
    dummy_lon = np.random.rand(80, 80) * 0.1 - 83.0
    dummy_lat = np.random.rand(80, 80) * 0.1 + 42.3
    
    data = DataContainer(
        data={
            'hyperspectral_image': dummy_image,
            'longitude_matrix': dummy_lon,
            'latitude_matrix': dummy_lat
        },
        data_type='hyperspectral',
        metadata={'source': 'custom_workflow_data'}
    )
    
    # Run custom workflow
    config = {
        'roi': {'roi_bounds': [42.3, 42.4, -83.0, -82.9]},
        'preprocessing': {'crop_size': 16, 'overlap': 0.1},
        'training': {
            'input_size': (16, 16, 12),
            'n_classes': 4,
            'epochs': 1,  # Very short for example
            'batch_size': 8
        }
    }
    
    try:
        result = pipeline.run_pipeline(data, config)
        print(f"✓ Custom workflow: {'SUCCESS' if result.success else 'FAILED'}")
        if result.success:
            print(f"  Total time: {result.total_time:.2f}s")
            print(f"  Stages completed: {len(result.stage_results)}")
            
            # Print detailed results
            for stage_name, stage_result in result.stage_results.items():
                print(f"    {stage_name}: {'SUCCESS' if stage_result.success else 'FAILED'}")
                if hasattr(stage_result, 'processing_time'):
                    print(f"      Processing time: {stage_result.processing_time:.2f}s")
        
        return result
        
    except Exception as e:
        print(f"✗ Custom workflow failed: {e}")
        return None


def demonstrate_pipeline_management():
    """Demonstrate pipeline management features."""
    print("\n=== Pipeline Management ===")
    
    # Create pipeline
    pipeline = ModularPipeline("management_demo")
    
    # Add some stages
    roi_processor = ROIProcessor()
    road_extractor = RoadExtractor()
    
    pipeline.add_custom_stage('roi', roi_processor)
    pipeline.add_custom_stage('road', road_extractor, dependencies=['roi'])
    
    # Show pipeline info
    info = pipeline.get_pipeline_info()
    print(f"✓ Pipeline created: {info['pipeline_name']}")
    print(f"  Available stages: {info['stages']}")
    print(f"  Execution order: {info['execution_order']}")
    print(f"  Dependencies: {info['dependencies']}")
    
    # Show available stage types
    available_stages = pipeline.get_available_stages()
    print(f"  Available stage types: {available_stages}")
    
    # Create stage from type
    try:
        new_stage = pipeline.create_stage_from_type('roi_processing', 'new_roi', {})
        print(f"✓ Created stage from type: {new_stage.name}")
    except Exception as e:
        print(f"✗ Failed to create stage from type: {e}")
    
    # Remove a stage
    pipeline.remove_stage('road')
    updated_info = pipeline.get_pipeline_info()
    print(f"✓ After removing 'road' stage: {updated_info['stages']}")


def main():
    """Run all modular pipeline examples."""
    print("APA Modular Pipeline Examples")
    print("=" * 50)
    
    try:
        # Run different pipeline examples
        create_data_import_pipeline()
        create_processing_pipeline()
        create_model_training_pipeline()
        create_custom_workflow()
        demonstrate_pipeline_management()
        
        print("\n" + "=" * 50)
        print("✓ All modular pipeline examples completed successfully!")
        
    except Exception as e:
        print(f"\n✗ Examples failed: {e}")
        logger.exception("Modular pipeline example execution failed")


if __name__ == "__main__":
    main()




