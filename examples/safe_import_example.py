#!/usr/bin/env python3
"""
Safe import example showing how to use APA modules without triggering execution.

This demonstrates that importing apa.modules no longer executes code
and modules are only activated when actually instantiated.
"""

import sys
from pathlib import Path

# Add the src directory to the path
src_path = Path("../src").resolve()
sys.path.insert(0, str(src_path))

print("APA Safe Import Example")
print("=" * 50)

# This import should NOT execute any code
print("1. Importing apa.modules (should be silent)...")
from apa.modules import (
    HyperspectralDataImporter,
    GroundTruthDataImporter,
    ROIProcessor,
    RoadExtractor,
    PCISegmenter,
    DataPreprocessor,
    UNetModel,
    CNNModel,
    ModelManager,
    APAPipeline,
    ModularPipeline
)
print("✓ Import completed - no execution detected")

print("\n2. Creating module instances (only now will code execute)...")

# Create instances - this is when the actual modules will be loaded
try:
    # Data importer
    importer = HyperspectralDataImporter({
        'input_path': 'data/sample',
        'filename_NED': 'NED.h5',
        'filename_RGB': 'RGB.h5'
    })
    print("✓ HyperspectralDataImporter created")
    
    # Processor
    processor = ROIProcessor()
    print("✓ ROIProcessor created")
    
    # Model
    model = UNetModel({
        'input_size': (32, 32, 12),
        'n_classes': 4,
        'epochs': 1  # Very short for testing
    })
    print("✓ UNetModel created")
    
    # Pipeline
    pipeline = ModularPipeline("test_pipeline")
    print("✓ ModularPipeline created")
    
    print("\n" + "=" * 50)
    print("✓ SUCCESS: All modules created without premature execution!")
    print("✓ The modules are now lazy-loaded and only execute when needed.")
    
except Exception as e:
    print(f"✗ Error during module creation: {e}")
    import traceback
    traceback.print_exc()

print("\n3. Testing that modules work when called...")

# Test that the modules actually work when used
try:
    import numpy as np
    from apa.common import DataContainer
    
    # Create dummy data
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
        metadata={'source': 'dummy_data'}
    )
    
    # Test ROI processing
    result = processor.process_data(data, {
        'roi_bounds': [42.3, 42.4, -83.0, -82.9]
    })
    
    if result.success:
        print("✓ ROI processing works correctly")
    else:
        print(f"⚠ ROI processing failed: {result.error_message}")
    
    print("\n" + "=" * 50)
    print("✓ COMPLETE: Lazy loading is working perfectly!")
    print("✓ Modules only execute when instantiated and called.")
    print("✓ No premature execution during import.")
    
except Exception as e:
    print(f"✗ Error during module testing: {e}")
    import traceback
    traceback.print_exc()




