# Pipeline Orchestration API

The `apa.pipeline` module provides the main orchestration system for the APA workflow, managing the execution of individual pipeline stages and coordinating the complete pavement analytics process.

## 📦 Module Overview

```python
from apa.pipeline import APAPipeline, PipelineStage
```

The pipeline module handles:
- Stage-based pipeline execution
- Error handling and logging
- Progress tracking and monitoring
- Result management and aggregation
- Pipeline state management

## 🔧 Classes

### APAPipeline

The main pipeline orchestrator that manages the execution of the complete APA workflow.

#### Constructor

```python
APAPipeline(config: Dict[str, Any])
```

**Parameters:**
- `config` (Dict[str, Any]): Configuration dictionary containing pipeline settings

**Example:**
```python
from apa.pipeline.runner import APAPipeline

# Initialize pipeline with configuration
pipeline = APAPipeline(config)
```

#### Methods

##### `run(stages: Optional[List[str]] = None) -> None`

Run the complete APA pipeline or specified stages.

**Parameters:**
- `stages` (Optional[List[str]]): List of stage names to run. If None, runs all stages.

**Raises:**
- `Exception`: If any stage execution fails

**Example:**
```python
# Run complete pipeline
pipeline.run()

# Run specific stages
pipeline.run(['data_import', 'roi_processing', 'road_extraction'])
```

##### `get_results() -> Dict[str, Any]`

Get pipeline execution results.

**Returns:**
- `Dict[str, Any]`: Dictionary containing pipeline results

**Example:**
```python
# Get results after pipeline execution
results = pipeline.get_results()

print(f"ROIs processed: {len(results['processed_rois'])}")
print(f"Stages completed: {', '.join(results['stages_completed'])}")
```

#### Pipeline Stages

The APA pipeline consists of the following stages:

1. **`data_import`** - Import hyperspectral imagery and ground truth data
2. **`roi_processing`** - Process regions of interest
3. **`road_extraction`** - Extract road networks from imagery
4. **`pci_segmentation`** - Perform PCI segmentation
5. **`data_preparation`** - Prepare data for neural network training
6. **`model_training`** - Train machine learning models

### PipelineStage

Represents a single stage in the APA pipeline with execution tracking and error handling.

#### Constructor

```python
PipelineStage(name: str, execute_func: Callable[[], None])
```

**Parameters:**
- `name` (str): Name of the stage
- `execute_func` (Callable[[], None]): Function to execute for this stage

**Example:**
```python
from apa.pipeline.stages import PipelineStage

def my_stage_function():
    """Custom stage function."""
    print("Executing custom stage...")
    # Stage implementation here

# Create custom stage
stage = PipelineStage("custom_stage", my_stage_function)
```

#### Methods

##### `execute() -> None`

Execute this pipeline stage.

**Raises:**
- `Exception`: If stage execution fails

**Example:**
```python
# Execute stage
try:
    stage.execute()
    print(f"Stage {stage.name} completed successfully")
except Exception as e:
    print(f"Stage {stage.name} failed: {e}")
```

##### `reset() -> None`

Reset the stage completion status.

**Example:**
```python
# Reset stage status
stage.reset()
print(f"Stage status: {stage.completed}")  # False
```

## 🎯 Usage Examples

### Basic Pipeline Execution

```python
from apa.pipeline.runner import APAPipeline
from apa.config.manager import ConfigManager

def run_basic_pipeline():
    """Run the basic APA pipeline."""
    
    # Load configuration
    config_manager = ConfigManager()
    config = config_manager.load_config('configs/detroit.yaml')
    
    # Initialize pipeline
    pipeline = APAPipeline(config['config'])
    
    # Run complete pipeline
    print("Starting APA pipeline...")
    pipeline.run()
    
    # Get results
    results = pipeline.get_results()
    print(f"Pipeline completed. Processed {len(results['processed_rois'])} ROIs")
    
    return results

# Usage
results = run_basic_pipeline()
```

### Selective Stage Execution

```python
def run_partial_pipeline():
    """Run only specific pipeline stages."""
    
    config_manager = ConfigManager()
    config = config_manager.load_config('configs/detroit.yaml')
    pipeline = APAPipeline(config['config'])
    
    # Run only data import and ROI processing
    stages_to_run = ['data_import', 'roi_processing']
    print(f"Running stages: {', '.join(stages_to_run)}")
    
    pipeline.run(stages_to_run)
    
    # Check which stages completed
    results = pipeline.get_results()
    print(f"Completed stages: {', '.join(results['stages_completed'])}")
    
    return results

# Usage
results = run_partial_pipeline()
```

### Pipeline with Error Handling

```python
def run_pipeline_with_error_handling():
    """Run pipeline with comprehensive error handling."""
    
    try:
        config_manager = ConfigManager()
        config = config_manager.load_config('configs/detroit.yaml')
        pipeline = APAPipeline(config['config'])
        
        # Run pipeline
        pipeline.run()
        
        # Get results
        results = pipeline.get_results()
        
        print("✓ Pipeline completed successfully")
        return results
        
    except FileNotFoundError as e:
        print(f"✗ Configuration file not found: {e}")
        return None
    except ValueError as e:
        print(f"✗ Configuration validation error: {e}")
        return None
    except Exception as e:
        print(f"✗ Pipeline execution error: {e}")
        return None

# Usage
results = run_pipeline_with_error_handling()
if results:
    print(f"Processed {len(results['processed_rois'])} ROIs")
```

### Custom Pipeline Stage

```python
from apa.pipeline.stages import PipelineStage

def custom_data_processing():
    """Custom data processing stage."""
    print("Executing custom data processing...")
    
    # Custom processing logic here
    # For example, apply custom filters, transformations, etc.
    
    print("Custom data processing completed")

def run_pipeline_with_custom_stage():
    """Run pipeline with custom stage."""
    
    config_manager = ConfigManager()
    config = config_manager.load_config('configs/detroit.yaml')
    pipeline = APAPipeline(config['config'])
    
    # Add custom stage
    custom_stage = PipelineStage("custom_processing", custom_data_processing)
    pipeline.stages.append(custom_stage)
    
    # Run pipeline including custom stage
    pipeline.run()
    
    return pipeline.get_results()

# Usage
results = run_pipeline_with_custom_stage()
```

### Pipeline Monitoring and Logging

```python
import logging

def setup_pipeline_logging():
    """Set up detailed logging for pipeline execution."""
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('pipeline.log'),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger('apa_pipeline')

def run_pipeline_with_logging():
    """Run pipeline with detailed logging."""
    
    # Set up logging
    logger = setup_pipeline_logging()
    
    try:
        config_manager = ConfigManager()
        config = config_manager.load_config('configs/detroit.yaml')
        pipeline = APAPipeline(config['config'])
        
        logger.info("Starting APA pipeline execution")
        
        # Run pipeline
        pipeline.run()
        
        # Get results
        results = pipeline.get_results()
        
        logger.info(f"Pipeline completed successfully. Processed {len(results['processed_rois'])} ROIs")
        
        return results
        
    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}")
        raise

# Usage
results = run_pipeline_with_logging()
```

## 🔧 Advanced Usage

### Pipeline State Management

```python
class PipelineStateManager:
    """Manage pipeline state and checkpoints."""
    
    def __init__(self, pipeline):
        self.pipeline = pipeline
        self.checkpoint_file = 'pipeline_checkpoint.json'
    
    def save_checkpoint(self):
        """Save current pipeline state."""
        import json
        
        state = {
            'completed_stages': [stage.name for stage in self.pipeline.stages if stage.completed],
            'config': self.pipeline.config,
            'results': self.pipeline.get_results() if hasattr(self.pipeline, 'processed_rois') else None
        }
        
        with open(self.checkpoint_file, 'w') as f:
            json.dump(state, f, indent=2)
        
        print(f"Checkpoint saved to {self.checkpoint_file}")
    
    def load_checkpoint(self):
        """Load pipeline state from checkpoint."""
        import json
        
        try:
            with open(self.checkpoint_file, 'r') as f:
                state = json.load(f)
            
            print(f"Checkpoint loaded. Completed stages: {state['completed_stages']}")
            return state
            
        except FileNotFoundError:
            print("No checkpoint found")
            return None
    
    def resume_pipeline(self):
        """Resume pipeline from checkpoint."""
        state = self.load_checkpoint()
        
        if state:
            # Mark completed stages
            for stage in self.pipeline.stages:
                if stage.name in state['completed_stages']:
                    stage.completed = True
            
            # Run remaining stages
            remaining_stages = [stage.name for stage in self.pipeline.stages if not stage.completed]
            if remaining_stages:
                print(f"Resuming pipeline. Remaining stages: {', '.join(remaining_stages)}")
                self.pipeline.run(remaining_stages)
            else:
                print("All stages already completed")

# Usage
config_manager = ConfigManager()
config = config_manager.load_config('configs/detroit.yaml')
pipeline = APAPipeline(config['config'])

state_manager = PipelineStateManager(pipeline)

# Save checkpoint after each stage
for stage in pipeline.stages:
    stage.execute()
    state_manager.save_checkpoint()
```

### Parallel Pipeline Execution

```python
import concurrent.futures
from typing import List

def run_parallel_pipelines(configs: List[Dict[str, Any]]):
    """Run multiple pipelines in parallel."""
    
    def run_single_pipeline(config):
        """Run a single pipeline with given configuration."""
        try:
            pipeline = APAPipeline(config)
            pipeline.run()
            return {
                'config': config,
                'results': pipeline.get_results(),
                'status': 'success'
            }
        except Exception as e:
            return {
                'config': config,
                'results': None,
                'status': f'error: {str(e)}'
            }
    
    # Run pipelines in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(run_single_pipeline, config) for config in configs]
        results = [future.result() for future in concurrent.futures.as_completed(futures)]
    
    return results

# Usage
configs = [
    load_config('configs/detroit.yaml'),
    load_config('configs/kiryat_ata.yaml')
]

results = run_parallel_pipelines(configs)
for result in results:
    print(f"Pipeline status: {result['status']}")
```

### Pipeline Performance Monitoring

```python
import time
from typing import Dict, Any

class PipelineProfiler:
    """Profile pipeline execution performance."""
    
    def __init__(self):
        self.stage_times: Dict[str, float] = {}
        self.start_time = None
    
    def start_stage(self, stage_name: str):
        """Start timing a stage."""
        self.stage_times[stage_name] = time.time()
        print(f"Starting stage: {stage_name}")
    
    def end_stage(self, stage_name: str):
        """End timing a stage."""
        if stage_name in self.stage_times:
            duration = time.time() - self.stage_times[stage_name]
            print(f"Completed stage: {stage_name} (took {duration:.2f}s)")
            return duration
        return 0
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get performance report."""
        total_time = sum(self.stage_times.values())
        
        return {
            'total_time': total_time,
            'stage_times': self.stage_times,
            'stage_percentages': {
                stage: (time / total_time) * 100 
                for stage, time in self.stage_times.items()
            }
        }

def run_pipeline_with_profiling():
    """Run pipeline with performance profiling."""
    
    profiler = PipelineProfiler()
    
    config_manager = ConfigManager()
    config = config_manager.load_config('configs/detroit.yaml')
    pipeline = APAPipeline(config['config'])
    
    # Profile each stage
    for stage in pipeline.stages:
        profiler.start_stage(stage.name)
        stage.execute()
        profiler.end_stage(stage.name)
    
    # Get performance report
    report = profiler.get_performance_report()
    print(f"Total pipeline time: {report['total_time']:.2f}s")
    
    return pipeline.get_results(), report

# Usage
results, performance_report = run_pipeline_with_profiling()
```

## 🚨 Error Handling

### Common Pipeline Errors

1. **Configuration Errors**
   ```python
   # Error: Invalid configuration
   ValueError: Configuration validation failed: Missing required section: data
   ```

2. **Data Import Errors**
   ```python
   # Error: Data import failed
   FileNotFoundError: Data file not found: data/satellite_images/image.tif
   ```

3. **Stage Execution Errors**
   ```python
   # Error: Stage execution failed
   Exception: Error in stage data_import: Invalid data format
   ```

### Error Handling Best Practices

```python
def robust_pipeline_execution():
    """Run pipeline with robust error handling."""
    
    try:
        # Load and validate configuration
        config_manager = ConfigManager()
        config = config_manager.load_config('configs/detroit.yaml')
        
        # Initialize pipeline
        pipeline = APAPipeline(config['config'])
        
        # Run pipeline with error handling
        pipeline.run()
        
        return pipeline.get_results()
        
    except FileNotFoundError as e:
        print(f"✗ File not found: {e}")
        return None
    except ValueError as e:
        print(f"✗ Configuration error: {e}")
        return None
    except Exception as e:
        print(f"✗ Pipeline error: {e}")
        return None

# Usage
results = robust_pipeline_execution()
```

## 🔗 Related Documentation

- [Main API Documentation](../index.md)
- [Configuration API](../config/)
- [Data Management API](../data/)
- [Utilities API](../utils/)
