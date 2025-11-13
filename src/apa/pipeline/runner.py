"""
Pipeline runner for APA.

Provides main pipeline execution functionality.
"""

from typing import Any, Dict, Optional

from apa.common import DataContainer, PipelineResult
from apa.modules.pipelines import APAPipeline as BaseAPAPipeline


class APAPipeline(BaseAPAPipeline):
    """
    Main APA pipeline runner.
    
    Wrapper around the base APAPipeline with additional convenience methods.
    """
    
    def run(self, config: Optional[Dict[str, Any]] = None) -> PipelineResult:
        """
        Run the complete pipeline.
        
        Args:
            config: Optional pipeline configuration
            
        Returns:
            PipelineResult with all stage results
        """
        if config:
            self.config.update(config)
            self._setup_stages()  # Re-setup with new config
        
        return self.run_pipeline(None, self.config)
    
    def get_results(self) -> Dict[str, Any]:
        """
        Get results from last pipeline run.
        
        Returns:
            Dictionary of stage results
        """
        # This would typically store results from the last run
        # For now, return empty dict
        return {}

