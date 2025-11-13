"""
Configuration manager for APA.

Handles loading, validation, and management of configuration files.
"""

from typing import Any, Dict, Optional
import yaml
import os
from pathlib import Path

from apa.common.exceptions import ConfigurationError


class ConfigManager:
    """
    Manager for APA configuration files.
    
    Provides functionality to load, validate, and manage YAML configuration files.
    """
    
    def __init__(self):
        """Initialize configuration manager."""
        self.default_config_path = Path(__file__).parent.parent.parent.parent / "configs"
    
    def load_config(self, config_path: str) -> Dict[str, Any]:
        """
        Load configuration from YAML file.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Configuration dictionary
            
        Raises:
            ConfigurationError if file cannot be loaded
        """
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise ConfigurationError(f"Configuration file not found: {config_path}")
        
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            if config is None:
                config = {}
            
            return {'config': config['config'], 'path': str(config_path)}
        
        except yaml.YAMLError as e:
            raise ConfigurationError(f"Invalid YAML in configuration file: {str(e)}") from e
        except Exception as e:
            raise ConfigurationError(f"Failed to load configuration: {str(e)}") from e
    
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """
        Validate configuration against schema.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            True if valid
            
        Raises:
            ConfigurationError if invalid
        """
        # TODO: Implement schema validation
        # For now, basic validation
        if not isinstance(config, dict):
            raise ConfigurationError("Configuration must be a dictionary")
        
        return True
    
    def create_config_from_template(self, template_name: str, output_path: str, 
                                   overrides: Optional[Dict[str, Any]] = None) -> str:
        """
        Create configuration from template.
        
        Args:
            template_name: Name of template
            output_path: Path to save new configuration
            overrides: Optional configuration overrides
            
        Returns:
            Path to created configuration file
        """
        # TODO: Implement template system
        # For now, create a basic template
        template = {
            'data_import': {
                'input_path': 'data/',
                'dataset': 'venus_Detroit',
            },
            'roi_processing': {
                'roi_bounds': None,
            },
            'road_extraction': {
                'method': 'osm',
            },
            'pci_segmentation': {
                'algorithm': 'dijkstra',
            },
            'data_preparation': {
                'normalize': True,
                'patch_size': [32, 32],
            },
            'model_training': {
                'model_type': 'unet',
                'input_size': [32, 32, 12],
                'n_classes': 4,
                'epochs': 100,
                'batch_size': 32,
            },
        }
        
        if overrides:
            template.update(overrides)
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            yaml.dump(template, f, default_flow_style=False)
        
        return str(output_path)
    
    def merge_with_defaults(self, config: Dict[str, Any], 
                           defaults: Dict[str, Any]) -> Dict[str, Any]:
        """
        Merge configuration with defaults.
        
        Args:
            config: User configuration
            defaults: Default configuration
            
        Returns:
            Merged configuration
        """
        merged = defaults.copy()
        
        def deep_merge(base: dict, override: dict):
            for key, value in override.items():
                if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                    deep_merge(base[key], value)
                else:
                    base[key] = value
        
        deep_merge(merged, config)
        return merged

