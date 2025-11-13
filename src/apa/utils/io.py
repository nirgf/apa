"""
I/O utilities for APA.

Provides file I/O operations for various formats (YAML, JSON, HDF5, Pickle).
"""

from typing import Any, Dict, Optional
import yaml
import json
import h5py
import pickle
import os
from pathlib import Path


class IOUtils:
    """Utilities for file I/O operations."""
    
    @staticmethod
    def load_yaml(filepath: str) -> Dict[str, Any]:
        """
        Load YAML file.
        
        Args:
            filepath: Path to YAML file
            
        Returns:
            Dictionary from YAML file
        """
        with open(filepath, 'r') as f:
            return yaml.safe_load(f)
    
    @staticmethod
    def save_yaml(data: Dict[str, Any], filepath: str):
        """
        Save dictionary to YAML file.
        
        Args:
            data: Dictionary to save
            filepath: Path to save file
        """
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            yaml.dump(data, f, default_flow_style=False)
    
    @staticmethod
    def load_json(filepath: str) -> Dict[str, Any]:
        """
        Load JSON file.
        
        Args:
            filepath: Path to JSON file
            
        Returns:
            Dictionary from JSON file
        """
        with open(filepath, 'r') as f:
            return json.load(f)
    
    @staticmethod
    def save_json(data: Dict[str, Any], filepath: str):
        """
        Save dictionary to JSON file.
        
        Args:
            data: Dictionary to save
            filepath: Path to save file
        """
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    @staticmethod
    def load_h5(filepath: str, key: Optional[str] = None) -> Any:
        """
        Load data from HDF5 file.
        
        Args:
            filepath: Path to HDF5 file
            key: Optional key to load (if None, returns file handle)
            
        Returns:
            Data from HDF5 file
        """
        with h5py.File(filepath, 'r') as f:
            if key is None:
                return f
            return f[key][:]
    
    @staticmethod
    def save_h5(data: Any, filepath: str, key: str = 'data'):
        """
        Save data to HDF5 file.
        
        Args:
            data: Data to save
            filepath: Path to save file
            key: Key to save data under
        """
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(filepath, 'w') as f:
            f.create_dataset(key, data=data)
    
    @staticmethod
    def load_pickle(filepath: str) -> Any:
        """
        Load data from Pickle file.
        
        Args:
            filepath: Path to Pickle file
            
        Returns:
            Unpickled data
        """
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    
    @staticmethod
    def save_pickle(data: Any, filepath: str):
        """
        Save data to Pickle file.
        
        Args:
            data: Data to save
            filepath: Path to save file
        """
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)

