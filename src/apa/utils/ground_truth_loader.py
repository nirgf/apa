"""
Ground truth data loading utilities for APA.

Provides functions for loading and processing PCI ground truth data from CSV and Excel files.
"""

from typing import Tuple, Optional
import numpy as np
import pandas as pd
from pathlib import Path

# Try to import coordinate conversion module from apa.geo_reference
try:
    from apa.geo_reference import CovertITM2LatLon
    HAS_COVERT_ITM = True
except ImportError:
    try:
        # Fallback: try direct import (if installed as package)
        import CovertITM2LatLon
        HAS_COVERT_ITM = True
    except ImportError:
        try:
            # Fallback: try src.geo_reference path
            from src.geo_reference import CovertITM2LatLon
            HAS_COVERT_ITM = True
        except ImportError:
            HAS_COVERT_ITM = False


def get_GT_xy_PCI(xls_path: str, isLatLon: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Get ground truth PCI data with coordinates from CSV or Excel file.
    
    This function loads PCI data from CSV or Excel files and extracts:
    - PCI values
    - Coordinates (x/y or latitude/longitude)
    - Segment IDs (if available)
    
    Args:
        xls_path: Path to CSV or Excel file
        isLatLon: If True, assumes coordinates are already in lat/lon format.
                 If False, assumes coordinates are in ITM format and converts to WGS84.
    
    Returns:
        Tuple of (lon_vec, lat_vec, pci_vec, seg_id):
        - lon_vec: Longitude values (numpy array)
        - lat_vec: Latitude values (numpy array)
        - pci_vec: PCI values (numpy array)
        - seg_id: Segment IDs (numpy array, empty if not available)
    
    Raises:
        ValueError: If file format is unsupported or required columns are missing
        DataError: If file cannot be read
    """
    from apa.common.exceptions import DataError
    
    # Get file suffix
    xls_suffix = Path(xls_path).suffix.lower()
    
    # Read file based on extension
    if xls_suffix == '.csv':
        try:
            df = pd.read_csv(xls_path)
        except Exception as e:
            raise DataError(f"Failed to read CSV file {xls_path}: {str(e)}") from e
    elif xls_suffix in ['.xls', '.xlsx']:
        try:
            df = pd.read_excel(xls_path)
        except ImportError:
            raise DataError("pandas with openpyxl is required for Excel files. Install with: pip install openpyxl")
        except Exception as e:
            raise DataError(f"Failed to read Excel file {xls_path}: {str(e)}") from e
    else:
        raise ValueError(f"Unsupported file extension: '{xls_suffix}'. Supported extensions are: ['.csv', '.xls', '.xlsx']")
    
    # Make column names case-insensitive
    df.columns = df.columns.str.lower()
    
    # Extract PCI values
    if 'pci' in df.columns:
        pci_vec = df.pci.values
    elif 'cond' in df.columns:
        pci_vec = df.cond.values
    else:
        raise ValueError("The DataFrame must contain a column named 'pci' or 'cond'.")
    
    # Extract segment IDs (optional)
    if 'seg_id' in df.columns.tolist():
        seg_id = df.seg_id.values
    else:
        seg_id = np.array([])
    
    # Extract coordinates
    if 'x' in df.columns and 'y' in df.columns:
        x_vec = df.x.values
        y_vec = df.y.values
    elif 'latitude' in df.columns and 'longitude' in df.columns:
        x_vec = df.latitude.values
        y_vec = df.longitude.values
    else:
        raise ValueError("The DataFrame must contain columns named 'x'/'y' or 'latitude'/'longitude'.")
    
    # Get dates if available
    dates = df.s_date.values if 's_date' in df.columns else None
    
    # Calculate lat/lon vectors
    if not isLatLon:
        # Convert from ITM to WGS84
        if HAS_COVERT_ITM and CovertITM2LatLon:
            try:
                lat_vec, lon_vec = CovertITM2LatLon.ITM2WGS(x_vec, y_vec)
                lat_vec = np.reshape(lat_vec, [len(lat_vec), 1])
                lon_vec = np.reshape(lon_vec, [len(lon_vec), 1])
            except Exception as e:
                raise DataError(f"Failed to convert ITM coordinates to WGS84: {str(e)}") from e
        else:
            raise DataError("CovertITM2LatLon module is required for ITM to WGS84 conversion. Please ensure it's available.")
    else:
        # Coordinates are already in lat/lon format
        lat_vec = x_vec
        lon_vec = y_vec
    
    return lon_vec, lat_vec, pci_vec, seg_id


def get_PCI_ROI(roi: list, GT_xy_PCI: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get PCI points and indices within a ROI.
    
    Args:
        roi: ROI bounds as [min_lon, max_lon, min_lat, max_lat] or [xmin, xmax, ymin, ymax]
        GT_xy_PCI: Tuple from get_GT_xy_PCI containing (lon_vec, lat_vec, pci_vec, seg_id)
                   Can pass first 3 elements: GT_xy_PCI[:3]
    
    Returns:
        Tuple of (points_PCI, ROI_point_idx):
        - points_PCI: PCI values for points within ROI
        - ROI_point_idx: Indices of points within ROI
    """
    lon_vec, lat_vec, pci_vec = GT_xy_PCI[:3]
    
    # Extract ROI bounds
    if len(roi) == 4:
        min_lon, max_lon, min_lat, max_lat = roi
    else:
        raise ValueError(f"ROI must have 4 elements [min_lon, max_lon, min_lat, max_lat], got {len(roi)}")
    
    # Find points within ROI
    lon_mask = (lon_vec.flatten() >= min_lon) & (lon_vec.flatten() <= max_lon)
    lat_mask = (lat_vec.flatten() >= min_lat) & (lat_vec.flatten() <= max_lat)
    roi_mask = lon_mask & lat_mask
    
    ROI_point_idx = np.where(roi_mask)[0]
    points_PCI = pci_vec[ROI_point_idx] if len(ROI_point_idx) > 0 else np.array([])
    
    return points_PCI, ROI_point_idx

