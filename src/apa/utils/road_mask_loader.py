"""
Road mask loading utilities for APA.

Provides functions for loading and creating road masks from OpenStreetMap data.
"""

from typing import Dict, Any, Optional, Tuple
import numpy as np
import json
import os
from pathlib import Path
from scipy.sparse import csr_matrix, load_npz

# Try to import pc_utils (may be external module)
try:
    import pc_utils
    HAS_PC_UTILS = True
except ImportError:
    HAS_PC_UTILS = False
    # Create minimal fallback functions
    def is_roi_within_bounds(roi1, roi2):
        """Check if roi1 is within bounds of roi2."""
        xmin1, xmax1, ymin1, ymax1 = roi1
        xmin2, xmax2, ymin2, ymax2 = roi2
        return (xmin2 <= xmin1 <= xmax2 and xmin2 <= xmax1 <= xmax2 and
                ymin2 <= ymin1 <= ymax2 and ymin2 <= ymax1 <= ymax2)
    
    def get_pixels_intersect_with_roads(lon_mat, lat_mat, lon_range, lat_range):
        """Get pixels that intersect with roads from OpenStreetMap."""
        # This is a placeholder - actual implementation should use OSM data
        # For now, return a dummy mask
        shape = lon_mat.shape if hasattr(lon_mat, 'shape') else (100, 100)
        return np.zeros(shape, dtype=bool)


def save_npz(filepath: str, sparse_matrix: csr_matrix) -> None:
    """
    Save a sparse matrix to NPZ file.
    
    Args:
        filepath: Path to save the NPZ file
        sparse_matrix: Sparse matrix to save
    """
    # Use scipy's save_npz if available, otherwise manual save
    try:
        from scipy.sparse import save_npz as scipy_save_npz
        scipy_save_npz(filepath, sparse_matrix)
    except ImportError:
        # Fallback: save as dense array (less efficient but works)
        np.savez_compressed(filepath, data=sparse_matrix.toarray())


def get_mask_from_roads_gdf(npz_filename: str, crop_rect: Tuple[int, int, int, int], 
                            data: Optional[Dict[str, Any]] = None) -> np.ndarray:
    """
    Get road mask from NPZ file or create it if it doesn't exist.
    
    Args:
        npz_filename: Path to NPZ file containing road mask
        crop_rect: Crop rectangle as (x_ind_min, y_ind_min, x_ind_max, y_ind_max)
        data: Dictionary containing 'roi', 'X_cropped', 'Y_cropped' (required if file doesn't exist)
    
    Returns:
        Road mask array cropped to crop_rect
    """
    npz_file_path = Path(npz_filename)
    metadata_filename = npz_file_path.with_suffix(".json")
    
    if not (os.path.exists(npz_filename) and os.path.exists(metadata_filename)):
        if data is None:
            raise ValueError("No data provided to save when the file does not exist.")
        else:
            create_mask_from_roads_gdf(npz_filename, data)
    else:
        # Load metadata
        with open(metadata_filename, 'r') as f:
            metadata_dict = json.load(f)
        
        # Check if ROI is within bounds
        if data is not None and 'roi' in data:
            if HAS_PC_UTILS:
                is_roi_within_bounds = pc_utils.is_roi_within_bounds(data["roi"], metadata_dict["roi"])
            else:
                is_roi_within_bounds = _is_roi_within_bounds(data["roi"], metadata_dict["roi"])
            
            if not is_roi_within_bounds:
                print("Existing ROI of GDF roads is not bounded by the requested data")  # Patched by Arie 30.01.2025
                # Note: Original code had this as a warning, not an error
        
        print(f"File '{npz_filename}' found. Loading data...")
        
        # Load NPZ file
        coinciding_mask = load_npz(npz_filename).toarray()
        
        # Crop to requested region
        x_ind_min, y_ind_min, x_ind_max, y_ind_max = crop_rect
        return coinciding_mask[y_ind_min:y_ind_max, x_ind_min:x_ind_max]
    
    # If we just created the file, load it
    coinciding_mask = load_npz(npz_filename).toarray()
    x_ind_min, y_ind_min, x_ind_max, y_ind_max = crop_rect
    return coinciding_mask[y_ind_min:y_ind_max, x_ind_min:x_ind_max]


def create_mask_from_roads_gdf(npz_filename: str, data: Dict[str, Any]) -> None:
    """
    Create road mask from OpenStreetMap data and save to NPZ file.
    
    Args:
        npz_filename: Path to save the NPZ file
        data: Dictionary containing:
            - 'roi': [xmin_cut, xmax_cut, ymin_cut, ymax_cut]
            - 'X_cropped': Latitude matrix
            - 'Y_cropped': Longitude matrix
    """
    roi = data["roi"]
    lon_mat = data["Y_cropped"]  # Note: Y is longitude
    lat_mat = data["X_cropped"]  # Note: X is latitude
    
    xmin_cut, xmax_cut, ymin_cut, ymax_cut = roi
    lat_range = (ymin_cut, ymax_cut)
    lon_range = (xmin_cut, xmax_cut)
    
    # Get pixels that intersect with roads
    if HAS_PC_UTILS:
        coinciding_mask = pc_utils.get_pixels_intersect_with_roads(
            lon_mat, lat_mat, lon_range, lat_range
        )
    else:
        # Fallback: create dummy mask (should be replaced with actual OSM implementation)
        shape = lon_mat.shape if hasattr(lon_mat, 'shape') else (100, 100)
        coinciding_mask = np.zeros(shape, dtype=bool)
        print("Warning: pc_utils not available, using dummy road mask")
    
    # Convert to sparse matrix and save
    rowAndColIdx = np.argwhere(coinciding_mask)
    save_npz(npz_filename, csr_matrix(coinciding_mask))
    
    # Save metadata file
    metadata = {
        "description": "Sparse matrix of roads in OpenStreetMap",
        "author": "apa",
        "version": 1.0,
        "roi": roi
    }
    npz_file_path = Path(npz_filename)
    metadata_filename = npz_file_path.with_suffix(".json")
    
    with open(metadata_filename, "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Saved compressed binary mask of OpenStreetMap roads into '{npz_filename}'.")


def _is_roi_within_bounds(roi1: list, roi2: list) -> bool:
    """
    Check if roi1 is within bounds of roi2.
    
    Args:
        roi1: ROI as [xmin, xmax, ymin, ymax]
        roi2: ROI as [xmin, xmax, ymin, ymax]
    
    Returns:
        True if roi1 is within roi2 bounds
    """
    xmin1, xmax1, ymin1, ymax1 = roi1
    xmin2, xmax2, ymin2, ymax2 = roi2
    return (xmin2 <= xmin1 <= xmax2 and xmin2 <= xmax1 <= xmax2 and
            ymin2 <= ymin1 <= ymax2 and ymin2 <= ymax1 <= ymax2)

