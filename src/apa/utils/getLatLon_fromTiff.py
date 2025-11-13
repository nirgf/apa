# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 20:52:45 2024

@author: ariep
"""
import numpy as np
import rasterio
from rasterio.transform import rowcol, xy
import matplotlib.pyplot as plt

# Import CovertITM2LatLon from apa.geo_reference
# Try multiple import paths for compatibility
try:
    from apa.geo_reference import CovertITM2LatLon
except ImportError:
    try:
        from src.geo_reference import CovertITM2LatLon
    except ImportError:
        try:
            import src.geo_reference.CovertITM2LatLon as CovertITM2LatLon
        except ImportError:
            # Last resort: try relative import
            from ..geo_reference import CovertITM2LatLon


def extents(f):
  delta = f[1] - f[0]
  return [f[0] - delta/2, f[-1] + delta/2]


def convert_raster_to_geocoords(file_path, zone_number = 36, zone_letter='U'):
    """
    Reads a raster file, extracts pixel data, converts pixel coordinates to geographic coordinates 
    (latitude and longitude), and returns the lat/lon matrix.

    Parameters:
    - file_path: str, the path to the raster file (TIF format).

    Returns:
    - latlon_mat: numpy array, a matrix containing latitude and longitude coordinates for each pixel.
    - metadata: dict, containing bounds, transform, and CRS information.
    """
    with rasterio.open(file_path) as src:
        # Read the data (assuming single band raster)
        data = src.read(1)
        
        # Extract metadata
        bounds = src.bounds
        transform = src.transform
        crs = src.crs
        
    # Generate row and column indices for the given data
    rows, cols = np.indices(data.shape)
    
    # Convert pixel coordinates to geospatial coordinates (easting and northing)
    xs, ys = rasterio.transform.xy(transform, rows.ravel(), cols.ravel())
    
    # Convert lists to numpy arrays for easier manipulation
    xs = np.array(xs.reshape(rows.shape))
    ys = np.array(ys.reshape(cols.shape))
    
    # Convert UTM coordinates to latitude and longitude
    lat_mat, lon_mat = CovertITM2LatLon.UTM2WGS(xs, ys, zone_number, zone_letter)
    
    # Reshape latitude and longitude matrices to match the shape of the input data
    lat_mat = np.array(lat_mat).reshape(list(np.shape(xs)) + [1])
    lon_mat = np.array(lon_mat).reshape(list(np.shape(ys)) + [1])
    
    # Combine latitude and longitude into a single matrix
    latlon_mat = np.append(lat_mat, lon_mat, -1)
    
    
    return latlon_mat #, metadata
