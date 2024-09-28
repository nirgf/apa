# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 20:52:45 2024

@author: ariep
"""
import CovertITM2LatLon
import numpy as np
import rasterio
from rasterio.transform import rowcol, xy
import matplotlib.pyplot as plt


def extents(f):
  delta = f[1] - f[0]
  return [f[0] - delta/2, f[-1] + delta/2]


file_path = 'venus data/VE_VM03_VSC_L2VALD_ISRAELWB_20230531/VE_VM03_VSC_PDTIMG_L2VALD_ISRAELWB_20230531_FRE.DBL.TIF'
with rasterio.open(file_path) as src:
    # Read the data
    data = src.read(1)  # Assuming single band raster
    
    # Extract metadata
    bounds = src.bounds
    transform = src.transform
    crs = src.crs


# Example: Converting a specific pixel coordinate to geographic coordinate
rows, cols = np.indices(data.shape)
xs, ys = xy(transform, rows, cols)

# Convert lists to numpy arrays for easier manipulation
xs = np.array(xs)
ys = np.array(ys)

lat_mat, lon_mat = CovertITM2LatLon.UTM2WGS(xs, ys)

lat_mat = np.array(lat_mat).reshape(list(np.shape(xs)) + [1])
lon_mat = np.array(lon_mat).reshape(list(np.shape(ys)) + [1])

latlon_mat = np.append(lat_mat, lon_mat, -1)


# plt.imshow(latlon_mat)

