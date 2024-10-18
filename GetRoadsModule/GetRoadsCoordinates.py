#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 17:05:01 2024

@author: Arie Pyasik, APA.inc
"""

import numpy as np
import geopandas as gpd
from shapely.geometry import box
from shapely.strtree import STRtree
from shapely.geometry import Polygon
import osmnx as ox
import matplotlib.pyplot as plt
import geopandas as gpd
from tqdm import tqdm

def get_road_mask(lat_range, lon_range):
    """
    Fetches the road data within a given lat/lon range and returns a road mask.

    Parameters:
    - lat_range: tuple of (min_lat, max_lat)
    - lon_range: tuple of (min_lon, max_lon)
    
    Returns:
    - roads_gdf: GeoDataFrame of roads within the specified area.
    """
    # Define the bounding box from the lat/lon ranges
    bounding_box = Polygon([
        (lon_range[0], lat_range[0]), 
        (lon_range[0], lat_range[1]),
        (lon_range[1], lat_range[1]), 
        (lon_range[1], lat_range[0])
    ])

    # Fetch road data from OpenStreetMap within the bounding box
    roads_gdf = ox.geometries.geometries_from_polygon(bounding_box, tags={'highway': True})

    return roads_gdf

def plot_road_mask(roads_gdf):
    """
    Plots the road geometries as a mask.

    Parameters:
    - roads_gdf: GeoDataFrame of road geometries
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    roads_gdf.plot(ax=ax, color='black')  # Plot roads in black (for mask-like visualization)
    plt.title("Road Mask")
    plt.show()

# Example usage
if __name__ == "__main__":
    # Define lat/lon range (example)
    lat_range = (32.804, 32.817)
    lon_range = (35.108, 35.127)

    # Get road data in the lat/lon area
    roads_gdf = get_road_mask(lat_range, lon_range)

    # Plot the road mask
    plot_road_mask(roads_gdf)


def get_coinciding_mask(roads_gdf, lat_matrix, lon_matrix):
    """
    Find the coinciding pixels (lat/lon grid cells) where roads pass through.

    Parameters:
    - roads_gdf: GeoDataFrame of road geometries.
    - lat_matrix: 2D NumPy array of latitudes (latitudes grid).
    - lon_matrix: 2D NumPy array of longitudes (longitudes grid).

    Returns:
    - boolean_mask: 2D Boolean NumPy array with the same shape as lat/lon matrices.
                    True where a road geometry intersects the lat/lon grid pixel.
    """
    # Get the shape of the lat/lon matrices
    shape = lat_matrix.shape

    # Initialize a boolean matrix of the same shape, filled with False
    boolean_mask = np.full(shape, False, dtype=bool)

    # Get grid step size (distance between adjacent lat/lon values)
    lat_step = abs(lat_matrix[1, 0] - lat_matrix[0, 0]) if shape[0] > 1 else 0
    lon_step = abs(lon_matrix[0, 1] - lon_matrix[0, 0]) if shape[1] > 1 else 0

    # Generate the pixel geometries (bounding boxes)
    pixel_polygons = []
    for i in tqdm(range(shape[0] - 1), desc='Generating Pixel Coordinates (Step 1 of 2)'):
        for j in range(shape[1] - 1):
            pixel_polygon = box(lon_matrix[i, j], lat_matrix[i, j], 
                                lon_matrix[i, j] + lon_step, lat_matrix[i, j] + lat_step)
            pixel_polygons.append((i, j, pixel_polygon))

    # Create spatial index using STRtree for road geometries
    road_tree = STRtree(roads_gdf.geometry)  # Use the public 'geometry' accessor directly from the GeoDataFrame

    # Check intersections for each pixel using the spatial index
    for i, j, pixel_polygon in tqdm(pixel_polygons, desc='Checking Intersections (Step 2 of 2)'):
        # Query the tree to find possible intersecting roads
        possible_roads = road_tree.query(pixel_polygon)  # Returns the actual geometries directly

        # Check if any of the queried roads actually intersect the pixel
        if any(road.intersects(pixel_polygon) for road in possible_roads):
            boolean_mask[i, j] = True

    return boolean_mask

if __name__ == "__main__":
    # Define lat/lon range (example)
    lat_range = (32.804, 32.817)
    lon_range = (35.108, 35.127)

    # Get road data in the lat/lon area
    roads_gdf = get_road_mask(lat_range, lon_range)
    
    # bool_mask = get_coinciding_mask(roads_gdf, lat_range, lon_range)

    # Plot the road mask
    plot_road_mask(roads_gdf)
