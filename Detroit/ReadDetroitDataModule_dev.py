#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  8 17:46:55 2024

@author: root
"""

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd

# Read the GeoJSON file
gdf = gpd.read_file("Detroit/Roads.geojson")

coord_dict = {}
for index, row in gdf.iterrows():
    geometry = row['geometry']
    # Extract coordinates from the geometry object
    coords = geometry.coords
    coord_dict[index] = coords
    #print(f"Segment {index+1} coordinates: {coords}")


# Read the CSV file
df = pd.read_csv("Detroit/Pavement_Condition.csv")

gdfTest = gdf.reset_index(drop=True)
merged_df = gdf.merge(gdfTest, on='seg_id', how='inner')

# Extract the 'cond' column
cond_values = df['cond']

# Plot the GeoDataFrame
gdf.plot(figsize=(10, 8))

# Customize the plot (optional)
plt.title("GeoJSON Data Plot")
plt.xlabel("Longitude")
plt.ylabel("Latitude")

# Display the plot
plt.show()