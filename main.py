#%% import pandas
import os.path

import numpy as np
from PIL.ImageColor import colormap

import CovertITM2LatLon
import pandas as pd
import ImportVenusModule
import matplotlib.pyplot as plt
from point_cloud_utils import get_lighttraffic_colormap,fill_mask_with_spline
## Make plots interactive
import matplotlib
# matplotlib.use('Qt5Agg')
matplotlib.use('TkAgg')

cmap_me = get_lighttraffic_colormap()
plt.ion()

#%% Update Git Rules so the push will not get stuck
# import UpdateGitIgnore
# UpdateGitIgnore.main()

#%% get NA data
df = pd.read_excel('seker_nezakim.xls')
pci_vec = df.PCI
x_vec = df.X
y_vec = df.Y
dates = df.S_Date

# calculate lat/lon vecs
lat_vec, lon_vec = CovertITM2LatLon.ITM2WGS(x_vec, y_vec)
lat_vec = np.reshape(lat_vec, [len(lat_vec), 1])
lon_vec = np.reshape(lon_vec, [len(lon_vec), 1])
NA_points_ls = np.append(lat_vec, lon_vec, 1)
# Update map file
CovertITM2LatLon.createFoliumMap(NA_points_ls, np.mean(NA_points_ls, 0), labels=pci_vec)

#%% open map in browser
CovertITM2LatLon.showMap('points_map.html')

#%% Get venus data
parent_path='/Users/nircko/DATA/apa'
data_dirname = os.path.join(parent_path,'venus data/VE_VM03_VSC_L2VALD_ISRAELWB_20230531/')
data_filename = 'VE_VM03_VSC_PDTIMG_L2VALD_ISRAELWB_20230531_FRE.DBL.TIF'
metadata_filename = 'M02_metadata.csv'
metadata_dirname = os.path.join(parent_path,'venus data/')

VenusImage = ImportVenusModule.getVenusData(data_dirname, data_filename)
venusMetadata = ImportVenusModule.getVenusMetaData(metadata_dirname, metadata_filename)

maxLat, minLon = CovertITM2LatLon.UTM2WGS(venusMetadata.MinX[0], venusMetadata.MinY[0])
minLat, maxLon = CovertITM2LatLon.UTM2WGS(venusMetadata.MaxX[0], venusMetadata.MaxY[0])


#%% Get lat/lon directly from VENUS data - Get kiryatAta only
import getLatLon_fromTiff
x = getLatLon_fromTiff.convert_raster_to_geocoords(data_dirname + data_filename)

# unpack lat/lon
lon_mat = x[:, :, 0] 
lat_mat = x[:, :, 1]

xmin_cut = 35.06
xmax_cut = 35.126738306451614

ymin_cut = 32.7440226939727
ymax_cut = 32.818

# Get the indices corresponding to the cut boundaries
kiryatAtaIdx = np.argwhere((lon_mat > ymin_cut) & (lon_mat < ymax_cut)\
                        & (lat_mat > xmin_cut) & (lat_mat < xmax_cut))
#%%
# Cut the image based on indices
kiryatAtaImg = VenusImage[np.min(kiryatAtaIdx[:, 0]):np.max(kiryatAtaIdx[:, 0]),\
                          np.min(kiryatAtaIdx[:, 1]):np.max(kiryatAtaIdx[:, 1]),
                          [6, 3, 1]].astype(float)
kiryatAtaImg[kiryatAtaImg < 0] = 0
norm_vec = np.max(np.max(kiryatAtaImg, 0), 0).astype(float)
for normBandIdx in range(len(norm_vec)):
    kiryatAtaImg[:, :, normBandIdx] = kiryatAtaImg[:, :, normBandIdx]/norm_vec[normBandIdx]

lon_mat_KiryatAta = lon_mat[np.min(kiryatAtaIdx[:, 0]):np.max(kiryatAtaIdx[:, 0]),\
                              np.min(kiryatAtaIdx[:, 1]):np.max(kiryatAtaIdx[:, 1])]

lat_mat_KiryatAta =  lat_mat[np.min(kiryatAtaIdx[:, 0]):np.max(kiryatAtaIdx[:, 0]),\
                              np.min(kiryatAtaIdx[:, 1]):np.max(kiryatAtaIdx[:, 1])]


#%% Plot Full Image With NA Data
plt.figure()
plt.pcolormesh(lat_mat_KiryatAta, lon_mat_KiryatAta,  kiryatAtaImg)
# Plot NA Data
print('Plotting The Full Image')

plt.xlim([xmin_cut, xmax_cut])
plt.ylim([ymin_cut, ymax_cut])
plt.scatter(lon_vec, lat_vec,c=pci_vec.values,cmap=cmap_me)

#roi
roi=((35.095,35.120),(32.802,32.818)) # (xmin,xmax),(ymin,ymax)
# Find indices corresponding to the current axis limits
x_indices = (lat_mat_KiryatAta >= roi[0][0]) & (lat_mat_KiryatAta <= roi[0][1])
y_indices = (lon_mat_KiryatAta >= roi[1][0]) & (lon_mat_KiryatAta <= roi[1][1])

mask = x_indices & y_indices
indices = np.where(mask)
# Get the bounding box (the min and max for each dimension)
x_min, x_max = np.min(indices[1]), np.max(indices[1])
y_min, y_max = np.min(indices[0]), np.max(indices[0])

# Crop the X, Y, and Z arrays based on these indices
X_cropped = lat_mat_KiryatAta[y_min:y_max+1, x_min:x_max+1]
Y_cropped = lon_mat_KiryatAta[y_min:y_max+1, x_min:x_max+1]
# Apply the mask to the image
Z_cropped = kiryatAtaImg[y_min:y_max+1, x_min:x_max+1,:]

scatter_indices = (lon_vec >= roi[0][0]) & (lon_vec <= roi[0][1]) & \
                  (lat_vec >= roi[1][0]) & (lat_vec <= roi[1][1])

filtered_x = lon_vec[scatter_indices]
filtered_y = lat_vec[scatter_indices]
filtered_PCI = pci_vec.values
filtered_PCI = filtered_PCI[scatter_indices]
import json
# Convert the NumPy arrays to lists
data = {
    'scatter_indices': scatter_indices.tolist(),
    'x_min': x_min.tolist(),
    'x_max': x_max.tolist(),
    'y_min': y_min.tolist(),
    'y_max': y_max.tolist()
}

# Save the data to a JSON file
with open('arrays.json', 'w') as f:
    json.dump(data, f)

fig_roi, ax_roi = plt.subplots()

# Plot the masked data using pcolormesh
ax_roi.pcolormesh(X_cropped, Y_cropped, Z_cropped)



plt.show()

#%% Get only pixels that intersect with roads
from GetRoadsModule import GetRoadsCoordinates
lat_range = (ymin_cut, ymax_cut)
lon_range = (xmin_cut, xmax_cut)

roads_gdf = GetRoadsCoordinates.get_road_mask(lat_range, lon_range)

## Plot the road maks and VENUS data on the same fig
# GetRoadsCoordinates.plot_road_mask(roads_gdf)
# plt.pcolormesh(lat_mat_KiryatAta, lon_mat_KiryatAta,  kiryatAtaImg)

# # boolean mask for VENUS Data
coinciding_mask = GetRoadsCoordinates.get_coinciding_mask(roads_gdf, lon_mat_KiryatAta, lat_mat_KiryatAta)
expanded_mask = np.repeat(coinciding_mask[:, :, np.newaxis], 3, axis=2)  # Shape becomes (2081, 1602, 3)

# Filter The Venus Data to include the results data
rowAndColIdx = np.argwhere(coinciding_mask)
filteredKiryaAtaImg = np.zeros(np.shape(kiryatAtaImg))
filteredKiryaAtaImg[rowAndColIdx[:, 0], rowAndColIdx[:, 1], :] \
    = (kiryatAtaImg[rowAndColIdx[:, 0], rowAndColIdx[:, 1], :])

plt.figure()
plt.pcolormesh(lat_mat_KiryatAta, \
               lon_mat_KiryatAta,  \
                   filteredKiryaAtaImg[:, :, 0])

