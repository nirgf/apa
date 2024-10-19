#%% import pandas
import os.path
import json
from cProfile import label

import numpy as np
from PIL.ImageColor import colormap
from sympy.abc import alpha
from CONST import bands_dict
import CovertITM2LatLon
import pandas as pd
import ImportVenusModule
import matplotlib.pyplot as plt
from point_cloud_utils import get_lighttraffic_colormap, fill_mask_with_spline, merge_close_points, \
    scatter_plot_with_annotations, fit_spline_pc, fill_mask_with_irregular_spline,dilate_mask,fill_mask_with_line_point_values,create_masks,apply_masks_and_average
from scipy.interpolate import splprep, splev,griddata
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
# # Update map file
# CovertITM2LatLon.createFoliumMap(NA_points_ls, np.mean(NA_points_ls, 0), labels=pci_vec)
#
# #%% open map in browser
# CovertITM2LatLon.showMap('points_map.html')

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

# roi=((35.095,35.120),(32.802,32.818))
# xmin_cut = 35.006
# xmax_cut = 35.120
# ymin_cut = 32.7440226939727
# ymax_cut = 32.818

xmin_cut = 35.095
xmax_cut = 35.120
ymin_cut = 32.802
ymax_cut = 32.818

# Get the indices corresponding to the cut boundaries
kiryatAtaIdx = np.argwhere((lon_mat > ymin_cut) & (lon_mat < ymax_cut)\
                        & (lat_mat > xmin_cut) & (lat_mat < xmax_cut))

#%%
# Cut the image based on indices
# Get the indices corresponding to the cut boundaries
#%%
x_ind_min,x_ind_max  = np.min(kiryatAtaIdx[:,1]), np.max(kiryatAtaIdx[:,1])
y_ind_min, y_ind_max = np.min(kiryatAtaIdx[:,0]), np.max(kiryatAtaIdx[:,0])
# Cut the image based on indices
kiryatAtaImg = VenusImage[y_ind_min:y_ind_max,x_ind_min:x_ind_max,
                          [6, 3, 1]].astype(float)
kiryatAtaImg[kiryatAtaImg <= 0] = np.nan
norm_vec = np.nanmax(kiryatAtaImg, axis=(0,1)).astype(float)
for normBandIdx in range(len(norm_vec)):
    kiryatAtaImg[:, :, normBandIdx] = kiryatAtaImg[:, :, normBandIdx]/norm_vec[normBandIdx]

lon_mat_KiryatAta = lon_mat[y_ind_min:y_ind_max,x_ind_min:x_ind_max]

lat_mat_KiryatAta =  lat_mat[y_ind_min:y_ind_max,x_ind_min:x_ind_max]

# fig_roi, ax_roi = plt.subplots()
# im_ax=ax_roi.pcolormesh(lat_mat_KiryatAta,lon_mat_KiryatAta, kiryatAtaImg)


# Filter the scatter points to include only those within the ROI
scatter_indices = (lon_vec >= xmin_cut) & (lon_vec <= xmax_cut) & \
                  (lat_vec >= ymin_cut) & (lat_vec <= ymax_cut)

# Crop the X, Y, and Z arrays based on these indices
X_cropped = lat_mat_KiryatAta
Y_cropped = lon_mat_KiryatAta
# Apply the mask to the image
Z_cropped = kiryatAtaImg

filtered_x = lon_vec[scatter_indices]
filtered_y = lat_vec[scatter_indices]
filtered_PCI = pci_vec.values
filtered_PCI = filtered_PCI[scatter_indices.ravel()]
points_PCI = np.c_[filtered_x,filtered_y,filtered_PCI]

# scatter_plot_with_annotations(points_PCI,ax_roi)
binary_mask = np.zeros(Z_cropped.shape[:-1])

# this function merge different lane into one PCI (assumption, may not always be valid)
# TODO: optimize threshold
points_merge_PCI = merge_close_points(points_PCI[:,:2], points_PCI[:,2], 50e-5) # TODO:
xy_points_merge = points_merge_PCI[:, :2]

# create a spline fit trajectory from point cloud using GreedyNN and this create a mask of of it (can be extended with existing mask)
extended_mask, line_string = fill_mask_with_irregular_spline(xy_points_merge,X_cropped,Y_cropped,binary_mask,
                                                   combine_mask=False)  # this return mask in the pixels the spline line passes through
x_new, y_new, _ = fit_spline_pc(xy_points_merge)

# Plot the masked data using pcolormesh
fig_roi, ax_roi = plt.subplots()
im_ax=ax_roi.pcolormesh(X_cropped, Y_cropped, Z_cropped)
scatter_plot_with_annotations(points_merge_PCI,ax_roi,markersize=200,linewidths=2,alpha=1)

ax_roi.plot(x_new, y_new, 'b--', label='Spline Fit')

# segment_mask = fill_mask_with_line_point_values(line_string, points_merge_PCI, extended_mask.shape, radius=3.5)
# create a segemented image of PCI values based on extendedn mask
grid_value= griddata(xy_points_merge, points_merge_PCI[:, 2], (X_cropped, Y_cropped), method='nearest')
segment_mask = grid_value*dilate_mask(extended_mask,3)
segment_mask[segment_mask<=0]=np.nan

# this part handle all bands
hys_img = VenusImage[y_ind_min:y_ind_max,x_ind_min:x_ind_max,:].astype(float)
hys_img_norm=np.zeros_like(hys_img)
for kk in range(hys_img.shape[-1]):
    hys_img_1chn = hys_img[:, :, kk]
    hys_img_1chn = hys_img_1chn / np.nanmax(hys_img_1chn)
    hys_img_1chn[hys_img_1chn <= 0] = np.nan
    hys_img_norm[:,:,kk]=hys_img_1chn

# Extract all the 'wavelength' values into a list
wavelengths = [info['wavelength'] for info in bands_dict.values()]
wavelengths_array = np.array(wavelengths)

# this part create mask based on segmented PCI image
mask_below_30, mask_30_to_70, mask_above_85 = create_masks(segment_mask)
mask_all_channel_values_30 = apply_masks_and_average(hys_img_norm,mask_below_30)
mask_all_channel_values_85 = apply_masks_and_average(hys_img_norm,mask_above_85)

# plot spectoroms
plt.figure(111)
plt.plot(wavelengths_array,np.nanmean(np.asarray(mask_all_channel_values_30),axis=1),'r',label=f'below30PCI,N_AVG={np.count_nonzero(mask_all_channel_values_30)}')
plt.plot(wavelengths_array,np.nanmean(np.asarray(mask_all_channel_values_85),axis=1),'g',label=f'above85PCI,N_AVG={np.count_nonzero(mask_all_channel_values_85)}')
plt.title('Avg spectrogram')
plt.xlabel('wavelength[nm]')
plt.legend()


for kk in range(hys_img.shape[-1]):
    fig_roi, ax_roi = plt.subplots()
    hys_img_1chn = hys_img[:, :, kk]
    hys_img_1chn = hys_img_1chn / np.nanmax(hys_img_1chn)
    hys_img_1chn[hys_img_1chn <= 0] = np.nan
    ax_roi.pcolormesh(X_cropped, Y_cropped, hys_img_1chn)
    ax_roi.set_title(f'Central Wavelength:{bands_dict[kk]['wavelength']}')
    scatter_plot_with_annotations(points_merge_PCI, ax_roi, markersize=100, linewidths=1, alpha=0.3)
    # plt.colorbar()
    plt.show()

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
# coinciding_mask = GetRoadsCoordinates.get_coinciding_mask(roads_gdf, lon_mat_KiryatAta, lat_mat_KiryatAta)
# expanded_mask = np.repeat(coinciding_mask[:, :, np.newaxis], 3, axis=2)  # Shape becomes (2081, 1602, 3)
# plt.figure()
# plt.imshow(coinciding_mask)
#
#
# rowAndColIdx = np.argwhere(coinciding_mask)
# filteredKiryaAtaImg = np.zeros(np.shape(kiryatAtaImg))
# filteredKiryaAtaImg[rowAndColIdx[:, 0], rowAndColIdx[:, 1], :] \
#     = (kiryatAtaImg[rowAndColIdx[:, 0], rowAndColIdx[:, 1], :])
#
# plt.pcolormesh(lat_mat_KiryatAta, \
#                lon_mat_KiryatAta,  \
#                    filteredKiryaAtaImg[:, :, 0])
#
