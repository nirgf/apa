#%% import pandas
import os.path
import json
import numpy as np
from PIL.ImageColor import colormap
from sympy.abc import alpha
from CONST import bands_dict
import CovertITM2LatLon
import pandas as pd
import ImportVenusModule
import matplotlib.pyplot as plt
from point_cloud_utils import get_lighttraffic_colormap,fill_mask_with_spline,merge_close_points,scatter_plot_with_annotations,fit_spline_pc
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

# Loading the data back
with open('arrays.json', 'r') as f:
    loaded_data = json.load(f)

# Convert the lists back to NumPy arrays
x_max = np.array(loaded_data['x_max'])
x_min = np.array(loaded_data['x_min'])
y_max = np.array(loaded_data['y_max'])
y_min = np.array(loaded_data['y_min'])
scatter_indices = np.array(loaded_data['scatter_indices'])


# Crop the X, Y, and Z arrays based on these indices
X_cropped = lat_mat_KiryatAta[y_min:y_max+1, x_min:x_max+1]
Y_cropped = lon_mat_KiryatAta[y_min:y_max+1, x_min:x_max+1]
# Apply the mask to the image
Z_cropped = kiryatAtaImg[y_min:y_max+1, x_min:x_max+1,:]

filtered_x = lon_vec[scatter_indices]
filtered_y = lat_vec[scatter_indices]
filtered_PCI = pci_vec.values
filtered_PCI = filtered_PCI[scatter_indices.ravel()]
points_PCI = np.c_[filtered_x,filtered_y,filtered_PCI]

# scatter_plot_with_annotations(points_PCI,ax_roi)
binary_mask = np.zeros_like(Z_cropped)
points_merge_PCI = merge_close_points(points_PCI[:,:2], points_PCI[:,2], 50e-5)
xy_points_merge = points_merge_PCI[:, :2]

extended_mask, line_string = fill_mask_with_spline(binary_mask, xy_points_merge,
                                                   combine_mask=False)  # this return mask in the pixels the spline line passes through
x_new, y_new, _ = fit_spline_pc(xy_points_merge)

# Plot the masked data using pcolormesh
fig_roi, ax_roi = plt.subplots()
im_ax=ax_roi.pcolormesh(X_cropped, Y_cropped, Z_cropped)
scatter_plot_with_annotations(points_merge_PCI,ax_roi,markersize=200,linewidths=2,alpha=1)

ax_roi.plot(x_new, y_new, 'b--', label='Spline Fit')


hys_img = VenusImage[np.min(kiryatAtaIdx[:, 0]):np.max(kiryatAtaIdx[:, 0]),\
                          np.min(kiryatAtaIdx[:, 1]):np.max(kiryatAtaIdx[:, 1]),
                          :].astype(float)

hys_img=hys_img[y_min:y_max+1, x_min:x_max+1,:]
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
