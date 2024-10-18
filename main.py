#%% import pandas
import numpy as np
import CovertITM2LatLon
import pandas as pd
import ImportVenusModule
import matplotlib.pyplot as plt

## Make plots interactive
import matplotlib
matplotlib.use('Qt5Agg')
plt.ion()

#%% Update Git Rules so the push will not get stuck
import UpdateGitIgnore
UpdateGitIgnore.main()

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
# CovertITM2LatLon.showMap('points_map.html') 

#%% Get venus data
data_dirname = 'venus data/VE_VM03_VSC_L2VALD_ISRAELWB_20230531/'
data_filename = 'VE_VM03_VSC_PDTIMG_L2VALD_ISRAELWB_20230531_FRE.DBL.TIF'
metadata_filename = 'M02_metadata.csv'
metadata_dirname = 'venus data/'

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

plt.scatter(lon_vec, lat_vec)

plt.show()

#%% Get only pixels that intersect with roads
from GetRoadsModule import GetRoadsCoordinates
lat_range = (ymin_cut, ymax_cut)
lon_range = (xmin_cut, xmax_cut)

roads_gdf = GetRoadsCoordinates.get_road_mask(lat_range, lon_range)

## Plot the road maks and VENUS data on the same fig
# GetRoadsCoordinates.plot_road_mask(roads_gdf)
# plt.pcolormesh(lat_mat_KiryatAta, lon_mat_KiryatAta,  kiryatAtaImg)

# boolean mask for VENUS Data
coinciding_mask = GetRoadsCoordinates.get_coinciding_mask(roads_gdf, lon_mat_KiryatAta, lat_mat_KiryatAta)
expanded_mask = np.repeat(coinciding_mask[:, :, np.newaxis], 3, axis=2)  # Shape becomes (2081, 1602, 3)
plt.figure()
plt.imshow(coinciding_mask)


rowAndColIdx = np.argwhere(coinciding_mask)
filteredKiryaAtaImg = np.zeros(np.shape(kiryatAtaImg))
filteredKiryaAtaImg[rowAndColIdx[:, 0], rowAndColIdx[:, 1], :] \
    = (kiryatAtaImg[rowAndColIdx[:, 0], rowAndColIdx[:, 1], :])

plt.pcolormesh(lat_mat_KiryatAta, \
               lon_mat_KiryatAta,  \
                   filteredKiryaAtaImg[:, :, 0])

