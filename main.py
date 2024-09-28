#%% import pandas
import numpy as np
import CovertITM2LatLon
import pandas as pd
import ImportVenusModule
import matplotlib.pyplot as plt
# Some changes
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

# code is fine, data is wrong ...
maxLat, minLon = CovertITM2LatLon.UTM2WGS(venusMetadata.MinX[0], venusMetadata.MinY[0])
minLat, maxLon = CovertITM2LatLon.UTM2WGS(venusMetadata.MaxX[0], venusMetadata.MaxY[0])


minLat, minLon = [32.669, 35.015]
maxLat, maxLon = [32.976, 35.391]

lonVec = np.linspace(minLon, maxLon, np.shape(VenusImage)[0])
latVec = np.linspace(minLat, maxLat, np.shape(VenusImage)[0])

#%% Get lat/lon directly from arnons data
import getLatLon_fromTiff
x = latlon_mat
#%% show final result
image = VenusImage[:, :, [9]]

plt.figure()
# plt.imshow(image, extent=(minLon, maxLon, minLat, maxLat), aspect='auto')

plt.imshow(image, aspect='auto', interpolation='none',
           extent=extents(lon_mat.flatten()) + extents(lat_mat.flatten()), origin='lower')

plt.colorbar()
plt.set_cmap('Greys')
plt.clim([100, 400])

#%% Cut the image
xmin_cut = 35.10857030645161
xmax_cut = 35.126738306451614

ymin_cut = 32.8040226939727
ymax_cut = 32.817552693972694

plt.xlim([xmin_cut, xmax_cut])
plt.ylim([ymin_cut, ymax_cut])

plt.scatter(lon_vec, lat_vec)

plt.show()

#%% Add correction to lat/lon of the points
move_scatterX = 35.11916-35.11733
move_scatterY = 32.80504-32.80719
plt.scatter(lon_vec + move_scatterX, lat_vec + move_scatterY)

