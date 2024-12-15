import os.path
import json
import numpy as np
from pathlib import Path
from PIL.ImageColor import colormap
from src.CONST import bands_dict
import pandas as pd
from src.utils import ImportVenusModule,getLatLon_fromTiff
from src.geo_reference import CovertITM2LatLon
import src.utils.point_cloud_utils as pc_utils
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix, save_npz, load_npz
## Make plots interactive
from matplotlib.path import Path as pltPath
from scipy.spatial import cKDTree
from skimage.draw import line
import matplotlib.pyplot as plt
import matplotlib

# matplotlib.use('Qt5Agg')
matplotlib.use('TkAgg')

def get_GT_xy_PCI(xls_path, isLatLon = False):
    #%% get NA data
    xls_suffix=Path(xls_path).suffix
    if xls_suffix == '.csv':
        df = pd.read_csv(xls_path)
    elif xls_suffix == '.xls':
        df = pd.read_excel(xls_path)
    else:
        raise ValueError(f"Unsupported file extension: '{xls_suffix}'. Supported extensions are: {['.csv','.xls']}")

    df.columns = df.columns.str.lower() # make the columns name case-insensitive

    pci_vec = df.pci

    # verify if latitude should be x and not longitude
    if 'x' in df.columns:
        x_vec = df.x
        y_vec = df.y
    elif 'latitude' in df.columns:
        x_vec = df.latitude
        y_vec = df.longitude
    else:
        raise ValueError("The DataFrame must contain a column named 'x' or 'latitude'.")


    dates = df.s_date # add date

    # calculate lat/lon vecs
    if not isLatLon:
        lat_vec, lon_vec = CovertITM2LatLon.ITM2WGS(x_vec, y_vec)
        lat_vec = np.reshape(lat_vec, [len(lat_vec), 1])
        lon_vec = np.reshape(lon_vec, [len(lon_vec), 1])
        NA_points_ls = np.append(lat_vec, lon_vec, 1)
    else:
        lat_vec = x_vec
        lon_vec = y_vec
    # # Update map file
    # CovertITM2LatLon.createFoliumMap(NA_points_ls, np.mean(NA_points_ls, 0), labels=pci_vec)
    #
    # #%% open map in browser
    # CovertITM2LatLon.showMap('points_map.html')

    return (lon_vec,lat_vec,pci_vec)

def get_hypter_spectral_imaginery(data_filename,data_dirname):
    bands = range(1, 13)
    VenusImage_ls = []
    for b in bands:
        data_filename = data_filename.split('.')
        num2replace = len(str(b-1))
        data_filename[0] = data_filename[0][:-num2replace] + str(b)
        data_filename = data_filename[0] + '.' + data_filename[1]
        VenusImage_band = ImportVenusModule.getVenusData(data_dirname, data_filename)
        VenusImage_ls += [VenusImage_band]
    VenusImage = np.asarray(VenusImage_ls)
    VenusImage = np.transpose(VenusImage, axes=(1, 2, 0))


    #%% Get lat/lon directly from VENUS data - Get kiryatAta only

    x = getLatLon_fromTiff.convert_raster_to_geocoords(os.path.join(data_dirname,data_filename), zone_number=17, zone_letter='T')
    # unpack lat/lon
    lon_mat = x[:, :, 0]
    lat_mat = x[:, :, 1]

    return lon_mat,lat_mat,VenusImage

def get_PCI_ROI(roi,xy_pci):
    # # example ROI format:
    # xmin_cut = 35.095
    # xmax_cut = 35.120
    # ymin_cut = 32.802
    # ymax_cut = 32.818
    lon_vec, lat_vec, pci_vec = xy_pci
    xmin_cut, xmax_cut, ymin_cut, ymax_cut = roi[0][0], roi[0][1], roi[1][0], roi[1][1]
    # Filter the scatter points to include only those within the ROI
    scatter_indices = (lon_vec >= xmin_cut) & (lon_vec <= xmax_cut) & \
                      (lat_vec >= ymin_cut) & (lat_vec <= ymax_cut)

    filtered_x = lon_vec[scatter_indices]
    filtered_y = lat_vec[scatter_indices]
    filtered_PCI = pci_vec.values
    filtered_PCI = filtered_PCI[scatter_indices.ravel()]
    points_PCI = np.c_[filtered_x, filtered_y, filtered_PCI]
    return points_PCI


def create_segments_mask(hys_img,segment_mask,masks_tags_bounds):
    # this part create mask based on segmented PCI image
    assert(len(masks_tags_bounds)%2==0)
    masks_segments = pc_utils.divide_array(segment_mask, *masks_tags_bounds)
    masks_tags_numerical = (np.mean([0, masks_tags_bounds[0]]),) + tuple(
        (np.mean([masks_tags_bounds[ii + 1], masks_tags_bounds[ii]])) for ii in
        range(1, len(masks_tags_bounds) - 2, 2)) + (np.mean([masks_tags_bounds[-1], 100]),) # numerical value of tags as a mean of lower bound and upper bound of each segement, this will be used as tag value for each segement


    print(f'# of masks segments: {len(masks_segments)}')
    mask_all_channel_values=[]
    for ms in masks_segments:
        mask_all_channel_values.append(np.asarray(pc_utils.apply_masks_and_average(hys_img, ms)))

    assert(len(mask_all_channel_values)==len(masks_tags_numerical))

    return mask_all_channel_values,masks_tags_numerical


def create_proximity_mask(xy_points_merge,X_Grid,Y_Grid,threshold=10e-5,*arg):
    # Build a KD-tree for the features
    tree = cKDTree(xy_points_merge)
    # Flatten grid points for querying
    grid_points = np.vstack((X_Grid.ravel(), Y_Grid.ravel())).T

    # Query the nearest distance to a feature for each grid point
    distances, _ = tree.query(grid_points)

    # Create a binary mask where distance <= threshold is 1, else 0
    KDTree_mask = (distances <= threshold).astype(int).reshape(X_Grid.shape)

    return KDTree_mask

def get_mask_from_roads_gdf(npz_filename,data=None):
    if os.path.exists(npz_filename):
        print(f"File '{npz_filename}' exists. Loading data...")
        return load_npz(npz_filename).toarray()

    else:
        if data is None:
            raise ValueError("No data provided to save when the file does not exist.")
        print(f"File '{npz_filename}' does not exist. Creating data...")
        # %% Get only pixels that intersect with roads

        roi = data["roi"]
        lon_mat=data["Y_cropped"]
        lat_mat=data["X_cropped"]
        xmin_cut, xmax_cut, ymin_cut, ymax_cut = roi[0][0], roi[0][1], roi[1][0], roi[1][1]
        lat_range = (ymin_cut, ymax_cut)
        lon_range = (xmin_cut, xmax_cut)
        coinciding_mask = pc_utils.get_pixels_intersect_with_roads(lon_mat, lat_mat, lon_range,
                                                                   lat_range)  # assume prefect fit and registration is not needed
        rowAndColIdx = np.argwhere(coinciding_mask)
        save_npz(npz_filename, csr_matrix(coinciding_mask))
        print(f"Saved compressed binary mask of OpenStreetMap roads into '{npz_filename}'.")

        return coinciding_mask

def fill_segement_pixels_to_curves(GT_xy_PCI,points_PCI,segment_id,X_cropped,Y_cropped):
    x_segment = GT_xy_PCI[0].values
    y_segment = GT_xy_PCI[1].values
    polygon_coords = np.array([x_segment, y_segment
                               ]).T
    # Suppose polygon_coords is a list of (x, y) tuples in the same coordinate system as Xgrid/Ygrid
    polygon_path = pltPath(polygon_coords)

    # Flatten the grid to test all points at once
    points = np.column_stack((X_cropped.ravel(), Y_cropped.ravel()))

    # Check which points lie inside the polygon
    inside = polygon_path.contains_points(points, radius=0.1)
    print(inside.any())

    # Reshape 'inside' back to the shape of Xgrid/Ygrid
    inside_mask = inside.reshape(X_cropped.shape)
    fill_mask = np.zeros_like(X_cropped, dtype=np.uint8)
    fill_mask[inside_mask] = 1
    plt.figure(114);
    plt.imshow(fill_mask)

    # Flatten the grid for nearest-neighbor search

    grid_points = np.vstack((X_cropped.ravel(), Y_cropped.ravel())).T
    tree = cKDTree(grid_points)

    # Map real-world coordinates to pixel indices
    pixel_indices = []
    for x, y in polygon_coords:
        dist, idx = tree.query([x, y])  # Nearest pixel index
        pixel_indices.append(np.unravel_index(idx, X_cropped.shape))

    # Create a blank mask and draw the line
    height, width = X_cropped.shape
    mask = np.zeros((height, width), dtype=np.uint8)

    for (r1, c1), (r2, c2) in zip(pixel_indices, pixel_indices[1:]):
        rr, cc = line(r1, c1, r2, c2)
        mask[rr, cc] = 1
    plt.figure(115);
    plt.plot(x_segment, y_segment, 'r.');
    plt.pcolormesh(X_cropped, Y_cropped, pc_utils.morphological_operator(mask, 'dilation',
                                                                         'square',
                                                                         20))
    ###