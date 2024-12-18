import os.path
import json
import numpy as np
from pathlib import Path
from PIL.ImageColor import colormap
from src.CONST import bands_dict
from scipy.interpolate import griddata
import pandas as pd
from src.utils import ImportVenusModule,getLatLon_fromTiff
from src.geo_reference import CovertITM2LatLon
import src.utils.point_cloud_utils as pc_utils
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix, save_npz, load_npz
import h5py
## Make plots interactive
from matplotlib.path import Path as pltPath
from scipy.spatial import cKDTree
from skimage.draw import line
import matplotlib.pyplot as plt
import matplotlib

# matplotlib.use('Qt5Agg')
matplotlib.use('TkAgg')

REPO_ROOT=os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

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

    pci_vec = df.pci.values
    
    if 'seg_id' in df.columns.tolist():
        seg_id = df.seg_id.values
    else : seg_id = []

    # verify if latitude should be x and not longitude
    if 'x' in df.columns:
        x_vec = df.x.values
        y_vec = df.y.values
    elif 'latitude' in df.columns:
        x_vec = df.latitude.values
        y_vec = df.longitude.values
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

    return (lon_vec,lat_vec,pci_vec, seg_id)

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
    filtered_PCI = pci_vec
    filtered_PCI = filtered_PCI[scatter_indices.ravel()]
    points_PCI = np.c_[filtered_x, filtered_y, filtered_PCI, ]
    return points_PCI, scatter_indices


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


def create_segments_mask(hys_img, segment_mask, masks_tags_bounds):
    """
    Creates mask segments based on segmented PCI image and their associated numerical tags.

    Parameters:
    -----------
    hys_img : np.ndarray
        Hyperspectral image array.
    segment_mask : np.ndarray
        Segmented PCI image.
    masks_tags_bounds : list
        List of bounds defining mask segments.

    Returns:
    --------
    mask_all_channel_values : list
        List of averaged masked values for each segment.
    masks_tags_numerical : tuple
        Numerical tag values for each segment, computed as the mean of lower and upper bounds.
    """
    # Ensure the bounds list has an even number of elements
    if len(masks_tags_bounds) % 2 != 0:
        raise ValueError("masks_tags_bounds must have an even number of elements.")

    # Divide the segment_mask into regions based on bounds
    masks_segments = pc_utils.divide_array(segment_mask, *masks_tags_bounds)

    # Calculate numerical tag values as the mean of bounds
    masks_tags_numerical = (
                               np.mean([0, masks_tags_bounds[0]])  # Lower bound for the first segment
                           ) + tuple(
        np.mean([masks_tags_bounds[i], masks_tags_bounds[i + 1]])
        for i in range(1, len(masks_tags_bounds) - 2, 2)
    ) + (
                               np.mean([masks_tags_bounds[-1], 100])  # Upper bound for the last segment
                           )

    print(f"Number of mask segments: {len(masks_segments)}")

    # Compute average masked values for each segment
    mask_all_channel_values = [
        np.asarray(pc_utils.apply_masks_and_average(hys_img, mask_segment))
        for mask_segment in masks_segments
    ]

    # Ensure the number of segments matches the number of tags
    if len(mask_all_channel_values) != len(masks_tags_numerical):
        raise ValueError("Mismatch between the number of mask segments and tags.")

    return mask_all_channel_values, masks_tags_numerical


def analyze_pixel_value_ranges(hys_img,segment_mask, masks_tags_numerical=[1,2,3]):
    stat_from_segments = [
        {"pci_value": i,
         "statistics": pc_utils.get_stats_from_segment_spectral(
             np.asarray(pc_utils.apply_masks_and_average(hys_img, segment_mask == i))
         )}
        for i in masks_tags_numerical
    ]
    return stat_from_segments


def cropROI_Venus_image(roi, lon_mat, lat_mat, VenusImage):
    xmin_cut, xmax_cut, ymin_cut, ymax_cut = roi[0][0], roi[0][1], roi[1][0], roi[1][1]
    # Get the indices corresponding to the cut boundaries
    kiryatAtaIdx = np.argwhere((lon_mat > ymin_cut) & (lon_mat < ymax_cut) \
                               & (lat_mat > xmin_cut) & (lat_mat < xmax_cut))

    # %
    # Cut the image based on indices
    # Get the indices corresponding to the cut boundaries
    # %
    x_ind_min, x_ind_max = np.min(kiryatAtaIdx[:, 1]), np.max(kiryatAtaIdx[:, 1])
    y_ind_min, y_ind_max = np.min(kiryatAtaIdx[:, 0]), np.max(kiryatAtaIdx[:, 0])
    # Cut the image based on indices
    kiryatAtaImg = VenusImage[y_ind_min:y_ind_max, x_ind_min:x_ind_max,
                   [6, 3, 1]].astype(float)
    kiryatAtaImg[kiryatAtaImg <= 0] = np.nan
    norm_vec = np.nanmax(kiryatAtaImg, axis=(0, 1)).astype(float)
    for normBandIdx in range(len(norm_vec)):
        kiryatAtaImg[:, :, normBandIdx] = kiryatAtaImg[:, :, normBandIdx] / norm_vec[normBandIdx]

    lon_mat_KiryatAta = lon_mat[y_ind_min:y_ind_max, x_ind_min:x_ind_max]

    lat_mat_KiryatAta = lat_mat[y_ind_min:y_ind_max, x_ind_min:x_ind_max]

    # normalize spectral image for all bands
    hys_img = VenusImage[y_ind_min:y_ind_max, x_ind_min:x_ind_max, :].astype(float)
    hys_img = pc_utils.normalize_hypersepctral_bands(hys_img)

    # Crop the X, Y, and Z arrays based on these indices
    X_cropped = lat_mat_KiryatAta
    Y_cropped = lon_mat_KiryatAta
    # Apply the mask to the image
    Z_cropped = kiryatAtaImg

    return X_cropped, Y_cropped, hys_img


def process_geo_data(config, data_dirname, data_filename, excel_path):
    lon_mat, lat_mat, VenusImage = get_hypter_spectral_imaginery(data_filename, data_dirname)
    if "rois" in config["data"]:
        rois = config["data"]["rois"]
        roi = rois[0]
        roi = ((roi[0], roi[1]), (roi[2], roi[3]))

    else:
        roi = (np.min(lat_mat), np.max(lat_mat), (np.min(lon_mat), np.max(lon_mat)))  # Use all data
        rois = [roi]

    GT_xy_PCI = get_GT_xy_PCI(excel_path, isLatLon=True)
    seg_id = GT_xy_PCI[-1]
    points_PCI, ROI_point_idx = get_PCI_ROI(roi, GT_xy_PCI[:3])
    ROI_seg = seg_id[ROI_point_idx]

    X_cropped, Y_cropped, hys_img = cropROI_Venus_image(roi, lon_mat, lat_mat, VenusImage)
    if 'osx_map_mask_path' in config["preprocessing"]["georeferencing"]:
        npz_filename = config["preprocessing"]["georeferencing"]["osx_map_mask_path"]
    else:
        npz_filename = os.path.join(REPO_ROOT, 'data/Detroit/masks_OpenStreetMap/Detroit_OpenSteet_roads_mask.npz')
    coinciding_mask = get_mask_from_roads_gdf(npz_filename,
                                                        {"roi": roi, "X_cropped": X_cropped, "Y_cropped": Y_cropped})

    # scatter_plot_with_annotations(points_PCI,ax_roi)
    binary_mask = np.zeros(hys_img.shape[:-1])

    # this function merge different lane into one PCI (assumption, may not always be valid)
    ### Plot the data// visualization only
    plt.figure()
    plt.pcolormesh(X_cropped, Y_cropped, coinciding_mask)
    plt.pcolormesh(X_cropped, Y_cropped, hys_img[:, :, -2], alpha=0.5)
    plt.scatter(points_PCI[:, 0], points_PCI[:, 1])

    # TODO: optimize threshold

    points_merge_PCI = pc_utils.merge_close_points(points_PCI[:, :2], points_PCI[:, 2], 50e-5)  # TODO:

    xy_points_merge = points_merge_PCI[:, :2]

    # create a mask based on proximity to point cloud data point
    extended_mask = create_proximity_mask(xy_points_merge, X_cropped, Y_cropped)
    # to mask out coninciding mask only where there it a PCI data
    # TODO: optimzie radius and size of sturcture element in the morphological operators
    combine_mask_roads = pc_utils.morphological_operator(extended_mask, 'dilation',
                                                         'square',
                                                         20) \
                         * coinciding_mask
    combine_mask_roads = pc_utils.morphological_operator(combine_mask_roads, 'closing', 'disk', 5)

    # Dijkstra merge point
    if 'dijkstra_map_mask_path' in config["preprocessing"]["georeferencing"]:
        npz_filename = config["preprocessing"]["georeferencing"]["dijkstra_map_mask_path"]
    else:
        npz_filename = os.path.join(REPO_ROOT, 'data/Detroit/masks_OpenStreetMap/Detroit_OpenSteet_roads_mask.npz')
    merge_points_dijkstra=pc_utils.merge_points_dijkstra(npz_filename,X_cropped, Y_cropped, coinciding_mask, points_PCI, ROI_seg)

    # create a segemented image of PCI values based on extendedn mask
    grid_value = griddata(points_PCI[:, :2], points_PCI[:, 2], (X_cropped, Y_cropped), method='nearest')
    segment_mask = grid_value * combine_mask_roads
    segment_mask = pc_utils.nan_arr(segment_mask)  # segment_mask[segment_mask <= 0] = np.nan
    stat_from_segments = analyze_pixel_value_ranges(hys_img, segment_mask)
    stat_from_segments = [pc_utils.get_stats_from_segment_spectral(
        np.asarray(pc_utils.apply_masks_and_average(hys_img, segment_mask == i))) for i in [1, 2, 3]]

    return X_cropped, Y_cropped, hys_img, points_merge_PCI, coinciding_mask, grid_value, segment_mask


# Function to save multi-band image parts and their tags
def save_to_hdf5(save_folder, file_name, segements, tags, metadata=None):
    with h5py.File(os.path.join(save_folder, file_name), 'a') as f:
        for i, (arr, tag) in enumerate(zip(segements, tags)):
            dataset_name = f'avg_PCI_{tag}'

            if dataset_name in f:
                if not (arr is not None and arr.size > 0):
                    print('Skip:', dataset_name)
                    continue
                # Handle appending logic if the dataset already exists
                dset = f[dataset_name]
                original_shape = dset.shape

                # the array to append has the same shape except for the first dimension
                new_shape = (original_shape[0],) + (original_shape[1] + arr.shape[1],)
                dset.resize(new_shape)  # Resize to accommodate new data
                dset[-arr.shape[0]:] = arr  # Append new data
                print(f"Appended data to existing dataset '{dataset_name}'.")

            else:
                if arr is not None and arr.size > 0:
                    # Create a new dataset with metadata if it doesn't exist
                    dset = f.create_dataset(dataset_name, data=arr, maxshape=(None,) + arr.shape[1:],
                                            compression="gzip")
                    dset.attrs['tag'] = tag  # Store the tag as an attribute
                    print(f"Created new dataset '{dataset_name}' with data and a tag.")

                    # Add custom metadata if provided
                    if metadata and i in metadata:
                        for key, value in metadata[i].items():
                            dset.attrs[key] = value  # Store custom metadata
                else:
                    # Create an empty dataset for None or empty arrays
                    dset = f.create_dataset(dataset_name, data=np.array([]), compression="gzip")
                    dset.attrs['tag'] = tag
                    print(f"Created empty dataset '{dataset_name}' with a tag.")

