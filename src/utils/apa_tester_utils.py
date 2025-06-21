import os.path
import json
import numpy as np
from pathlib import Path
from PIL.ImageColor import colormap
import src.utils.pc_plot_utils
from src.CONST import get_spectral_bands
from scipy.interpolate import griddata
import pandas as pd
from src.utils import ImportVenusModule,getLatLon_fromTiff
from src.geo_reference import CovertITM2LatLon, GetOptimalRoadOffset
import src.utils.point_cloud_utils as pc_utils
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix, save_npz, load_npz
from skimage import exposure
import h5py
## Make plots interactive
from matplotlib.path import Path as pltPath
from scipy.spatial import cKDTree
from skimage.draw import line
import matplotlib.pyplot as plt
import matplotlib
from enums.datasets_enum import Dataset as enum_Dataset
from src.ImagePreProcessModule.shadow_manipulation_submodule import ylim, classShadow



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

@pc_utils.log_execution_time
def get_multi_spectral_imaginery_Venus(data_filename,data_dirname,config_data):
    if config_data['zone']=="Detroit" and config_data['big_tiff']==False:
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

    elif config_data['zone']=="Israel" and config_data['big_tiff']==True:
        VenusImage = ImportVenusModule.getVenusData(data_dirname, data_filename)
        # %% Get lat/lon directly from VENUS data - Get kiryatAta only
        x = getLatLon_fromTiff.convert_raster_to_geocoords(data_dirname + data_filename)
        # unpack lat/lon
        lon_mat = x[:, :, 0]
        lat_mat = x[:, :, 1]
        
        return lon_mat, lat_mat, VenusImage
    else:
        raise Exception("Currently only supports Detroit conversion of coordinates or Israel")

#%%        
def get_multi_spectral_imaginery_airbus(data_filename,data_dirname,config_data):
    
    if not (config_data['zone']=="Detroit" and config_data['big_tiff']==False):
        Exception("Config error, chack data and supported formats")
    
    Image_ls = []
    for band_filename in data_filename:
        Image_band = ImportVenusModule.getVenusData(data_dirname, band_filename)
        Image_ls += [Image_band]
    
    fin_image = np.asarray(Image_ls)
    fin_image = fin_image.reshape(-1, fin_image.shape[2], fin_image.shape[3])
    fin_image = np.transpose(fin_image, axes=(1, 2, 0))



    #%% Get lat/lon from data

    x = getLatLon_fromTiff.convert_raster_to_geocoords(os.path.join(data_dirname, band_filename),\
                                                       zone_number=17, \
                                                       zone_letter='T')
    # unpack lat/lon  
    lon_mat = x[:, :, 0]
    lat_mat = x[:, :, 1]
    
    return lon_mat, lat_mat, fin_image

#%%
def get_PCI_ROI(roi,xy_pci):
    # # example ROI format:
    # xmin_cut = 35.095
    # xmax_cut = 35.120
    # ymin_cut = 32.802
    # ymax_cut = 32.818
    lon_vec, lat_vec, pci_vec = xy_pci
    xmin_cut, xmax_cut, ymin_cut, ymax_cut = roi
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

@pc_utils.log_execution_time
def get_mask_from_roads_gdf(npz_filename,crop_rect,data=None):
    npz_file_path = Path(npz_filename)
    metda_filename=npz_file_path.with_suffix(".json")
    if not (os.path.exists(npz_filename) and os.path.exists(metda_filename)):
        if data is None:
            raise ValueError("No data provided to save when the file does not exist.")
        else:
            create_mask_from_roads_gdf(npz_filename, data)
    else:
        with open(metda_filename, 'r') as f:
            metadata_dict = json.load(f)
        is_roi_within_bounds=pc_utils.is_roi_within_bounds(data["roi"],metadata_dict["roi"])
        if not is_roi_within_bounds:
            print("Exisiting ROI of GDF roads is not bounded by the requested data") # Patched by Arie 30.01.2025
            
            #raise ValueError(f"Exisiting ROI:{metadata_dict["roi"]}  of GDF roads is not bounded by the requested: {data["roi"]}")
        print(f"File '{npz_filename}' found. Loading data...")
        # %% Get only pixels that intersect with roads
        coinciding_mask = load_npz(npz_filename).toarray()
        x_ind_min, y_ind_min, x_ind_max, y_ind_max = crop_rect
        return coinciding_mask[y_ind_min: y_ind_max, x_ind_min: x_ind_max]

@pc_utils.log_execution_time
def create_mask_from_roads_gdf(npz_filename, data):
    roi = data["roi"]
    lon_mat = data["Y_cropped"]
    lat_mat = data["X_cropped"]
    xmin_cut, xmax_cut, ymin_cut, ymax_cut = roi
    lat_range = (ymin_cut, ymax_cut)
    lon_range = (xmin_cut, xmax_cut)
    coinciding_mask = pc_utils.get_pixels_intersect_with_roads(lon_mat, lat_mat, lon_range,
                                                               lat_range)  # assume prefect fit and registration is not needed
    rowAndColIdx = np.argwhere(coinciding_mask)
    save_npz(npz_filename, csr_matrix(coinciding_mask))
    # save metadata file
    metadata = {"description": "Sparse matrix of roads in _", "author": "apa", "version": 1.0, "roi":roi}
    npz_file_path = Path(npz_filename)
    metda_filename=npz_file_path.with_suffix(".json")
    with open(metda_filename, "w") as f:
        json.dump(metadata, f)
    print(f"Saved compressed binary mask of OpenStreetMap roads into '{npz_filename}'.")
    pass

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


def cropROI_Venus_image(roi, lon_mat, lat_mat, MSP_Image, ie_datasource):
    xmin_cut, xmax_cut, ymin_cut, ymax_cut = roi
    # Get the indices corresponding to the cut boundaries
    idx_roi = np.argwhere((lon_mat > ymin_cut) & (lon_mat < ymax_cut) \
                               & (lat_mat > xmin_cut) & (lat_mat < xmax_cut))

    # %
    # Cut the image based on indices
    # Get the indices corresponding to the cut boundaries
    # %
    x_ind_min, x_ind_max = np.min(idx_roi[:, 1]), np.max(idx_roi[:, 1])
    y_ind_min, y_ind_max = np.min(idx_roi[:, 0]), np.max(idx_roi[:, 0])
    # Cut the image based on indices
    cropped_MSP_img = MSP_Image[y_ind_min:y_ind_max, x_ind_min:x_ind_max, :].astype(float)
    
    match enum_Dataset(ie_datasource):
        case enum_Dataset.venus_Detroit:    
            RGB_Img = cropped_MSP_img[:, :, [6, 3, 1]].astype(float)
        
        case enum_Dataset.airbus_HSP_Detroit:
            RGB_Img = cropped_MSP_img[:, :, 3:].astype(float)
        
        case enum_Dataset.airbus_Pan_Detroit:
            RGB_Img = np.repeat(cropped_MSP_img, repeats=3, axis=2)
            
    RGB_Img[RGB_Img <= 0] = np.nan
    norm_vec = np.nanpercentile(RGB_Img, q = 95, axis=(0, 1)).astype(float)
    for normBandIdx in range(len(norm_vec)):
        img=RGB_Img[:, :, normBandIdx]
        # Create a mask for NaN values
        RGB_Img[:, :, normBandIdx] = img / norm_vec[normBandIdx]
        # RGB_Img[:, :, normBandIdx] = equalize_image(img,1)


    lon_mat_roi = lon_mat[y_ind_min:y_ind_max, x_ind_min:x_ind_max]
    lat_mat_roi = lat_mat[y_ind_min:y_ind_max, x_ind_min:x_ind_max]

    # normalize spectral image for all bands
    # TODO: verfiy normalization method if has any effect on results
    cropped_MSP_img = pc_utils.normalize_hypersepctral_bands(cropped_MSP_img)

    # Crop the X, Y, and Z arrays based on these indices
    X_cropped = lat_mat_roi
    Y_cropped = lon_mat_roi
    # Apply the mask to the image
    Z_cropped = RGB_Img

    return X_cropped, Y_cropped, cropped_MSP_img, Z_cropped, (x_ind_min, y_ind_min, x_ind_max, y_ind_max)

def equalize_image(img,fill=0):
    nan_mask = np.isnan(img)
    # fill instead of nan value
    if fill==0:
        val = 0
    else:
        val = np.nanmean(img)
    image_nonan = np.nan_to_num(img, nan=val)
    image_eq = exposure.equalize_hist(image_nonan)
    image_eq[nan_mask] = np.nan # fill back with nan value
    return image_eq

def data_importer(config,data_dirname,data_filename,metadata_filename):
    
    ie_datasource = config['data']['enum_data_source']
    match enum_Dataset(ie_datasource):
        case enum_Dataset.venus_IL:
            lon_mat, lat_mat, VenusImage = get_multi_spectral_imaginery_Venus(data_filename, data_dirname, config["data"])
            
        case enum_Dataset.venus_Detroit:
            lon_mat, lat_mat, VenusImage = get_multi_spectral_imaginery_Venus(data_filename, data_dirname, config["data"])
                
        case enum_Dataset.airbus_HSP_Detroit:
            
            lon_mat, lat_mat, VenusImage = get_multi_spectral_imaginery_airbus(data_filename, data_dirname, config["data"])
            
        case enum_Dataset.airbus_Pan_Detroit:
            return

    if "rois" in config["data"]:
        rois = config["data"]["rois"]
    else:
        roi = [np.min(lat_mat), np.max(lat_mat), np.min(lon_mat), np.max(lon_mat)]  # Use all data
        rois = [roi]
    return lon_mat, lat_mat, VenusImage,rois

def process_geo_data(config, lon_mat, lat_mat, VenusImage, excel_path, roi):
    # TODO : change names and add support to enum
    enum_data_source = config["data"].get("enum_data_source")
    if config["data"]["zone"] == "Israel":
        GT_xy_PCI = get_GT_xy_PCI(excel_path, isLatLon=False)  # convert to Israel Coord to LatLon
        points_PCI, ROI_point_idx = get_PCI_ROI(roi, GT_xy_PCI[:3])
        ROI_seg = []
    else:
        GT_xy_PCI = get_GT_xy_PCI(excel_path, isLatLon=True)
        points_PCI, ROI_point_idx = get_PCI_ROI(roi, GT_xy_PCI[:3])
        seg_id = GT_xy_PCI[-1]
        ROI_seg = seg_id[ROI_point_idx]
    #USE only relevant ROI
    X_cropped, Y_cropped, hys_img, RGB_enchanced, cropped_rect = \
        cropROI_Venus_image(roi, lon_mat, lat_mat, VenusImage, enum_data_source) # retun also optical center x,y relative to orignal map
    OC_xy=cropped_rect[:2]
    print(f'Optical center ROI in xy[column][row]{OC_xy}\n')

    # this function merges different lanes into one PCI (assumption, may not always be valid)
    # TODO: optimize threshold
    points_merge_PCI = pc_utils.merge_close_points(points_PCI[:, :2], points_PCI[:, 2], 50e-5)  # TODO:
    xy_points_merge = points_merge_PCI[:, :2]


    # scatter_plot_with_annotations(points_PCI,ax_roi)
    binary_mask = np.zeros(hys_img.shape[:-1])

    # USE open street maps for creating map of all
    ie_datasource = config["data"]["enum_data_source"]
    
    if 'osx_map_mask_path' in config["preprocessing"]["georeferencing"]:        
        npz_filename = config["preprocessing"]["georeferencing"]["osx_map_mask_path"]
    else:
        npz_filename = 'data/Detroit/masks_OpenStreetMap/Detroit_OpenSteet_roads_mask.npz'
    
    ## Add ie_ie_datasource suffix
    npz_filename = npz_filename[:npz_filename.find('.npz')] + str(ie_datasource) + '.npz'
    
    # Get Mask
    npz_filename = os.path.join(REPO_ROOT, npz_filename)
    
    mask_data = {"roi":roi, "X_cropped":X_cropped, "Y_cropped":Y_cropped}
    coinciding_mask = get_mask_from_roads_gdf(npz_filename, cropped_rect, mask_data)

    lut = None
    if len(ROI_seg)==0: # if there is not segemts ID with the PCI data, use the old method for building mask for roads
        # create a segemented image of PCI/SegID values based on extended mask

        # create a mask based on proximity to point cloud data point
        # TODO: optimzie radius and size of sturcture element in the morphological operators
        extended_mask = create_proximity_mask(xy_points_merge, X_cropped, Y_cropped)
        # to mask out coninciding mask only where there it a PCI data
        combine_mask_roads = pc_utils.morphological_operator(extended_mask, 'dilation',
                                                             'square',
                                                             1) \
                             * coinciding_mask
        combine_mask_roads = pc_utils.morphological_operator(combine_mask_roads, 'closing', 'disk', 3)
        grid_value = griddata(points_PCI[:, :2], points_PCI[:, 2], (X_cropped, Y_cropped), method='nearest')
        classified_roads_mask = grid_value * combine_mask_roads
    else:
        # Dijkstra merge point
        if 'dijkstra_map_mask_path' in config["preprocessing"]["georeferencing"]:
            npz_filename = config["preprocessing"]["georeferencing"]["dijkstra_map_mask_path"]
        else:
            npz_filename = 'data/Detroit/masks_OpenStreetMap/Detroit_dijkstra_roads_mask.npz'
            
        ## Add ie_ie_datasource suffix
        npz_filename = npz_filename[:npz_filename.find('.npz')] + str(ie_datasource) + '.npz'
        
        # Get Mask
        npz_filename = os.path.join(REPO_ROOT, npz_filename)
        merge_points_dijkstra,lut = pc_utils.merge_points_dijkstra(npz_filename, X_cropped, Y_cropped, coinciding_mask,
                                                               points_PCI, ROI_seg)


        classified_roads_mask = merge_points_dijkstra

    wt  = config["preprocessing"].get("white_threshold", None)
    gyt = config["preprocessing"].get("gray_threshold", None)
    gdt = config["preprocessing"].get("grad_threshold", None)
    
    if all(x is None for x in (wt, gyt,gdt)):
        segment_mask=classified_roads_mask
    else:
    
        ###
        # Segments mask enhacement based on heuristic of gray color as asphalt indicator and object detection based on sobel gradient magnitude over Y channel (gray-level channel only)
        ###
        
        title_dict={"wt":wt,"gyt":gyt,"gdt":gdt}
        print(title_dict)
        
        # TODO: optimize all the parameters of enhancment
        
        enhance_morph_operator_type=config["preprocessing"].get("enhance_morph_operator_type", "dilation")
        enhance_morph_operator_size=config["preprocessing"].get("enhance_morph_operator_size", 5)
        
        if enhance_morph_operator_type is not None and enhance_morph_operator_size > 0:
            segment_mask_nan = pc_utils.nan_arr(
                pc_utils.morphological_operator_multiclass_mask(classified_roads_mask, enhance_morph_operator_type, 'square', enhance_morph_operator_size))
        else:
            segment_mask_nan = pc_utils.nan_arr(classified_roads_mask)

        if gyt>=0:
            
            # TODO: add function parameters to config + sort out outputs
            # _,_,imageClear = classShadow.removeShadow(hys_img[:, :, 0:3], brightness=70, contrast=2)
            
            ############## Ben's Shadow Part Should be incorperated here ####################
            
            ## Remove of non-gray parts due to incorrect geo-reference
            gray_color_enhanced, x_off, y_off = enhance_gray_based_on_RGB(config, RGB_enchanced, segment_mask_nan)

            # combine_mask_roads = pc_utils.morphological_operator_multiclass_mask(gray_color_enhanced, 'closing', 'square', 1)
            segment_mask_nan = pc_utils.nan_arr(gray_color_enhanced)

            # Use sobel gradient magnitude over Y channel to "remove" pixels containing suspected object and not roads
            segment_mask_nan=enhance_mask_grad(gdt, classified_roads_mask, RGB_enchanced,segment_mask_nan)

        segment_mask=segment_mask_nan

        # Offset all the variables by the new found offset :
        
        # Define source and target ranges
        ix_x_start = max(0, -x_off)
        ix_x_end = segment_mask.shape[0] - max(0, x_off)
        ix_y_start = max(0, -y_off)
        ix_y_end = segment_mask.shape[1] - max(0, y_off)
                
        X_cropped       = X_cropped[ix_x_start:ix_x_end, ix_y_start:ix_y_end]
        Y_cropped       = Y_cropped[ix_x_start:ix_x_end, ix_y_start:ix_y_end]
        hys_img         = hys_img  [ix_x_start:ix_x_end, ix_y_start:ix_y_end]
        coinciding_mask = coinciding_mask[ix_x_start:ix_x_end, ix_y_start:ix_y_end]
        segment_mask    = segment_mask   [ix_x_start:ix_x_end, ix_y_start:ix_y_end]
        
    return X_cropped, Y_cropped, hys_img, points_merge_PCI, coinciding_mask, segment_mask, lut


def enhance_mask_grad(grad_threshold, classified_roads_mask, RGB_enchanced, segment_mask_nan):
    # Remove of outliers based on object detection on the RGB image
    Y, _, _ = pc_utils.rgb_to_yuv(RGB_enchanced)
    # do sobel over mean filter luminace visible image
    _, _, mag = pc_utils.sobel_gradient(pc_utils.mean_filter(Y, 3))  # do sobel over mean filter luminace visible image

    # naive object detection using threshold over graident image, so find in region where there is a valid PCI mask where the grad of the RGB image is larger than defined threshold
    objects_detected_im_mask = np.where(
        pc_utils.morphological_operator_multiclass_mask(classified_roads_mask, 'dilation', 'square', 3) > 0, mag,
        0) > grad_threshold

    segment_mask_obj_removed = np.where(objects_detected_im_mask, np.nan, segment_mask_nan)
    print(
        f'After object detection, Not nan pixels in new mask {np.sum(~np.isnan(segment_mask_obj_removed))} pixels\nRemoved {100 * (1 - np.sum(~np.isnan(segment_mask_obj_removed)) / np.sum(~np.isnan(segment_mask_nan))):.2f}% of pixels from original mask\n\n')
    return segment_mask_obj_removed

def enhance_gray_based_on_RGB(config, RGB_enchanced, dilated_mask):
    """
    A function that takes mask of roads and refine it only where there is a gray color"
    Args:
        config:
        RGB_enchanced: an RGB 3 channel image post histogram equalization
        dilated_mask:
    """
    gray_threshold = config["preprocessing"].get("gray_threshold", 0.1)
    # Calculate differences between RGB channels
    diff_rg = np.abs(RGB_enchanced[:, :, 0] - RGB_enchanced[:, :, 1])  # Red vs Green
    diff_rb = np.abs(RGB_enchanced[:, :, 0] - RGB_enchanced[:, :, 2])  # Red vs Blue
    diff_gb = np.abs(RGB_enchanced[:, :, 1] - RGB_enchanced[:, :, 2])  # Green vs Blue
    # Identify gray regions (all channel differences are within tolerance)
    gray_color_mask = (diff_rg <= gray_threshold) & (diff_rb <= gray_threshold) & (diff_gb <= gray_threshold)
    intensity = np.mean(RGB_enchanced, axis=2)
    # Identify gray but not white regions
    white_threshold = config["preprocessing"].get("white_threshold", 0.92)
    not_white_mask = intensity < white_threshold
    gray_but_not_white_mask = gray_color_mask & not_white_mask
    
    # Compute spatial offset between an ideal OSM road map and the road mask from measurment using cross-correlation.
    int_max_shift = config["preprocessing"]["georeferencing"].get("max_reg_offset")
                
    # Cross-correlate & Find peak correlation
    from skimage.morphology import dilation, disk
    (x_off, y_off) = GetOptimalRoadOffset.find_local_road_offset_from_arrays( ~np.isnan(dilated_mask), \
                                                                              gray_but_not_white_mask, \
                                                                              max_shift=int_max_shift )

    
    # Combine gray regions with the input mask
    bin_offset_dialeted_mask = np.roll(~np.isnan(dilated_mask), shift=(y_off, x_off), axis=(0, 1))
    combined_mask = np.where(bin_offset_dialeted_mask & gray_but_not_white_mask,dilated_mask,0)
    
    return combined_mask, x_off, y_off


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



def vis_hys_image(hys_img,channels='rgb'):
    # a function that visualize multi channels image into a color/false color image
    if isinstance(channels,str) and channels=='rgb':
        return np.stack((hys_img[:, :, 6], hys_img[:, :, 3], hys_img[:, :, 1]),axis=-1)
    elif len(channels)>3:
        return pc_utils.false_color_hyperspectral_image(hys_img)
    elif len(channels)==3:
        return np.stack((hys_img[:, :, channels[0]], hys_img[:, :, channels[1]], hys_img[:, :, channels[2]]),axis=-1)
    elif len(channels)==1:
        return hys_img[:, :, channels[0]]
    else:
        raise ValueError("Invalid input: channels can be 'rgb' or list of valid channels.")


