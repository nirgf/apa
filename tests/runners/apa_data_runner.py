import os.path
import json
import numpy as np
from pathlib import Path
import src.utils.io_utils
from src.utils.apa_tester_utils import get_GT_xy_PCI,get_PCI_ROI,get_mask_from_roads_gdf,get_hypter_spectral_imaginery,create_proximity_mask
import src.utils.PrepareDataForNN_module as pp
from src.utils import ReadDetroitDataModule
import matplotlib.pyplot as plt
import src.utils.point_cloud_utils as pc_utils
import src.utils.pc_plot_utils as plt_utils
from scipy.interpolate import griddata
## Make plots interactive
import matplotlib
import h5py

import src.utils.io_utils as io_utils

# matplotlib.use('Qt5Agg')
matplotlib.use('TkAgg')

cmap_me = plt_utils.get_lighttraffic_colormap()
plt.ion()

REPO_ROOT=os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))




#%% Generate Database For NN
def create_database_from_VENUS(config_path,data_dirname,data_filename,metadata_filename, excel_path,output_path=None):

    ### Get data and prepare it for NN ###
    # This also saves the data as .h5 files
    config = io_utils.read_yaml_config(config_path)
    config=io_utils.fill_with_defaults(config['config'])
    X_cropped,Y_cropped,hys_img,points_merge_PCI,coinciding_mask,grid_value,segment_mask =\
        process_geo_data(config,data_dirname=data_dirname, data_filename=data_filename,excel_path=excel_path)
    road_hys_filter = np.reshape(coinciding_mask, list(np.shape(coinciding_mask)) + [1])
    # Gets the roads in general
    crop_size=config['preprocessing']['augmentations']['crop_size']
    hys_roads = np.repeat(road_hys_filter, 12, -1)*hys_img
    NN_inputs = pp.crop_image_to_segments(hys_roads, crop_size=crop_size, overlap=0.4, image_dim=12)
    NN_inputs[np.isnan(NN_inputs)] = 0
    
    # Gets only the labeled roads
    labeled_road_mask = np.ones(np.shape(coinciding_mask))
    labeled_road_mask[np.isnan(segment_mask)] = 0
    labeled_road_mask = np.reshape(labeled_road_mask*coinciding_mask, list(np.shape(labeled_road_mask)) + [1])
    hys_labeled_roads = np.repeat(labeled_road_mask, 12, -1)*hys_img
    NN_labeled_inputs = pp.crop_image_to_segments(hys_labeled_roads, crop_size=crop_size, overlap=0.4, image_dim=12)
    NN_labeled_inputs[np.isnan(NN_labeled_inputs)] = 0
    true_labels_full_image = np.reshape(segment_mask, list(np.shape(segment_mask)) + [1]) * labeled_road_mask
    true_labels_full_image[np.isnan(true_labels_full_image)] = 0
    true_labels = pp.crop_image_to_segments(true_labels_full_image, crop_size=crop_size, overlap=0.4, image_dim=1)
    
    # Remove frames with zeros only
    non_zero_idx = np.argwhere(np.sum(np.sum(np.sum(true_labels, -1), -1), -1) > 0)
    fin_NN_inputs = NN_inputs[non_zero_idx[:, 0], :, :, :]
    fin_true_labels = true_labels[non_zero_idx[:, 0], :, :, :]
    fin_NN_labeled_inputs = NN_labeled_inputs[non_zero_idx[:, 0], :, :, :]
    
    ### Save the data ###
    pp.save_cropped_segments_to_h5(fin_NN_inputs, 'All_RoadVenus.h5')
    pp.save_cropped_segments_to_h5(fin_true_labels, 'PCI_labels.h5')
    pp.save_cropped_segments_to_h5(fin_NN_labeled_inputs, 'Labeld_RoadsVenus.h5')

#%% Update Git Rules so the push will not get stuck
# import UpdateGitIgnore
# UpdateGitIgnore.main()


def cropROI_Venus_image(roi,lon_mat,lat_mat,VenusImage):
    xmin_cut, xmax_cut, ymin_cut, ymax_cut = roi[0][0], roi[0][1], roi[1][0], roi[1][1]
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

    # normalize spectral image for all bands
    hys_img = VenusImage[y_ind_min:y_ind_max, x_ind_min:x_ind_max, :].astype(float)
    hys_img=pc_utils.normalize_hypersepctral_bands(hys_img)

    # Crop the X, Y, and Z arrays based on these indices
    X_cropped = lat_mat_KiryatAta
    Y_cropped = lon_mat_KiryatAta
    # Apply the mask to the image
    Z_cropped = kiryatAtaImg

    return X_cropped,Y_cropped,hys_img


def process_geo_data(config,data_dirname,data_filename,excel_path):
    lon_mat, lat_mat, VenusImage = get_hypter_spectral_imaginery(data_filename, data_dirname)
    if "roi" in config["data"]:
        rois = config["data"]["rois"]
        roi = rois[0]
        roi = ((roi[0], roi[1]), (roi[2], roi[3]))

    else:
        # roi = ((35.095, 35.120), (32.802, 32.818))  # North East Kiryat Ata for train set
        # roi = ((35.064, 35.072), (32.746, 32.754))  # South West Kiryat Ata for test set
        # roi = ((-83.14294, -83.00007), (42.34429, 42.39170))  # Detroit test data
        roi = (np.min(lat_mat), np.max(lat_mat),(np.min(lon_mat), np.max(lon_mat)))  # Use all data
        rois=[roi]

    GT_xy_PCI=get_GT_xy_PCI(excel_path, isLatLon=True)

    points_PCI = get_PCI_ROI(roi,GT_xy_PCI)

    X_cropped,Y_cropped,hys_img = cropROI_Venus_image(roi,lon_mat,lat_mat,VenusImage)
    npz_filename=os.path.join(REPO_ROOT,'data/Detroit/masks_OpenStreetMap/Detroit_OpenSteet_roads_mask.npz')
    coinciding_mask = get_mask_from_roads_gdf(npz_filename, {"roi":roi,"X_cropped":X_cropped,"Y_cropped":Y_cropped})


    # scatter_plot_with_annotations(points_PCI,ax_roi)
    binary_mask = np.zeros(hys_img.shape[:-1])


    # this function merge different lane into one PCI (assumption, may not always be valid)
    # TODO: optimize threshold
    points_merge_PCI = pc_utils.merge_close_points(points_PCI[:, :2], points_PCI[:, 2], 50e-5)  # TODO:


    xy_points_merge = points_merge_PCI[:, :2]

    # create a mask based on proximity to point cloud data point
    extended_mask = create_proximity_mask(xy_points_merge,X_cropped,Y_cropped)
    # to mask out coninciding mask only where there it a PCI data
    # TODO: optimzie radius and size of sturcture element in the morphological operators
    combine_mask_roads = pc_utils.morphological_operator(extended_mask,'dilation',
                                                         'square',
                                                          20) \
                          * coinciding_mask
    combine_mask_roads = pc_utils.morphological_operator(combine_mask_roads,'closing','disk', 5)

    # create a segemented image of PCI values based on extendedn mask
    grid_value = griddata(points_PCI[:,:2], points_PCI[:, 2], (X_cropped, Y_cropped), method='nearest')
    segment_mask = grid_value * combine_mask_roads
    segment_mask = pc_utils.nan_arr(segment_mask)  # segment_mask[segment_mask <= 0] = np.nan


    return X_cropped,Y_cropped,hys_img,points_merge_PCI,coinciding_mask,grid_value,segment_mask


# Function to save multi-band image parts and their tags
def save_to_hdf5(save_folder, file_name, segements, tags, metadata=None):
    with h5py.File(os.path.join(save_folder, file_name), 'a') as f:
        for i, (arr, tag) in enumerate(zip(segements, tags)):
            dataset_name = f'avg_PCI_{tag}'

            if dataset_name in f:
                if not (arr is not None and arr.size > 0):
                    print('Skip:',dataset_name)
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




if __name__ == "__main__":
    # change only these paths or the ROI
    # %% Get venus data
    parent_path = ''
    # data_dirname = os.path.join(parent_path, 'venus data/Detroit_20230710/')
    data_dirname='/Users/nircko/DATA/apa/Detroit_20230710'
    data_filename = 'VENUS-XS_20230710-160144-000_L2A_DETROIT_C_V3-1_FRE_B1.tif'
    metadata_filename = 'data/dummy_metadata.json'

    convert_KML2CSV=False
    if convert_KML2CSV:
        kml_fullpath='/Users/nircko/DATA/apa/Detroit_20230710/Detroit_metadata/Pavement_Condition.kml'
        PCI_df, roi = ReadDetroitDataModule.parse_kml(kml_file = kml_fullpath)
        excel_path = 'data/Detroit/Pavement_Condition.csv'
    else:
        excel_path=os.path.join(REPO_ROOT,'data/Detroit/Pavement_Condition.csv')

    config_path = '/Users/nircko/GIT/apa/configs/apa_config.yaml'
    create_database_from_VENUS(config_path,data_dirname, data_filename,metadata_filename, excel_path)



