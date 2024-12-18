import os.path
import numpy as np
from src.CONST import bands_dict
import src.utils.apa_tester_utils as apa_utils
import src.utils.PrepareDataForNN_module as pp
from src.utils import ReadDetroitDataModule
import matplotlib.pyplot as plt
import src.utils.point_cloud_utils as pc_utils
import src.utils.pc_plot_utils as plt_utils
## Make plots interactive
import matplotlib

import src.utils.io_utils as io_utils

# matplotlib.use('Qt5Agg')
matplotlib.use('TkAgg')

cmap_me = plt_utils.get_lighttraffic_colormap()
plt.ion()

REPO_ROOT=os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))



#%% Generate Database For NN
def create_database_from_VENUS(config_path,data_dirname,data_filename,metadata_filename, excel_path,output_path=None):
    ### Get data and prepare it for NN ###

    # Read config and merge it with default values of configs if some keys are missing
    config = io_utils.read_yaml_config(config_path)
    config=io_utils.fill_with_defaults(config['config'])

    # Main function for doing geo-reference between PCI data and HSI images
    X_cropped,Y_cropped,hys_img,points_merge_PCI,coinciding_mask,segment_mask =\
        apa_utils.process_geo_data(config,data_dirname=data_dirname, data_filename=data_filename,excel_path=excel_path)

    stat_from_segments = apa_utils.analyze_pixel_value_ranges(hys_img, segment_mask)
    stat_from_segments = [pc_utils.get_stats_from_segment_spectral(
        np.asarray(pc_utils.apply_masks_and_average(hys_img, segment_mask == i))) for i in [1, 2, 3]]
    road_hys_filter = np.reshape(coinciding_mask, list(np.shape(coinciding_mask)) + [1])

    # Gets the roads in general
    # crop_size=config['preprocessing']['augmentations']['crop_size'][0]
    hys_roads = np.repeat(road_hys_filter, 12, -1)*hys_img
    NN_inputs = pp.crop_image_to_segments(config,hys_roads, image_dim=12)
    NN_inputs[np.isnan(NN_inputs)] = 0

    # TODO: validate config values and add output path to functions that outputs files
    # Gets only the labeled roads
    labeled_road_mask = np.ones(np.shape(coinciding_mask))
    labeled_road_mask[np.isnan(segment_mask)] = 0
    labeled_road_mask = np.reshape(labeled_road_mask*coinciding_mask, list(np.shape(labeled_road_mask)) + [1])
    hys_labeled_roads = np.repeat(labeled_road_mask, 12, -1)*hys_img
    NN_labeled_inputs = pp.crop_image_to_segments(config,hys_labeled_roads, image_dim=12)
    NN_labeled_inputs[np.isnan(NN_labeled_inputs)] = 0
    true_labels_full_image = np.reshape(segment_mask, list(np.shape(segment_mask)) + [1]) * labeled_road_mask
    true_labels_full_image[np.isnan(true_labels_full_image)] = 0
    true_labels = pp.crop_image_to_segments(config,true_labels_full_image, image_dim=1)
    
    # Remove frames with zeros only
    non_zero_idx = np.argwhere(np.sum(np.sum(np.sum(true_labels, -1), -1), -1) > 0)
    fin_NN_inputs = NN_inputs[non_zero_idx[:, 0], :, :, :]
    fin_true_labels = true_labels[non_zero_idx[:, 0], :, :, :]
    fin_NN_labeled_inputs = NN_labeled_inputs[non_zero_idx[:, 0], :, :, :]
    # TODO: validate config values and add output path to functions that outputs files
    ### Save the data ###
    pp.save_cropped_segments_to_h5(fin_NN_inputs, 'All_RoadVenus.h5')
    pp.save_cropped_segments_to_h5(fin_true_labels, 'PCI_labels.h5')
    pp.save_cropped_segments_to_h5(fin_NN_labeled_inputs, 'Labeld_RoadsVenus.h5')



if __name__ == "__main__":
    # change only these paths
    parent_path = ''
    config_path = os.path.join(apa_utils.REPO_ROOT,'configs/apa_config.yaml')
    data_dirname='/Users/nircko/DATA/apa/Detroit_20230710'

    data_filename = 'VENUS-XS_20230710-160144-000_L2A_DETROIT_C_V3-1_FRE_B1.tif'
    # make use of dummy metadata until full metadata will be available
    metadata_filename = 'data/dummy_metadata.json'

    convert_KML2CSV=False # if need to convert KML file into csv
    if convert_KML2CSV:
        kml_fullpath='Detroit/Pavement_Condition.kml'
        PCI_df, roi = ReadDetroitDataModule.parse_kml(kml_file = kml_fullpath)
        excel_path = 'data/Detroit/Pavement_Condition.csv'
    else:
        excel_path=os.path.join(REPO_ROOT,'data/Detroit/Pavement_Condition.csv')

    create_database_from_VENUS(config_path,data_dirname, data_filename,metadata_filename, excel_path)



