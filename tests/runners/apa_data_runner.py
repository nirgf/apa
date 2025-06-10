import os.path
import numpy as np
from src.CONST import get_spectral_bands
import src.utils.apa_tester_utils as apa_utils
import src.utils.PrepareDataForNN_module as pp
from src.utils import ReadDetroitDataModule
import matplotlib.pyplot as plt
import src.utils.point_cloud_utils as pc_utils
import src.utils.pc_plot_utils as plt_utils
## Make plots interactive
import matplotlib
from pathlib import Path
import src.utils.io_utils as io_utils

# matplotlib.use('Qt5Agg')
matplotlib.use('TkAgg')

cmap_me = plt_utils.get_lighttraffic_colormap()
plt.ion()

REPO_ROOT=apa_utils.REPO_ROOT
print(REPO_ROOT)


#%% Generate Database For NN
def create_database(config,data_dirname,data_filename,metadata_filename, excel_path,output_dirname=None,debug_plots=True):
    ### Get data and prepare it for NN ###

    # Read config and merge it with default values of configs if some keys are missing
    config=io_utils.fill_with_defaults(config['config'])
    
    # Get the data from datasource, finally the output should comprise of :
    ##########################################################################
    #####                  [lon_mat, lat_mat, Image, rois]               #####
    ##########################################################################
                
    lon_mat, lat_mat, Msp_Image, rois = apa_utils.data_importer(config, data_dirname, data_filename, metadata_filename)
    
    print(f'Selected ROIs:{rois}')

    # Create data set for each ROI
    for roi in rois:
        
        # Main function for doing geo-reference between PCI data and HSI images
        X_cropped, Y_cropped, cropped_msp_img, points_merge_PCI, coinciding_mask, segment_mask, segID_PCI_LUT =\
            apa_utils.process_geo_data(config, lon_mat, lat_mat, Msp_Image, excel_path, roi)
        
        if segID_PCI_LUT is not None:
            mask_null_fill_value = config["preprocessing"].get("mask_null_fill_value", 0)
            # Create a mapping from keys to unique integers
            unique_key_map = {key: idx for idx, key in enumerate(segID_PCI_LUT.keys()) if not isinstance(key, int)}
            key_to_int_map = {int(key): int(segID_PCI_LUT[key]) for key in segID_PCI_LUT.keys()}
            # Convert the dictionary to a NumPy array
            numerical_segID_PCI_LUT = np.array([(int(key), value) for key, value in segID_PCI_LUT.items()])
            filled_with = np.nan_to_num(segment_mask, nan=mask_null_fill_value)
            boudningbox_list_labeled_image = pc_utils.process_labeled_image(cropped_msp_img, segment_mask, segID_PCI_LUT,
                                                                            dilation_radius=1)
            replaced_mask = np.vectorize(key_to_int_map.get)(filled_with.astype('int'))
            # replace nan/None/null with some other value
            segment_mask = np.where(replaced_mask == None, mask_null_fill_value, replaced_mask).astype('int')
        
        # TODO fix naming scheme
        stat_from_segments = apa_utils.analyze_pixel_value_ranges(cropped_msp_img, segment_mask)
        bands_dict = get_spectral_bands(config=config)
        wavelengths_array = 1e-3 * np.array([info['wavelength'] for info in bands_dict.values()])
        if debug_plots:
            plt_utils.plot_spectral_curves(wavelengths_array, stat_from_segments,None,'mean')
        binary_seg_mask = (segment_mask > 0)*1
        road_hys_filter = np.reshape(binary_seg_mask, list(np.shape(segment_mask)) + [1])

        # Gets the roads in general
        num_of_channels = np.shape(cropped_msp_img)[-1]
        hys_roads = np.repeat(road_hys_filter, num_of_channels, -1)*cropped_msp_img
        NN_inputs = pp.crop_image_to_segments(config,hys_roads, image_dim=num_of_channels)
        NN_inputs[np.isnan(NN_inputs)] = 0

        # TODO: validate config values and add output path to functions that outputs files
        # Gets only the labeled roads
        labeled_road_mask = np.ones(np.shape(binary_seg_mask))
        labeled_road_mask[np.isnan(segment_mask)] = 0
        labeled_road_mask = np.reshape(labeled_road_mask*binary_seg_mask, list(np.shape(labeled_road_mask)) + [1])
        hys_labeled_roads = np.repeat(labeled_road_mask, num_of_channels, -1)*cropped_msp_img
        NN_labeled_inputs = pp.crop_image_to_segments(config,hys_labeled_roads, image_dim=num_of_channels)
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
        if output_dirname is None:
            output_dirname=data_dirname
        # os.mkdir(output_dirname)

        output_path = Path(output_dirname).mkdir(parents=True, exist_ok=True)
        output_path = Path(output_dirname)
        base_files = ["All_RoadVenus", "PCI_labels", "Labeld_RoadsVenus","BoudingBoxList"]
        formatted_string = "_".join(map(lambda x: str(round(x)), roi)) + \
                           "_DataSource_" + str(config['data']['enum_data_source'])
        
        crop_size = eval(config["cnn_model"]["input_shape"])[0] # Assume symmetric crop size
        normalized_masks = []
        normalized_masks_labels = []
        
        ### This is Temp for testing ###
        import pickle

        # with open('boudningbox_list_labeled_image_enh.pickle', 'rb') as f:
        #     boudningbox_list_labeled_image = pickle.load(f)
            
        ### End Temp For Testing ###
        
        
        for i in range(len(boudningbox_list_labeled_image)):
            
            ## replace nans with zeros
            nan_idx = np.isnan(boudningbox_list_labeled_image[i]['mask'])
            boudningbox_list_labeled_image[i]['mask'][nan_idx] = 0
            
            normalized_masks = normalized_masks + \
                [pp.normalize_mask(boudningbox_list_labeled_image[i]['mask'], crop_size)]            
                
            label_mat = np.zeros(normalized_masks[i].shape[0:2])
            label_mat[normalized_masks[i][:, :, 0] != 0] = boudningbox_list_labeled_image[i]['label']
            
            normalized_masks_labels = normalized_masks_labels + [label_mat.reshape(list(np.shape(label_mat)) + [1])]
        
        normalized_masks_labels = np.asarray(normalized_masks_labels)
        normalized_masks = np.asarray(normalized_masks)
        
        # Save Seg Data
        pp.save_cropped_segments_to_h5(normalized_masks, output_path / f"BoudingBoxList{formatted_string}.h5")
        pp.save_cropped_segments_to_h5(normalized_masks_labels, output_path / f"BoudingBoxLabel{formatted_string}.h5")
        
        # Old data, might need to remove
        # TODO : Sort out the old thing
        pp.save_cropped_segments_to_h5(fin_NN_inputs, output_path / f"All_Road_{formatted_string}.h5")
        pp.save_cropped_segments_to_h5(fin_true_labels, output_path / f"PCI_labels_{formatted_string}.h5")
        pp.save_cropped_segments_to_h5(fin_NN_labeled_inputs, output_path / f"Labeld_Roads_{formatted_string}.h5")
        basename = Path(Path(Path(data_filename).stem).stem).stem + '.h5'
        print(basename)

    print('Done.')


if __name__ == "__main__":
      
    #Detroit
    config_path  = os.path.join(apa_utils.REPO_ROOT, 'configs/apa_config_detroit_Aribus_2025.yaml')
    
    # change only paths
    config = io_utils.read_yaml_config(config_path)

    data_dirname  = config['config']['data'].get('input_path')    
    data_filename = [ config['config']['data'].get('filename_NED'), \
                     config['config']['data'].get('filename_RGB') ]
    excel_path    = os.path.join(REPO_ROOT, 'data/Detroit/Pavement_Condition.csv')
    
    # TODO: sort out kiryat Ata paths
    # #Kiryat Ata
    # config_path = os.path.join(apa_utils.REPO_ROOT, 'configs/apa_config_kiryat_ata.yaml')
    # parent_path = '/home/ariep/Hyperspectral Road/brach_from_github/apa'
    # data_dirname = os.path.join(parent_path, 'venus data/VE_VM03_VSC_L2VALD_ISRAELWB_20230531/')
    # data_filename = 'VE_VM03_VSC_PDTIMG_L2VALD_ISRAELWB_20230531_FRE.DBL.TIF'
    # excel_path = os.path.join(REPO_ROOT, 'data/KiryatAta/seker_nezakim.xls')

    # make use of dummy metadata until full metadata will be available
    metadata_filename = 'data/dummy_metadata.json'
    
    
    # out path to cretae database script
    dataset_out_path = config['config']['data'].get('output_path')
    
    # Add root path to config & Change folder to it
    config['config']['data']['root_repo'] = apa_utils.REPO_ROOT
    os.chdir(config['config']['data']['root_repo'])
    
    create_database(config, data_dirname, data_filename,metadata_filename, excel_path,output_dirname=dataset_out_path)



