import os.path
import numpy as np
from src.CONST import bands_dict
import src.utils.apa_tester_utils as apa_utils
from src.utils.apa_tester_utils import PavementDataProcessor
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
def create_database_from_VENUS(config_path,data_dirname,data_filename,metadata_filename, excel_path,output_dirname=None,debug_plots=True):
    ### Get data and prepare it for NN ###

    # Read config and merge it with default values of configs if some keys are missing
    config = io_utils.read_yaml_config(config_path)
    config=io_utils.fill_with_defaults(config['config'])
    lon_mat, lat_mat, VenusImage,rois=PavementDataProcessor.image_data_loader(config, data_dirname, data_filename, metadata_filename)
    print(f'Selected ROIs:{rois}')

    # Create data set for each ROI
    for roi in rois:
        # Main function for doing geo-reference between PCI data and HSI images
        pdp=PavementDataProcessor(config)
        pdp.import_data(lon_mat, lat_mat, VenusImage, excel_path, roi)
        X_cropped=pdp.X_cropped
        Y_cropped=pdp.Y_cropped
        hys_img=pdp.HSI
        coinciding_mask,segment_mask,segID_PCI_LUT = pdp.create_roads_segID_mask()
        if segID_PCI_LUT is not None:
            mask_null_fill_value = config["preprocessing"].get("mask_null_fill_value", 0)
            # Create a mapping from keys to unique integers
            unique_key_map = {key: idx for idx, key in enumerate(segID_PCI_LUT.keys()) if not isinstance(key, int)}
            key_to_int_map = {int(key): int(segID_PCI_LUT[key]) for key in segID_PCI_LUT.keys()}
            # Convert the dictionary to a NumPy array
            numerical_segID_PCI_LUT = np.array([(int(key), value) for key, value in segID_PCI_LUT.items()])
            filled_with = np.nan_to_num(segment_mask, nan=mask_null_fill_value)
            boudningbox_list_labeled_image = pc_utils.process_labeled_image(hys_img, segment_mask, segID_PCI_LUT,
                                                                            dilation_radius=1)
            replaced_mask = np.vectorize(key_to_int_map.get)(filled_with.astype('int'))
            # replace nan/None/null with some other value
            segment_mask = np.where(replaced_mask == None, mask_null_fill_value, replaced_mask).astype('int')

        stat_from_segments = apa_utils.analyze_pixel_value_ranges(hys_img, segment_mask)
        wavelengths_array = 1e-3 * np.array([info['wavelength'] for info in bands_dict.values()])
        if debug_plots:
            plt.figure()
            plt_utils.plot_spectral_curves(wavelengths_array, stat_from_segments,None,'mean')
            plt.suptitle(Path(data_filename).stem[:-10])

            plt.figure(figsize=(8, 6))

            # Plot the original pcolormesh grid
            plt.pcolormesh(X_cropped, Y_cropped, hys_img[:,:,-1], cmap='gray', shading='auto')
            # plt.colorbar(label="Z_cropped values")

            # Overlay the mask using pcolormesh
            plt.pcolormesh(X_cropped, Y_cropped, pc_utils.nan_arr(segment_mask.astype('float')), cmap=cmap_me, shading='auto', alpha=0.6)
            plt.colorbar(label="Mask values")

            plt.title("Mask Overlay on Grid")
            plt.xlabel("X")
            plt.ylabel("Y")
            plt.show()

        binary_seg_mask = (segment_mask > 0)*1
        road_hys_filter = np.reshape(binary_seg_mask, list(np.shape(segment_mask)) + [1])

        # Gets the roads in general
        # crop_size=config['preprocessing']['augmentations']['crop_size'][0]
        num_of_channels = np.shape(hys_img)[-1]
        hys_roads = np.repeat(road_hys_filter, num_of_channels, -1)*hys_img
        NN_inputs = pp.crop_image_to_segments(config,hys_roads, image_dim=num_of_channels)
        NN_inputs[np.isnan(NN_inputs)] = 0

        # TODO: validate config values and add output path to functions that outputs files
        # Gets only the labeled roads
        labeled_road_mask = np.ones(np.shape(binary_seg_mask))
        labeled_road_mask[np.isnan(segment_mask)] = 0
        labeled_road_mask = np.reshape(labeled_road_mask*binary_seg_mask, list(np.shape(labeled_road_mask)) + [1])
        hys_labeled_roads = np.repeat(labeled_road_mask, num_of_channels, -1)*hys_img
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
        os.mkdir(output_dirname)

        output_path = Path(output_dirname).mkdir(parents=True, exist_ok=True)
        base_files = ["All_RoadVenus", "PCI_labels", "Labeld_RoadsVenus","BoudingBoxList"]
        formatted_string = "_".join(map(lambda x: str(round(x)), roi))

        pp.save_cropped_segments_to_h5(boudningbox_list_labeled_image, output_path / f"BoudingBoxList{formatted_string}.h5")
        pp.save_cropped_segments_to_h5(fin_NN_inputs, output_path / f"All_RoadVenus_{formatted_string}.h5")
        pp.save_cropped_segments_to_h5(fin_true_labels, output_path / f"PCI_labels_{formatted_string}.h5")
        pp.save_cropped_segments_to_h5(fin_NN_labeled_inputs, output_path / f"Labeld_RoadsVenus_{formatted_string}.h5")
        basename = Path(Path(Path(data_filename).stem).stem).stem + '.h5'
        print(basename)

    print('Done.')


if __name__ == "__main__":
    # change only paths

    #Detroit
    config_path = os.path.join(apa_utils.REPO_ROOT, 'configs/apa_config_detroit.yaml')

    # data_dirname = '/root/APA/Data/'
    #/
    # data_dirname = '/Users/nircko/DATA/apa/Detroit_20230710'
    # data_filename = 'VENUS-XS_20230710-160144-000_L2A_DETROIT_C_V3-1_FRE_B1.tif'


    # data_dirname = '/Users/nircko/Downloads/VENUS-XS_20240219-154042-000_L2A_DETROIT_C_V3-1'
    # data_filename='VENUS-XS_20240219-154042-000_L2A_DETROIT_C_V3-1_FRE_B1.tif'

    data_dirname = '/Users/nircko/Downloads/VENUS-XS_20230401-000000-000_L3A_DETROIT_C_V2-3'
    data_filename = 'VENUS-XS_20230401-000000-000_L3A_DETROIT_C_V2-3_FRC_B1.tif'

    excel_path = os.path.join(REPO_ROOT, 'data/Detroit/Pavement_Condition.csv')

    # # #Kiryat Ata
    # config_path = os.path.join(apa_utils.REPO_ROOT, 'configs/apa_config_kiryat_ata.yaml')
    # parent_path = '/Users/nircko/DATA/apa'
    # data_dirname = os.path.join(parent_path, 'venus data/VE_VM03_VSC_L2VALD_ISRAELWB_20230531/')
    # data_filename = 'VE_VM03_VSC_PDTIMG_L2VALD_ISRAELWB_20230531_FRE.DBL.TIF'
    # excel_path = os.path.join(REPO_ROOT, 'data/KiryatAta/seker_nezakim.xls')

    # make use of dummy metadata until full metadata will be available
    metadata_filename = 'data/dummy_metadata.json'
    # TODO : add the out path to cretae database script
    dataset_out_path = ''

    create_database_from_VENUS(config_path,data_dirname, data_filename,metadata_filename, excel_path)



