import os.path
import json
import numpy as np
from pathlib import Path
from PIL.ImageColor import colormap
from CONST import bands_dict
import CovertITM2LatLon
import pandas as pd
import ImportVenusModule
import matplotlib.pyplot as plt
import point_cloud_utils as pc_utils
import pc_plot_utils as plt_utils
from scipy.interpolate import splprep, splev, griddata
## Make plots interactive
import matplotlib
import h5py
# matplotlib.use('Qt5Agg')
matplotlib.use('TkAgg')

cmap_me = plt_utils.get_lighttraffic_colormap()
plt.ion()

#%% Update Git Rules so the push will not get stuck
# import UpdateGitIgnore
# UpdateGitIgnore.main()

def get_GT_xy_PCI(xls_path):
    #%% get NA data
    df = pd.read_excel(xls_path)
    pci_vec = df.PCI
    x_vec = df.X
    y_vec = df.Y
    dates = df.S_Date

    # calculate lat/lon vecs
    lat_vec, lon_vec = CovertITM2LatLon.ITM2WGS(x_vec, y_vec)
    lat_vec = np.reshape(lat_vec, [len(lat_vec), 1])
    lon_vec = np.reshape(lon_vec, [len(lon_vec), 1])
    NA_points_ls = np.append(lat_vec, lon_vec, 1)
    # # Update map file
    # CovertITM2LatLon.createFoliumMap(NA_points_ls, np.mean(NA_points_ls, 0), labels=pci_vec)
    #
    # #%% open map in browser
    # CovertITM2LatLon.showMap('points_map.html')

    return (lon_vec,lat_vec,pci_vec)

def get_hypter_spectral_imaginery(data_filename,data_dirname,metadata_filename,metadata_dirname):
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

    return lon_mat,lat_mat,VenusImage,venusMetadata

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


def cropROI_Venus_image(roi,lon_mat,lat_mat,VenusImage):
    # # example ROI format:
    # xmin_cut = 35.095
    # xmax_cut = 35.120
    # ymin_cut = 32.802
    # ymax_cut = 32.818
    #

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

    #%% Get only pixels that intersect with roads
    lat_range = (ymin_cut, ymax_cut)
    lon_range = (xmin_cut, xmax_cut)
    coinciding_mask=pc_utils.get_pixels_intersect_with_roads(lon_mat_KiryatAta,lat_mat_KiryatAta,lon_range,lat_range) # assume prefect fit and registration is not needed
    rowAndColIdx = np.argwhere(coinciding_mask)

    return X_cropped,Y_cropped,hys_img,coinciding_mask


def process_geo_data(roi,data_dirname,data_filename,metadata_dirname,metadata_filename, excel_path):
    GT_xy_PCI=get_GT_xy_PCI(excel_path)
    points_PCI = get_PCI_ROI(roi,GT_xy_PCI)

    lon_mat,lat_mat,VenusImage,venusMetadata = get_hypter_spectral_imaginery(data_filename, data_dirname, metadata_filename, metadata_dirname)
    X_cropped,Y_cropped,hys_img,coinciding_mask = cropROI_Venus_image(roi,lon_mat,lat_mat,VenusImage)

    # scatter_plot_with_annotations(points_PCI,ax_roi)
    binary_mask = np.zeros(hys_img.shape[:-1])

    # this function merge different lane into one PCI (assumption, may not always be valid)
    # TODO: optimize threshold
    points_merge_PCI = pc_utils.merge_close_points(points_PCI[:, :2], points_PCI[:, 2], 50e-5)  # TODO:
    xy_points_merge = points_merge_PCI[:, :2]

    # # create a spline fit trajectory from point cloud using GreedyNN and this create a mask of of it (can be extended with existing mask)
    extended_mask, line_string, xy_spline = pc_utils.fill_mask_with_irregular_spline(xy_points_merge, X_cropped,
                                                                                     Y_cropped, binary_mask,
                                                                                     combine_mask=False)  # this return mask in the pixels the spline line passes through
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

    x_new, y_new = xy_spline

    return X_cropped,Y_cropped,hys_img,points_merge_PCI,x_new,y_new,coinciding_mask,grid_value,segment_mask

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


def stats_from_mask(mask_all_channel_values,X_cropped,Y_cropped,hys_img,points_merge_PCI,x_new,y_new,coinciding_mask,grid_value,segment_mask,plot=False ,plot_animation=False,dump_json=False):
    # Extract all the 'wavelength' values into a list
    wavelengths = [info['wavelength'] for info in bands_dict.values()]
    wavelengths_array = 1e-3 * np.array(wavelengths)
    wavelengths_bw_array = np.array([info['wavelength'] for info in bands_dict.values()])

    masks_tags_description=('Critical','Moderate','Good') # optional verbal description of the tags
    stats_30PCI = pc_utils.get_stats_from_segment_spectral(mask_all_channel_values[0])
    stats_50_70PCI = pc_utils.get_stats_from_segment_spectral(mask_all_channel_values[1])
    stats_85PCI = pc_utils.get_stats_from_segment_spectral(mask_all_channel_values[2])
    if dump_json:
        data = {
            'wavelengths_array': wavelengths_array.tolist(),
            'avg_30PCI': stats_30PCI[1].tolist(),
            'avg_85PCI': stats_85PCI[1].tolist(),
            'std_30PCI': stats_30PCI[2].tolist(),
            'std_85PCI': stats_85PCI[2].tolist()
        }

        # Convert to list and save as JSON
        with open("pci_spectral_stats.json", "w") as file:
            json.dump(data, file)

    def plot_spectral_curves():
        # plot spectoroms
        plt.figure(110+np.random.randint(1,10))
        plt.plot(wavelengths_array, stats_30PCI[1], 'r', label=f'Critical , N_AVG={stats_30PCI[0]}')
        plt.errorbar(wavelengths_array, stats_30PCI[1], yerr=stats_30PCI[2], fmt='o', color='r', alpha=0.5)
        plt.plot(wavelengths_array, stats_50_70PCI[1], 'y', label=f'Moderate , N_AVG={stats_50_70PCI[0]}')
        plt.errorbar(wavelengths_array, stats_50_70PCI[1], yerr=stats_50_70PCI[2], fmt='o', color='y', alpha=0.5)
        plt.plot(wavelengths_array, stats_85PCI[1], 'g', label=f'Good , N_AVG={stats_85PCI[0]}')
        plt.errorbar(wavelengths_array, stats_85PCI[1], yerr=stats_85PCI[2], fmt='o', color='g', alpha=0.5)
        plt.axvline(x=0.45, color='pink', linestyle='--', linewidth=2)
        plt.axvline(x=0.75, color='pink', linestyle='--', linewidth=2)
        plt.text(0.72, plt.ylim()[1] * 0.9, 'VIS', color='pink', fontsize=12, ha='center')
        plt.text(0.77, plt.ylim()[1] * 0.9, 'IR', color='pink', fontsize=12, ha='center')
        plt.title('Spectral Stats')
        plt.ylabel('AU')
        plt.xlabel('wavelength[mu]')
        plt.legend()

    if plot:
        plt_utils.plot_scatter_over_map(X_cropped,Y_cropped,hys_img,points_merge_PCI,x_new,y_new,coinciding_mask)
        plot_spectral_curves()

    plt.show()

    if plot_animation:
        from matplotlib.animation import FuncAnimation

        # Initialize the plot
        fig_ani, ax_ani = plt.subplots()
        ax_ani.set_xticks([])  # Remove x ticks
        ax_ani.set_yticks([])  # Remove y ticks
        ax_ani.set_title('Gradually Appearing Segments')

        # Adjust the axis to fill the figure
        ax_ani.set_position([0, 0, 1, 1])  # Fill entire figure with the axis
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)  # No margins around the plot
        demo_mask = pc_utils.nan_arr(coinciding_mask * grid_value)
        # Use pcolormesh to create the initial empty grid (binary mask)
        im_ax = ax_ani.pcolormesh(X_cropped, Y_cropped, hys_img[:, :, -1], cmap='gray')
        c_ax = ax_ani.pcolormesh(X_cropped, Y_cropped, segment_mask, cmap=cmap_me, vmin=10, vmax=100)
        # use 35.10 to miss missing HSI data
        # ax_ani.set_xlim(35.10, np.max(X_cropped))
        ax_ani.set_xlim(np.min(X_cropped), np.max(X_cropped))

        ax_ani.set_ylim(np.min(Y_cropped), np.max(Y_cropped))

        # Function to update the mask in each frame
        def update(frame):
            # Create a mask that gradually reveals more segments
            reveal_mask = np.full(demo_mask.shape, np.nan, dtype=float)
            reveal_mask[:10 * frame, :] = demo_mask[:10 * frame, :]
            reveal_mask = pc_utils.or_nan(reveal_mask, segment_mask)  # add existing PCI
            c_ax.set_array(reveal_mask.ravel())
            return c_ax,

        # Create the animation
        ani = FuncAnimation(fig_ani, update, frames=range(1, demo_mask.shape[0] // 10 + 1), blit=True, interval=500)

        # Turn off the grid lines and ticks
        plt.axis('off')
        # Save as AVI
        ani.save('animation.avi', writer='ffmpeg', fps=10)

        # Save as GIF
        ani.save('animation.gif', writer='imagemagick', fps=100)

    pass


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


def read_from_hdf5(file_name):
    arrays = []
    tags = []
    with h5py.File(file_name, 'r') as f:
        for key in f.keys():  # Iterate through dataset names
            print(f'key:{key}')
            arr = f[key][:]
            arrays.append(arr)
            tag = f.attrs[f'tag_{key.split("_")[1]}']  # Extract the tag by splitting the dataset name
            tags.append(tag)
    # Print the shapes and tags of the read data
    for i, (arr, tag) in enumerate(zip(arrays, tags)):
        print(f"Array {i}: shape={arr.shape}, tag={tag}")
    return arrays, tags


def create_hdf5_segemets_tags(roi,data_dirname,data_filename,metadata_dirname,metadata_filename, excel_path,masks_tags_bounds):
    X_cropped, Y_cropped, hys_img, points_merge_PCI, x_new, y_new, coinciding_mask, grid_value, segment_mask = process_geo_data(
        roi=roi, data_dirname=data_dirname, data_filename=data_filename, metadata_dirname=metadata_dirname,
        metadata_filename=metadata_filename, excel_path=excel_path)

    mask_all_channel_values,masks_tags_numerical = create_segments_mask(hys_img, segment_mask,masks_tags_bounds)
    basename = Path(Path(Path(data_filename).stem).stem).stem + '.h5'

    save_to_hdf5(data_dirname,basename,mask_all_channel_values,masks_tags_numerical)

    pass

def crop_runner_main(roi,data_dirname,data_filename,metadata_dirname,metadata_filename, excel_path):
    X_cropped, Y_cropped, hys_img, points_merge_PCI, x_new, y_new, coinciding_mask, grid_value, segment_mask = process_geo_data(
        roi=roi, data_dirname=data_dirname, data_filename=data_filename, metadata_dirname=metadata_dirname,
        metadata_filename=metadata_filename, excel_path=excel_path)

    unique_values, counts = pc_utils.analyze_and_plot_grouped_histogram(segment_mask,group_range=5,min_value=1)
    masks_tags_bounds = (30, 50, 70, 85)  # bounds tags of each segements in format
    # (segement1_upperbound,segement2_lowerbound,segement2_upperbound,segement3_lowerbound,segement3_upperbound.....,segement_N_lowerbound)
    mask_all_channel_values, masks_tags_numerical = create_segments_mask(hys_img, segment_mask,masks_tags_bounds)
    stats_from_mask(mask_all_channel_values, X_cropped, Y_cropped, hys_img, points_merge_PCI, x_new, y_new,
                    coinciding_mask, grid_value, segment_mask, plot=True, plot_animation=False, dump_json=False)

    pass


if __name__ == "__main__":
    # change only these paths or the ROI
    # %% Get venus data
    parent_path = '/Users/nircko/DATA/apa'
    data_dirname = os.path.join(parent_path, 'venus data/VE_VM03_VSC_L2VALD_ISRAELWB_20230531/')
    data_filename = 'VE_VM03_VSC_PDTIMG_L2VALD_ISRAELWB_20230531_FRE.DBL.TIF'
    metadata_filename = 'M02_metadata.csv'
    metadata_dirname = os.path.join(parent_path, 'venus data/')

    excel_path = 'seker_nezakim.xls'

    roi = ((35.095, 35.120), (32.802, 32.818))  # North East Kiryat Ata for train set
    # roi = ((35.064, 35.072), (32.746, 32.754))  # South West Kiryat Ata for test set

    # masks_tags_bounds = (30,30,60,60, 80, 85)  # PCI bounds tags of each segements in format
    # (segement1_upperbound,segement2_lowerbound,segement2_upperbound,segement3_lowerbound,segement3_upperbound.....,segement_N_lowerbound)
    # create_hdf5_segemets_tags(roi=roi,data_dirname=data_dirname,data_filename=data_filename, metadata_dirname=metadata_dirname,
    #                       metadata_filename=metadata_filename,excel_path=excel_path,masks_tags_bounds=masks_tags_bounds)

    crop_runner_main(roi=roi,data_dirname=data_dirname,data_filename=data_filename, metadata_dirname=metadata_dirname,
                          metadata_filename=metadata_filename,excel_path=excel_path)


