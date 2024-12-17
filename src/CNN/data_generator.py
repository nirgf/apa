import os
import numpy as np
import h5py
from crop_runner import create_hdf5_segemets_tags


def data_generator(hdf5_files, batch_size=128):
    while True:
        for file_name in hdf5_files:
            with h5py.File(file_name, 'r') as f:
                images = f['images']
                tags = f['tags']
                num_samples = images.shape[0]

                for start in range(0, num_samples, batch_size):
                    end = min(start + batch_size, num_samples)
                    yield images[start:end], tags[start:end]


# Function to read and process HDF5 data in chunks
def process_in_chunks(hdf5_files, chunk_size=1000):
    for file_name in hdf5_files:
        with h5py.File(file_name, 'r') as f:
            images = f['images']
            tags = f['tags']

            num_samples = images.shape[0]
            for start in range(0, num_samples, chunk_size):
                end = min(start + chunk_size, num_samples)
                image_chunk = images[start:end]  # Load a chunk of images
                tag_chunk = tags[start:end]  # Load corresponding tags

                # Process your data chunk (e.g., append to training set, perform calculations)
                # Add your custom processing code here
                yield image_chunk, tag_chunk



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

    masks_tags_bounds = (30, 50, 70, 85)  # PCI bounds tags of each segements in format
    # (segement1_upperbound,segement2_lowerbound,segement2_upperbound,segement3_lowerbound,segement3_upperbound.....,segement_N_lowerbound)
    create_hdf5_segemets_tags(roi=roi,data_dirname=data_dirname,data_filename=data_filename, metadata_dirname=metadata_dirname,
                          metadata_filename=metadata_filename,excel_path=excel_path,masks_tags_bounds=masks_tags_bounds)



    # Example usage
    hdf5_files = ['/Users/nircko/DATA/apa/venus data/VE_VM03_VSC_L2VALD_ISRAELWB_20230531/VE_VM03_VSC_PDTIMG_L2VALD_ISRAELWB_20230531_FRE.h5', '/Users/nircko/DATA/apa/venus data/VE_VM03_VSC_L2VALD_ISRAELWB_20230531/VE_VM03_VSC_PDTIMG_L2VALD_ISRAELWB_20230531_FRE.h5']
    for image_chunk, tag_chunk in process_in_chunks(hdf5_files):
        # Handle each chunk (e.g., train model, save processed data)
        print(f"Processed chunk with shape: {image_chunk.shape}")

    # Usage example: Training loop that uses the generator
    generator = data_generator(hdf5_files, batch_size=128)
    for batch_images, batch_tags in generator:
        # Use batch_images and batch_tags for model training or processing
        print(f"Batch shape: {batch_images.shape}")
