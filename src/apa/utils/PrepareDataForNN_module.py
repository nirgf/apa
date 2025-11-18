#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 14:26:39 2024

@author: Arie Pyasik
"""

import h5py
import numpy as np

def crop_image_to_segments(config,image,image_dim=12):
    """
    Crops a 3D image into segments of specified size with overlap.
    
    Parameters:
        image (np.ndarray): Input image of shape (m, n, 12).
        crop_size (int): Size of each crop along width and height (default is 64).
        overlap (float): Overlap fraction between segments (0 <= overlap < 1). Default is 0.1.
        image_dim (int) : last dimation of the image vector. is used for validation only
        
    Returns:
        np.ndarray: Cropped segments of shape (k, crop_size, crop_size, 12).
    """

    overlap=config['cnn_model']['overlap']
    crop_size=config['cnn_model']['crop_size']
    m, n, d = image.shape
    if d != image_dim:
        raise ValueError("Input image must have a third dimension of size " + str(image_dim))
    if not (0 <= overlap < 1):
        raise ValueError("Overlap must be between 0 and 1.")
    if crop_size > m or crop_size > n:
        raise ValueError("Crop size must be smaller than the image dimensions.")
    
    step = int(crop_size * (1 - overlap))  # Step size considering the overlap
    segments = []

    # Calculate ranges for cropping
    for i in range(0, m - crop_size + 1, step):
        for j in range(0, n - crop_size + 1, step):
            # Crop the current segment
            segment = image[i:i + crop_size, j:j + crop_size, :]
            segments.append(segment)

    # Handle edge case for the last rows/columns
    if (m - crop_size) % step != 0:
        for j in range(0, n - crop_size + 1, step):
            segment = image[-crop_size:, j:j + crop_size, :]
            segments.append(segment)
    if (n - crop_size) % step != 0:
        for i in range(0, m - crop_size + 1, step):
            segment = image[i:i + crop_size, -crop_size:, :]
            segments.append(segment)
    if (m - crop_size) % step != 0 and (n - crop_size) % step != 0:
        segment = image[-crop_size:, -crop_size:, :]
        segments.append(segment)

    # Convert list of segments to a numpy array
    return np.array(segments)

# Example usage:
# image = np.random.rand(200, 150, 12)  # Example image
# cropped_segments = crop_image_to_segments(image, crop_size=32, overlap=0.2)
# print(cropped_segments.shape)  # Shape will be (k, crop_size, crop_size, 12)


def save_cropped_segments_to_h5(cropped_segments, file_name):
    """
    Saves cropped segments to an HDF5 file.
    
    Parameters:
        cropped_segments (np.ndarray): A numpy array of shape (k, crop_size, crop_size, 12).
        file_name (str): Name of the HDF5 file to save the data (e.g., "cropped_segments.h5").
        
    Returns:
        None
    """
    if not isinstance(cropped_segments, np.ndarray):
        raise ValueError("Cropped segments must be a numpy array.")
    
    with h5py.File(file_name, 'w') as h5_file:
        h5_file.create_dataset("cropped_segments", data=cropped_segments)
        print(f"Cropped segments successfully saved to {file_name}")
        


def normalize_mask(mask, crop_size):
    binary_mask = np.zeros(mask[:, :, 0].shape)
    binary_mask[mask[:, :, 0] != 0] = 1
    num_of_channels = mask.shape[-1]
    
    CNN_input = np.zeros([crop_size, crop_size])
    input_height, input_width = CNN_input.shape
    if all(np.asarray(np.shape(CNN_input)) > np.asarray(binary_mask.shape)):  

        # Fill all the channels (not only binary)         
        CNN_input = np.reshape(CNN_input, list(CNN_input.shape) + [1])
        CNN_input = np.repeat(CNN_input, num_of_channels, -1)
        CNN_input[np.where(mask != 0)] = mask[mask != 0]
        
    else:
        
        # Deal with the part in the frame
        val_idx = np.asarray(np.where(binary_mask > 0))
        in_frame_idx = (val_idx[0] < input_height) & (val_idx[1] < input_width)
        val_idx_in_frame = val_idx[:, in_frame_idx]
        
        # Wrap the part outside the frame
        out_frame_idx = val_idx[:, np.where((val_idx[0] > input_height) | (val_idx[1] > input_width))]
        out_frame_idx = out_frame_idx[:, 0, :] # sort out the 'where' thing
    
        og_out_frame_idx = np.zeros(out_frame_idx.shape).astype(int)
        og_out_frame_idx[:, :] = out_frame_idx[:, :].astype('int') # to be used on the diagonal array
        idx_to_correct = (out_frame_idx != np.mod(out_frame_idx, crop_size))
        out_frame_idx[idx_to_correct] = crop_size - (np.mod(out_frame_idx[idx_to_correct], crop_size) + 1)
        
        # Fill non-zero pixels only
        empty_out_idx = np.where(CNN_input[out_frame_idx[0], out_frame_idx[1]] == 0)
        out_frame_idx = [out_frame_idx[0][empty_out_idx], out_frame_idx[1][empty_out_idx]]
        og_out_frame_idx = [og_out_frame_idx[0][empty_out_idx], og_out_frame_idx[1][empty_out_idx]]
        
        # Fill all the channels (not only binary)
        CNN_input = np.reshape(CNN_input, list(CNN_input.shape) + [1])
        CNN_input = np.repeat(CNN_input, num_of_channels, -1)
        
        # Put the correct values        
        CNN_input[val_idx_in_frame[0], val_idx_in_frame[1], :] = mask[val_idx_in_frame[0], val_idx_in_frame[1], :]
        CNN_input[out_frame_idx[0], out_frame_idx[1], :] = mask[og_out_frame_idx[0], og_out_frame_idx[1], :]
        
    return CNN_input

