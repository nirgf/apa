#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 28 15:30:25 2024

@author: Arie Pyasik
"""
import numpy as np
from scipy.ndimage import rotate
import random
import tqdm as tq

def translate_binaty_image(matrix, seed):
    """
    Translates an image randomly inside a matrix without overriding non-zero values
    and without exceeding matrix boundaries.

    Args:
        matrix (np.ndarray): The target matrix where the image will be placed.
        image (np.ndarray): The image to translate inside the matrix.

    Returns:
        np.ndarray: The resulting matrix after placing the translated image.
    """
    random.seed(seed)
    image_idx = np.where(matrix != 0)
    image = matrix[min(image_idx[0]):max(image_idx[0])+1,\
                   min(image_idx[1]):max(image_idx[1])+1, :]
    

    matrix_height, matrix_width, mat_channel_num = matrix.shape
    image_height, image_width, im_channel_num = image.shape

    if image_height > matrix_height or image_width > matrix_width:
        raise ValueError("The image cannot be larger than the matrix.")

    # Find the range of valid top-left corners for placing the image
    max_row_offset = matrix_height - image_height
    max_col_offset = matrix_width - image_width

    # Randomly choose a valid top-left corner
    top_left_row = random.randint(0, max_row_offset)
    top_left_col = random.randint(0, max_col_offset)

    # Copy the matrix to avoid modifying the original
    result_matrix = np.zeros(matrix.shape)

    # Place the image inside the matrix without overriding non-zero values
    for i in range(image_height):
        for j in range(image_width):
            target_value = result_matrix[top_left_row + i, top_left_col + j]
            if all(target_value == 0):  # Place only if the position in the matrix is empty
                result_matrix[top_left_row + i, top_left_col + j] = image[i, j]

    return result_matrix


def add_images_with_override(image1, label1, image2, label2):
    """
    Combines two images and labels by overriding pixels from image2 onto image1.
    
    Parameters:
        image1 (numpy.ndarray): First image array (128x128x12).
        label1 (numpy.ndarray): Labels corresponding to image1 (128x128).
        image2 (numpy.ndarray): Second image array (128x128x12).
        label2 (numpy.ndarray): Labels corresponding to image2 (128x128).

    Returns:
        numpy.ndarray: New combined image.
        numpy.ndarray: New combined label.
    """
    combined_image = np.where(image2 != 0, image2, image1)
    combined_label = np.where(label2 != 0, label2, label1)
    return combined_image, combined_label

def apply_random_rotation(image, label, max_angle=180):
    """
    Applies random rotation to the image and label within the range [-max_angle, max_angle].

    Parameters:
        image (numpy.ndarray): Image array (128x128x12).
        label (numpy.ndarray): Label array (128x128).
        max_angle (int): Maximum rotation angle in degrees.

    Returns:
        numpy.ndarray: Rotated image.
        numpy.ndarray: Rotated label.
    """
    angle = random.uniform(-max_angle, max_angle)
    rotated_image = rotate(image, angle, axes=(0, 1), reshape=False, order=1, mode='constant', cval=0)
    rotated_label = rotate(label, angle, reshape=False, order=0, mode='constant', cval=0)
    return rotated_image, rotated_label

def add_random_noise(image, noise_level=0.01):
    """
    Adds random noise to the image.

    Parameters:
        image (numpy.ndarray): Image array (128x128x12).
        noise_level (float): Standard deviation of the Gaussian noise.

    Returns:
        numpy.ndarray: Image with added noise.
    """
    noise = np.random.normal(0, noise_level, image.shape)
    noisy_image = image + noise  # Ensuring pixel values remain in [0, 1]
    return noisy_image

def augment_dataset(dataset, labels, num_augmented_samples=100, aug_depth=2, add_noise = False):
    """
    Generates augmented samples by combining, rotating, and adding noise to images.

    Parameters:
        dataset (numpy.ndarray): Original dataset of shape (n, 128, 128, 12).
        labels (numpy.ndarray): Corresponding labels of shape (n, 128, 128).
        num_augmented_samples (int): Number of augmented samples to generate.
        aug_depth Number - of times to augment image
    Returns:
        list: List of augmented images.
        list: List of augmented labels.
    """
    aug_database = []
    aug_labels = []
    n = dataset.shape[0]
    
    for i in tq.tqdm(range(num_augmented_samples)):
        # Randomly select two images and their labels
        idx1, idx2 = np.random.choice(n, 2, replace=False)
        image1, label1 = dataset[idx1], labels[idx1]
        image2, label2 = dataset[idx2], labels[idx2]

        # Apply random rotation
        translated_image1 = translate_binaty_image(image1, seed = i)
        translated_label1 = translate_binaty_image(label1, seed = i)
        
        image1_rot, label1_rot = apply_random_rotation(translated_image1, translated_label1)

        # Combine images and labels
        combined_image, combined_label = add_images_with_override(image1_rot, label1_rot, image2, label2)

        # Apply random rotation
        rotated_image, rotated_label = apply_random_rotation(combined_image, combined_label)

        for aug_iter in range(aug_depth-1):

            new_image_idx = np.random.randint(n)
            image2, label2 = dataset[new_image_idx], labels[new_image_idx]

            # Translate
            translated_image2 = translate_binaty_image(image2, seed = aug_iter*5)
            translated_label2 = translate_binaty_image(label2, seed = aug_iter*5)

            # Combine images and labels
            combined_image, combined_label = add_images_with_override(rotated_image, rotated_label,\
                                                                      translated_image2, translated_label2)

            # Apply random rotation
            rotated_image, rotated_label = apply_random_rotation(combined_image, combined_label)

        # Add random noise
        if add_noise:
            noisy_image = add_random_noise(rotated_image)

        # Store the augmented sample
        aug_database.append(rotated_image)
        aug_labels.append(rotated_label)

    return aug_database, aug_labels

