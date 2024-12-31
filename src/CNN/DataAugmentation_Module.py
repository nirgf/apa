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

def apply_random_rotation(image, label, max_angle=30):
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

def augment_dataset(dataset, labels, num_augmented_samples=100, aug_depth=2):
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
    
    for _ in tq.tqdm(range(num_augmented_samples)):
        # Randomly select two images and their labels
        idx1, idx2 = np.random.choice(n, 2, replace=False)
        image1, label1 = dataset[idx1], labels[idx1]
        image2, label2 = dataset[idx2], labels[idx2]

        # Combine images and labels
        combined_image, combined_label = add_images_with_override(image1, label1, image2, label2)

        # Apply random rotation
        rotated_image, rotated_label = apply_random_rotation(combined_image, combined_label)

        for aug_iter in range(aug_depth-1):
            new_image_idx = np.random.randint(n)
            image2, label2 = dataset[new_image_idx], labels[new_image_idx]
            # Combine images and labels
            combined_image, combined_label = add_images_with_override(rotated_image, rotated_label, image2, label2)

            # Apply random rotation
            rotated_image, rotated_label = apply_random_rotation(combined_image, combined_label)

        # Add random noise
        # noisy_image = add_random_noise(rotated_image)

        # Store the augmented sample
        aug_database.append(rotated_image)
        aug_labels.append(rotated_label)

    return aug_database, aug_labels
