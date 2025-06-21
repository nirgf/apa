#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 12 16:31:33 2025

@author: root
"""

import numpy as np
import matplotlib.pyplot as plt
from skimage import draw
from skimage.util import img_as_float
from tqdm import tqdm as tq
import src.utils.point_cloud_utils as pc_utils


# Include the function from earlier
@pc_utils.log_execution_time
def find_local_road_offset_from_arrays(ideal_bin, detected_bin, max_shift=20, debug=False):
    h, w = ideal_bin.shape
    if detected_bin.shape != (h, w):
        raise ValueError("Images must be the same size for local correlation.")

    max_corr = -np.inf
    best_offset = (0, 0)

    for dy in tq(range(-max_shift, max_shift + 1), desc='Finding best offset: '):
        for dx in range(-max_shift, max_shift + 1):
            shifted = np.roll(np.roll(detected_bin, dy, axis=0), dx, axis=1)

            if dy > 0:
                shifted[:dy, :] = 0
            elif dy < 0:
                shifted[dy:, :] = 0
            if dx > 0:
                shifted[:, :dx] = 0
            elif dx < 0:
                shifted[:, dx:] = 0

            corr = np.sum(ideal_bin & shifted)

            if corr > max_corr:
                max_corr = corr
                best_offset = (dy, dx)

    if debug:
        print(f"Best offset (dy, dx): {best_offset} with correlation score: {max_corr}")
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 3, 1)
        plt.imshow(ideal_bin)
        plt.title("Ideal Road Map")

        plt.subplot(1, 3, 2)
        plt.imshow(detected_bin)
        plt.title("Shifted Detected Map")

        plt.subplot(1, 3, 3)
        aligned = np.roll(np.roll(detected_bin, best_offset[0], axis=0), best_offset[1], axis=1)
        plt.imshow(aligned)
        plt.title("Aligned Detected Map")
        plt.tight_layout()
        plt.show()
    
    print(f"Estimated Offset: {best_offset} [pixels]")
    return best_offset

# Example of usage
# Step 1: Create toy ideal road mask
# def create_synthetic_road_image(shape=(2000, 5000)):
#     img = np.zeros(shape, dtype=bool)
#     for i in range(100):
#         rr, cc = draw.line(int(2000*np.random.rand()), \
#                            int(5000*np.random.rand()),\
#                            int(2000*np.random.rand()), \
#                            int(5000*np.random.rand()))  # horizontal line
#         img[rr, cc] = True
#     return img

# # Step 2: Apply a known shift
# true_offset = (5, -19)  # dy, dx
# ideal = create_synthetic_road_image()
# detected = np.roll(np.roll(ideal, true_offset[0], axis=0), true_offset[1], axis=1)

# # Mask out the wraparound
# if true_offset[0] > 0:
#     detected[:true_offset[0], :] = 0
# elif true_offset[0] < 0:
#     detected[true_offset[0]:, :] = 0
# if true_offset[1] > 0:
#     detected[:, :true_offset[1]] = 0
# elif true_offset[1] < 0:
#     detected[:, true_offset[1]:] = 0

# # Step 3: Estimate the offset
# estimated_offset = find_local_road_offset_from_arrays(ideal, detected, max_shift=20, debug=True)
# print(f"True Offset: {true_offset}, Estimated Offset: {estimated_offset}")