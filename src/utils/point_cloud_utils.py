import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev,griddata
from skimage.draw import line, polygon
from scipy.spatial import Delaunay
from scipy.spatial.distance import pdist, squareform
from scipy.sparse.csgraph import minimum_spanning_tree
from shapely.geometry import LineString, Point
from scipy.spatial.distance import cdist
from skimage.morphology import disk, square,skeletonize
from scipy.ndimage import binary_dilation,binary_erosion,binary_opening,binary_closing,distance_transform_edt, label,generate_binary_structure
from scipy.sparse import csr_matrix, save_npz, load_npz
from scipy.spatial import distance
from collections import defaultdict, deque
from scipy.ndimage import map_coordinates
from skimage.measure import regionprops, regionprops_table
from skimage.morphology import disk
import os
from scipy import stats
import cv2
from scipy.signal import convolve2d,convolve
import time
import functools
from pathlib import Path
from src.geo_reference import GetRoadsCoordinates
from sklearn.decomposition import PCA
import json
import src.utils.pc_plot_utils as plt_utils
# ANSI escape codes for purple text with a black background
PURPLE_ON_BLACK = "\033[45;30m"
RESET = "\033[0m"

PIXEL_SZ= 4 # pixel size in meter
REPO_ROOT=os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
def log_execution_time(func):
    """
    A decorator that logs the function being called and its execution time.
    Logs are printed in purple with a black background.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        print(f"{PURPLE_ON_BLACK}Function '{func.__name__}' was called {RESET}")
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time

        # Log function name and execution time in purple on black background
        print(f"{PURPLE_ON_BLACK}Function '{func.__name__}' took {execution_time:.4f} seconds{RESET}")

        return result

    return wrapper



def nan_arr(arr):
    arr[arr <= 0] = np.nan
    return arr


def or_nan(x1, x2):
    # Apply "or" operation between the arrays
    result = np.where(np.isnan(x1) & np.isnan(x2), np.nan,  # If both are NaN, keep NaN
                      np.where(np.isnan(x1), x2,  # If only x1 is NaN, use x2
                               np.where(np.isnan(x2), x1,  # If only x2 is NaN, use x1
                                        x1)))  # If neither is NaN, use x1
    return result

# Example usage


# Function to categorize a segment based on the closest point's value
def get_category(x, y, points, PCI):
    distances = np.sqrt((points[:, 0] - x) ** 2 + (points[:, 1] - y) ** 2)
    # print(distances)
    closest_index = np.argmin(distances)
    # print(closest_index)
    value = PCI.ravel()[closest_index]

    if value < 30:
        return "below_30"
    elif 30 <= value <= 70:
        return "between_30_and_70"
    else:
        return "above_70"

def reorder_points_greedy(points,**kwargs):
    """
    Reorder points to form a continuous path by connecting each point to its nearest neighbor.
    Parameters:
    - points: numpy array of shape (n, 2), representing (x, y) coordinates.
    Returns:
    - ordered_points: numpy array of reordered points (x, y).
    """
    n_points = points.shape[0]
    visited = np.zeros(n_points, dtype=bool)

    start_point = kwargs.get('start_point', None)
    print(f'start_point greedy merege:{start_point}')

    #TODO: support first point robustness as it's affect the trajectory

    #Assume min is eastmost_x,nortmost_y for ISR

    westmost_x,southmost_y = np.argmin(points, axis=0)
    points[np.argmin(points, axis=0)[1]]
    if start_point=='westmost':
        ordered_points = [points[westmost_x]]
    elif start_point=='northmost':
        ordered_points = [points[southmost_y]]
    else:
        ordered_points = [points[0]]
    visited[0] = True

    for _ in range(n_points - 1):
        last_point = ordered_points[-1]

        # Calculate distances from the last point to all unvisited points
        distances = cdist([last_point], points[~visited])[0]

        # Find the index of the nearest unvisited point
        nearest_index = np.argmin(distances)

        # Get the index of the nearest unvisited point in the original array
        true_nearest_index = np.where(~visited)[0][nearest_index]

        # Add the nearest point to the ordered list and mark it as visited
        ordered_points.append(points[true_nearest_index])
        visited[true_nearest_index] = True

    return np.array(ordered_points)

@log_execution_time
def fit_spline_pc(points,**kwargs):
    """
    Reorder the points using a greedy nearest-neighbor algorithm and fit a spline through them.
    Parameters:
    - points: numpy array of shape (n, 2), representing (x, y) coordinates.
    Returns:
    - x_new, y_new, line_string: Spline-fitted points and the corresponding LineString.
    """
    spline_res = kwargs.get('spline_res', 100)
    start_point_indx = kwargs.get('start_point_indx', None)
    # Reorder points using the greedy nearest-neighbor algorithm
    reordered_points = reorder_points_greedy(points,start_point=start_point_indx)

    # Fit a spline through the reordered points
    tck, uk = splprep(reordered_points.T, s=0,per=0)


    x_new, y_new = splev(np.linspace(0, 1, spline_res), tck)

    # Create a Shapely LineString from the spline points
    line_string = LineString(np.c_[x_new, y_new])

    return x_new, y_new, line_string
# -------------------------------
# Continuesly connect Close Points
# -------------------------------

import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.csgraph import dijkstra
from scipy.sparse import csr_matrix
import tqdm

def get_nearest_road_point(point_lat, point_lon, coinciding_mask, X_cropped, Y_cropped):
    XY_ind = np.unravel_index(np.argmin(np.square(X_cropped - point_lat)\
                              + np.square(Y_cropped - point_lon))
                              , np.shape(X_cropped))
    temp_distance, nearest_indices = distance_transform_edt(~coinciding_mask, return_indices=True)
    coinciding_true_index = tuple(nearest_indices[:, XY_ind[0], XY_ind[1]])
    return coinciding_true_index

def is_roi_within_bounds(small_roi, large_roi):
    """
    Check if the four corners of a smaller ROI are within a larger ROI.
    """
    # Unpack coordinates
    x_min_small,x_max_small, y_min_small, y_max_small = small_roi
    x_min_large,x_max_large, y_min_large, y_max_large = large_roi
    # Define the four corners of the smaller ROI
    corners = [
        (x_min_small, y_min_small),  # Top-left
        (x_max_small, y_min_small),  # Top-right
        (x_max_small, y_max_small),  # Bottom-right
        (x_min_small, y_max_small)   # Bottom-left
    ]
    # Check if all corners are within the bounds of the larger ROI
    for x, y in corners:
        if not (x_min_large <= x <= x_max_large and y_min_large <= y <= y_max_large):
            print('Misbound')
            return False
    return True


@log_execution_time
def merge_points_dijkstra(npz_filename,X_cropped, Y_cropped,coinciding_mask, points_PCI, ROI_seg):
    npz_file_path = Path(npz_filename)
    PCI_segID_LUT_filename=npz_file_path.with_suffix(".json")
    if os.path.exists(npz_filename) and  os.path.exists(PCI_segID_LUT_filename):
        print(f"File '{npz_filename}' exists. Loading data...")
        # Load from JSON
        with open(PCI_segID_LUT_filename, 'r') as f:
            PCI_segID_LUT = json.load(f)

        return load_npz(npz_filename).toarray(),PCI_segID_LUT

    else:
        print(f"File '{npz_filename}' does not exist. Creating data...")

        ROI_seg = ROI_seg # Covert to list
        PCI_mask = np.zeros(np.shape(coinciding_mask))
        segID_mask = np.zeros(np.shape(coinciding_mask))
        PCI_segID_LUT= {}
        for point_idx in tqdm.tqdm(range(len(points_PCI)-1), desc = 'Filling PCI Data in Road Matrix'):

            point_lat = points_PCI[point_idx, 0]
            point_lon = points_PCI[point_idx, 1]

            coinciding_true_index = get_nearest_road_point(point_lat, point_lon, coinciding_mask, X_cropped, Y_cropped)
            PCI_mask[coinciding_true_index] =  points_PCI[point_idx, 2]

            if ROI_seg[point_idx] == ROI_seg[point_idx + 1]:
                point_lat_nxtPnt = points_PCI[point_idx+1, 0]
                point_lon_nxtPnt = points_PCI[point_idx+1, 1]

                coinciding_true_index_nxtPnt = get_nearest_road_point(point_lat_nxtPnt, point_lon_nxtPnt, coinciding_mask, X_cropped, Y_cropped)

                min_x_idx_mini_roi = min(coinciding_true_index[0], coinciding_true_index_nxtPnt[0])
                max_x_idx_mini_roi = max(coinciding_true_index[0], coinciding_true_index_nxtPnt[0])

                min_y_idx_mini_roi = min(coinciding_true_index[1], coinciding_true_index_nxtPnt[1])
                max_y_idx_mini_roi = max(coinciding_true_index[1], coinciding_true_index_nxtPnt[1])

                cropped_start = [coinciding_true_index[0] - min_x_idx_mini_roi,\
                                 coinciding_true_index[1] - min_y_idx_mini_roi]
                cropped_end = [coinciding_true_index_nxtPnt[0] - min_x_idx_mini_roi,\
                                 coinciding_true_index_nxtPnt[1] - min_y_idx_mini_roi]


                cropped_mask = coinciding_mask[min_x_idx_mini_roi : max_x_idx_mini_roi + 1,\
                                               min_y_idx_mini_roi : max_y_idx_mini_roi + 1]
                ## Get the path using dijkstra
                path = dijkstra_vectorized(cropped_mask, cropped_start, cropped_end)
                path = np.asarray(path)
                full_path = np.asarray([path[:, 0] + min_x_idx_mini_roi,\
                                        path[:, 1] + min_y_idx_mini_roi])
                PCI_mask[full_path[0, :], full_path[1, :]] = points_PCI[point_idx, 2]
                segID_mask[full_path[0, :], full_path[1, :]] = ROI_seg[point_idx]
                PCI_segID_LUT[str(ROI_seg[point_idx])]=points_PCI[point_idx, 2]


        npz_filename=os.path.join(REPO_ROOT,'data/Detroit/masks_OpenStreetMap/Detroit_dijkstra_roads_mask.npz')
        PCI_segID_LUT_filename=npz_file_path.with_suffix(".json")

        save_npz(npz_filename, csr_matrix(segID_mask))
        with open(PCI_segID_LUT_filename, 'w') as f:
            json.dump(PCI_segID_LUT, f)
        print(f"Saved compressed binary mask of OpenStreetMap roads into '{npz_filename}'.")
        with open(PCI_segID_LUT_filename, 'r') as f:
            PCI_segID_LUT = json.load(f)
        return load_npz(npz_filename).toarray(),PCI_segID_LUT

        


# Vectorized Dijkstra's algorithm using scipy, with diagonal moves
def dijkstra_vectorized(mask, start, stop):
    rows, cols = mask.shape

    # Create a sparse adjacency matrix for the grid
    num_nodes = rows * cols
    graph = np.zeros((num_nodes, num_nodes))

    # Directions: Up, Down, Left, Right, and the 4 diagonals
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1),  # Up, Down, Left, Right
                  (-1, -1), (-1, 1), (1, -1), (1, 1)]  # Diagonal directions

    # Map (row, col) coordinates to node indices
    def to_index(r, c):
        return r * cols + c

    # Construct the graph
    for r in range(rows):
        for c in range(cols):
            if mask[r, c] == 1:  # Only valid nodes
                node_index = to_index(r, c)
                for dr, dc in directions:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < rows and 0 <= nc < cols and mask[nr, nc] == 1:
                        neighbor_index = to_index(nr, nc)
                        graph[node_index, neighbor_index] = 1  # Add an edge with weight 1

    # Convert the graph to a sparse matrix
    graph_sparse = csr_matrix(graph)

    # Perform Dijkstra's algorithm
    start_idx = to_index(start[0], start[1])
    stop_idx = to_index(stop[0], stop[1])

    # Find the shortest path using Dijkstra
    dist, pred = dijkstra(graph_sparse, indices=start_idx, return_predecessors=True)

    # Reconstruct the path
    path = []
    current = stop_idx
    while current != start_idx and current != -9999:  # -9999 is the sentinel for no path
        path.append(current)
        current = pred[current]
    if current == start_idx:
        path.append(start_idx)

    # Convert path indices back to (row, col)
    path = [(i // cols, i % cols) for i in path[::-1]]  # Reverse the path

    return path

# -------------------------------
# Merge Close Points
# -------------------------------
def merge_close_points(points,PCI, threshold=0.551,**kwargs):
    # np.concatenate([points,PCI.reshape(-1, 1)],axis=1)
    points=np.c_[points,PCI]
    merged_points = []
    skip_indices = set()
    merge_method = kwargs.get('merge_method', 'mean_min')
    print(f'merge_method:__{merge_method}__\n\n')
    for i in range(len(points)):
        if i in skip_indices:
            continue
        close_indices = np.where(cdist([points[i, :2]], points[:, :2])[0] < threshold)[0]
        if merge_method == 'first_location':
            merged_x = points[close_indices[0], 0]
            merged_y = points[close_indices[0], 1]
            merged_value = points[close_indices[0], 2]  # Get  value
        elif merge_method == 'middle_location':
            merged_x = points[close_indices[(len(close_indices)-1)//2], 0]
            merged_y = points[close_indices[(len(close_indices)-1)//2], 1]
            merged_value = points[close_indices[(len(close_indices)-1)//2], 2]  # Get  value
        elif merge_method == 'min_min':
            merged_x = np.min(points[close_indices, 0])
            merged_y = np.min(points[close_indices, 1])
            merged_value = np.min(points[close_indices, 2])  # Take minimum value
        else:
            merged_x = np.mean(points[close_indices, 0])
            merged_y = np.mean(points[close_indices, 1])
            merged_value = np.min(points[close_indices, 2])  # Take minimum value
        merged_points.append([merged_x, merged_y, merged_value])
        skip_indices.update(close_indices)

    return np.array(merged_points)


def fill_mask_with_irregular_spline(xy_points, X_grid, Y_grid, binary_mask,combine_mask=False):
    """
    Fill a mask based on a LineString and point cloud values, where points are mapped to a grid.
    Parameters:
    - line_string: shapely.geometry.LineString representing the allowed pixel path.
    - points: numpy array of shape (n, 3), where columns represent (x, y, value).
    - X_grid, Y_grid: 2D numpy arrays representing the grid coordinates for pcolormesh plotting.
    - mask_shape: tuple (height, width) defining the shape of the output mask.
    - radius: float. Maximum distance for assigning point values to pixels.

    Returns:
    - mask_grid: A 2D numpy array of the same shape as X_grid and Y_grid filled with the nearest point values.
    """
    x_new, y_new, line_string = fit_spline_pc(xy_points,start_point_indx=None)
    # Initialize the mask with NaN to indicate unfilled pixels
    mask_shape=binary_mask.shape[:2]
    updated_mask = np.zeros_like(binary_mask)
    # updated_mask = np.full(mask_shape, np.nan, dtype=float)

    # Extract x and y coordinates from the LineString
    line_coords = np.array(line_string.coords)

    #  #Use griddata to interpolate the point cloud values onto the mask grid
    #  #We use the points[:, 0] for x, points[:, 1] for y, and points[:, 2] for values
    # grid_values = griddata(xy_points[:, 0], xy_points[:, 1], (X_grid, Y_grid), method='nearest')
    # grid_values = griddata(xy_points, np.ones(len(xy_points)), (X_grid, Y_grid), method='nearest')
    # mask_grid = np.zeros(mask_shape, dtype=bool)
    # mask_grid[~np.isnan(grid_values)] = True

    for i in range(len(line_coords) - 1):
        # Now, update the mask along the LineString within the given radius
        x0, y0 = line_coords[i]
        x1, y1 = line_coords[i + 1]

        # Find the nearest pixel coordinates in the grid
        # Convert spline points to grid indices by finding the closest X, Y grid indices
        ix0, iy0 = np.argmin(np.abs(X_grid[0, :] - x0)), np.argmin(np.abs(Y_grid[:, 0] - y0))
        ix1, iy1 = np.argmin(np.abs(X_grid[0, :] - x1)), np.argmin(np.abs(Y_grid[:, 0] - y1))

        # Draw a line between the two points in the mask
        rr, cc = line(iy0, ix0, iy1, ix1)
        updated_mask[rr, cc] = 1

    if combine_mask:
        # Combine the original binary mask with the updated mask
        extended_mask = np.maximum(binary_mask, updated_mask)
    else:
        extended_mask = updated_mask

    return extended_mask,line_string, (x_new, y_new)






def fill_mask_with_spline(binary_mask,xy_points,combine_mask=False):
    x_new, y_new, line_string = fit_spline_pc(xy_points)
    # Initialize an empty mask to track pixels touched by the spline
    updated_mask = np.zeros_like(binary_mask)
    # Iterate over the spline points and draw line segments
    for i in range(len(x_new) - 1):
        # Get the integer coordinates for the start and end of each segment
        y0, x0 = int(round(x_new[i])), int(round(y_new[i]))
        y1, x1 = int(round(x_new[i + 1])), int(round(y_new[i + 1]))

        # Use Bresenham's line algorithm to get the pixels between the points
        rr, cc = line(x0, y0, x1, y1)

        # Set these pixels to 1 in the updated mask
        updated_mask[rr, cc] = 1
    if combine_mask:
        # Combine the original binary mask with the updated mask
        extended_mask = np.maximum(binary_mask, updated_mask)
    else:
        extended_mask = updated_mask

    return extended_mask,line_string



# -------------------------------
# Step 3: Segment the Spline by 7-Unit Lengths
# -------------------------------
def get_segment_mask(center, radius=3.5, size=10):
    """Return a binary mask for a segment within the given radius."""
    mask = np.zeros((size, size), dtype=int)
    rr, cc = np.ogrid[:size, :size]
    dist = np.sqrt((rr - center[0]) ** 2 + (cc - center[1]) ** 2)
    mask[dist <= radius] = 1
    return mask

@log_execution_time
def fill_mask_with_line_point_values(line_string, points_xy_values, mask_shape, radius=3.5):
    """
    Fill the pixels along a LineString with the value of the closest point in the point cloud,
    but only if the closest point is within the given radius.

    Parameters:
    - line_string: shapely.geometry.LineString representing the allowed pixel path.
    - points: numpy array of shape (n, 3), where columns represent (x, y, value).
    - mask_shape: tuple (height, width) defining the shape of the output mask.
    - radius: float. Maximum distance for assigning point values to pixels.

    Returns:
    - segment_mask: numpy array with the same shape as mask_shape,
                    containing the assigned values from the point cloud.
    """
    # Initialize the mask with NaN to indicate unfilled pixels
    segment_mask = np.full(mask_shape, np.nan, dtype=float)

    # Convert LineString to integer pixel coordinates
    line_coords = np.array(list(line_string.coords))
    for i in range(len(line_coords) - 1):
        # Get integer coordinates for each segment using Bresenham's line algorithm
        rr, cc = line(int(round(line_coords[i][1])), int(round(line_coords[i][0])),
                      int(round(line_coords[i + 1][1])), int(round(line_coords[i + 1][0])))

        # Flatten the pixel coordinates into (N, 2) for distance computation
        pixel_coords = np.c_[cc, rr]

        # Compute distances between the pixel coordinates and the point cloud (x, y)
        distances = cdist(pixel_coords, points_xy_values[:, :2])

        # Find the closest point index for each pixel
        closest_indices = np.argmin(distances, axis=1)
        closest_distances = np.min(distances, axis=1)

        # Assign values only if the closest point is within the given radius
        for j, (r, c) in enumerate(zip(rr, cc)):
            if closest_distances[j] <= radius:
                segment_mask[r, c] = points_xy_values[closest_indices[j], 2]

    return segment_mask


def fill_mask_with_line_point_values_irregular(line_string, points_xy_values, mask_shape, radius=3.5):
    pass


def divide_array(array, *args):
    # Use default range values if *args is empty
    if not args:
        args = (30, 30, 70, 85)

    # Check if *args has valid length (either 4 or 6)
    if len(args) == 4:
        arg1, arg2, arg3, arg4 = args
        # Define conditions based on default or user-defined ranges
        below_1 = array < arg1
        between_1 = (array >= arg2) & (array <= arg3)
        above_2 = array > arg4
        return below_1, between_1, above_2

    elif len(args) == 6:
        arg1, arg2, arg3, arg4, arg5, arg6 = args
        # Define conditions based on 6 argument ranges
        below_1 = array < arg1
        between_1 = (array >= arg2) & (array <= arg3)
        between_2 = (array > arg4) & (array <= arg5)
        above_3 = array > arg6
        return below_1, between_1, between_2, above_3

    else:
        raise ValueError("The function requires either 4 or 6 arguments.")


def create_masks(segmented_image):
    """
    Create masks based on the segmented image for three categories:
    below 30, between 30 and 70, and above 85.

    Parameters:
    - segmented_image: 2D numpy array containing label values from 0 to 100.

    Returns:
    - mask_below_30: Boolean mask for labels < 30.
    - mask_30_to_70: Boolean mask for labels between 30 and 70.
    - mask_above_85: Boolean mask for labels > 85.
    """
    mask_below_30 = segmented_image < 40
    mask_30_to_70 = (segmented_image >= 40) & (segmented_image <= 70)
    mask_above_85 = segmented_image > 85

    return mask_below_30, mask_30_to_70, mask_above_85


def dilate_mask(mask, dilation_pixels=3):
    """
    Dilate the binary mask by a specified number of pixels.

    Parameters:
    - mask: 2D binary mask to be dilated.
    - dilation_pixels: Number of pixels to dilate the mask by.

    Returns:
    - dilated_mask: The dilated binary mask.
    """
    # Define the structure for dilation (square of size (2*dilation_pixels+1))
    structure = np.ones((2 * dilation_pixels + 1, 2 * dilation_pixels + 1), dtype=np.uint8)

    # Perform binary dilation on the mask
    dilated_mask = binary_dilation(mask, structure=structure).astype(np.uint8)

    return dilated_mask


def apply_masks_and_average(image, mask, debug_plots=False, **kwargs):
    """
    Apply a mask to the multi-channel image and compute the average pixel values for each channel.

    Parameters:
    - image: 3D numpy array representing the image (height, width, 12).
    - mask: 2D boolean numpy array representing the mask to be applied.

    Returns:
    - channel_values: A list containing the averaged pixel values for each channel.
    """
    channels=image.shape[2]
    # Apply the mask and extract the corresponding pixel values for each channel
    channels_values = [image[:, :, channel][mask] for channel in range(channels)]
    print(channels_values)
    print(len(channels_values))
    if debug_plots:
        # Compute layout for subplots
        rows = int(np.ceil(np.sqrt(channels)))
        cols = int(np.ceil(channels / rows))
        # Prepare the plot
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3))
        axes = axes.flatten()
        for i in range(channels):
            ax = axes[i]
            channel_data = channels_values[i]
            mean = np.nanmean(channel_data)
            median = np.nanmedian(channel_data)
            ax.hist(channels_values[i], bins=50, color='gray', edgecolor='black', alpha=0.7)
            ax.set_title(f"Channel {i + 1}")
            ax.set_xlabel("Pixel Value")
            ax.set_ylabel("Frequency")
            # Annotate mean and median
            ax.axvline(mean, color='blue', linestyle='dashed', linewidth=1, label=f"Mean: {mean:.2f}")
            ax.axvline(median, color='red', linestyle='solid', linewidth=1, label=f"Median: {median:.2f}")
            ax.legend()

        # Hide unused subplots if any
        for j in range(channels, len(axes)):
            axes[j].axis('off')
        # Overall plot title
        suptitle_kwargs = kwargs.get('suptitle', "Pixel Value Histograms for Image Channels")
        plt.suptitle(suptitle_kwargs)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()

    return channels_values


def plot_normalized_histograms(channel_values, title):
    """
    Plot the normalized histograms for each channel.

    Parameters:
    - channel_values: A list of arrays of pixel values for the respective channels.
    - title: Title of the plot.
    """
    plt.figure(figsize=(10, 8))

    # Plot normalized histograms (probabilities) for each channel
    for i, values in enumerate(channel_values):
        plt.hist(values, bins=50, alpha=0.5, density=True, label=f'Channel {i + 1}')

    plt.title(f"Normalized Histogram for {title}")
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Probability")
    plt.legend()
    plt.show()

def get_stats_from_segment_spectral(segmented_pci_spectral):
    # Calculate the mean and standard deviation

    # Copy the data to avoid modifying the original array
    data = np.copy(segmented_pci_spectral)

    # Number of data points
    spectral_bands, n_points = data.shape

    # Compute statistics
    mean = np.nanmean(data, axis=1)  # Mean ignoring NaNs
    std_dev = np.nanstd(data, ddof=1, axis=1)  # Sample standard deviation (ddof=1)
    sem = std_dev / np.sqrt(n_points)  # Standard Error of the Mean (SEM)
    median = np.nanmedian(data, axis=1)  # Median ignoring NaNs
    iqr = stats.iqr(data, axis=1, nan_policy='omit')  # Interquartile range (IQR)
    min_val = np.nanmin(data, axis=1)  # Minimum value ignoring NaNs
    max_val = np.nanmax(data, axis=1)  # Maximum value ignoring NaNs

    # Calculate the margin of error for the mean (95% confidence interval)
    confidence_level = 0.95
    t_critical = stats.t.ppf((1 + confidence_level) / 2, df=n_points - 1)
    margin_of_error = t_critical * sem

    # Compile results into a dictionary
    results = {
        "N_points": n_points,
        "Mean": mean,
        "Median": median,
        "Std_dev": std_dev,
        "SEM": sem,
        "Margin_of_error": margin_of_error,
        "IQR": iqr,
        "Min": min_val,
        "Max": max_val,
    }

    return results


def laplacian_of_gaussian(image, sigma):
    """Computes the Laplacian of Gaussian of an image.

    Args:
      image: The image to be processed.
      sigma: The sigma of the Gaussian kernel.

    Returns:
      The Laplacian of Gaussian of the image.
    """

    kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
    blurred_image = cv2.GaussianBlur(image, (5, 5), sigma)
    laplacian_image = cv2.filter2D(blurred_image, -1, kernel)
    return laplacian_image

#%% Get only pixels that intersect with roads
@log_execution_time
def get_pixels_intersect_with_roads(lon_mat,lat_mat,lon_range,lat_range):
    roads_gdf = GetRoadsCoordinates.get_road_mask(lat_range, lon_range)
    # boolean mask for VENUS Data
    coinciding_mask = GetRoadsCoordinates.get_coinciding_mask(roads_gdf, lon_mat, lat_mat)
    return coinciding_mask

def normalize_hypersepctral_bands(hys_img):
    # this part handle all bands
    hys_img_norm = np.zeros_like(hys_img)
    for kk in range(hys_img.shape[-1]):
        hys_img_1chn = hys_img[:, :, kk]
        hys_img_1chn = hys_img_1chn / np.nanmax(hys_img_1chn)
        hys_img_1chn[hys_img_1chn <= 0] = np.nan
        hys_img_norm[:, :, kk] = hys_img_1chn

    return hys_img_norm



def normalize_img(img):
    # Normalize each channel to [0, 1] for display
    return (img - np.min(img)) / (np.max(img) - np.min(img))


def false_color_hyperspectral_image(hys_img):
    # Reshape hyperspectral data for PCA (flatten spatial dimensions)
    pixels = hys_img.reshape(-1, hys_img.shape[2])  # Shape: (npixels, nbands)
    pixels[np.isnan(pixels)] = 0
    # Apply PCA to reduce to 3 components
    pca = PCA(n_components=3)
    pca_result = pca.fit_transform(pixels)  # Shape: (npixels, 3)
    # Reshape the result back to 2D image with 3 channels
    pca_image = pca_result.reshape(hys_img.shape[0], hys_img.shape[1], 3)


    # Normalize each PCA component for display
    red = normalize_img(pca_image[:, :, 1])
    green = normalize_img(pca_image[:, :, 0])
    blue = normalize_img(pca_image[:, :, 2])

    # Stack the normalized components to create an RGB image
    false_color_image = np.stack((red, green, blue), axis=-1)
    return false_color_image


def grow_mask_along_line(mask_shape, line_coords, parallel_width=10, perpendicular_width=2):
    """
    Grows a binary mask around a line in specific parallel and perpendicular directions.

    Parameters:
    - mask_shape: Tuple representing the shape of the binary mask (height, width).
    - line_coords: List of tuples [(x1, y1), (x2, y2), ...] representing the line's pixel coordinates.
    - parallel_width: Number of pixels to grow along the line's direction.
    - perpendicular_width: Number of pixels to grow perpendicular to the line's direction.

    Returns:
    - A binary mask of the same shape as `mask_shape`, grown around the input line.
    """
    mask = np.zeros(mask_shape, dtype=bool)

    for i in range(len(line_coords) - 1):
        # Current segment of the line
        x1, y1 = line_coords[i]
        x2, y2 = line_coords[i + 1]

        # Compute tangent vector and normalize
        dx, dy = x2 - x1, y2 - y1
        length = np.sqrt(dx ** 2 + dy ** 2)
        tangent = (dx / length, dy / length)

        # Compute perpendicular vector
        perpendicular = (-tangent[1], tangent[0])  # Rotate tangent by 90 degrees

        # Create a rectangular region around the segment
        for offset_parallel in range(-parallel_width, parallel_width + 1):
            for offset_perpendicular in range(-perpendicular_width, perpendicular_width + 1):
                # Shift along parallel and perpendicular directions
                x_shift = int(x1 + offset_parallel * tangent[0] + offset_perpendicular * perpendicular[0])
                y_shift = int(y1 + offset_parallel * tangent[1] + offset_perpendicular * perpendicular[1])

                # Add to the mask if within bounds
                if 0 <= x_shift < mask_shape[0] and 0 <= y_shift < mask_shape[1]:
                    mask[x_shift, y_shift] = True

    return mask

def morphological_operator(binary_image, operation, structuring_element='disk', radius_or_size=3):
    """
    Applies a general morphological operation to a binary image.

    Parameters:
    - binary_image: ndarray
        A binary image (2D array with True/False or 1/0 values).
    - operation: str
        The morphological operation to apply. Options are:
        - 'dilation', 'erosion', 'opening', 'closing'.
    - structuring_element: str or ndarray
        The type of structuring element to use. Options are:
        - 'disk', 'square', or a custom 2D ndarray.
    - radius_or_size: int
        The radius (for disk) or size (for square) of the structuring element.

    Returns:
    - result: ndarray
        The binary image after applying the specified morphological operation.
    """
    # Create the structuring element
    if isinstance(structuring_element, str):
        if structuring_element == 'disk':
            selem = disk(radius_or_size)
        elif structuring_element == 'square':
            selem = square(radius_or_size)
        else:
            raise ValueError("Unsupported structuring element. Use 'disk', 'square', or provide a custom ndarray.")
    elif isinstance(structuring_element, np.ndarray):
        selem = structuring_element
    else:
        raise ValueError("Invalid structuring element format. Must be 'disk', 'square', or a 2D ndarray.")

    # Apply the specified morphological operation
    if operation == 'dilation':
        result = binary_dilation(binary_image, structure=selem)
    elif operation == 'erosion':
        result = binary_erosion(binary_image, structure=selem)
    elif operation == 'opening':
        result = binary_opening(binary_image, structure=selem)
    elif operation == 'closing':
        result = binary_closing(binary_image, structure=selem)
    else:
        raise ValueError("Unsupported operation. Use 'dilation', 'erosion', 'opening', or 'closing'.")

    return result

@log_execution_time
def morphological_operator_multiclass_mask(multiclass_mask, operation='dilation', structuring_element='disk', radius_or_size=3):
    """
    Perform a morphological operation on a sparse multiclass mask using patch-based processing.
    Args:
        multiclass_mask (np.ndarray): Input mask with multiple class values (sparse).
    Returns:
        np.ndarray: Output mask after morphological operation.
    """
    # Get unique class values (skip 0, background)
    unique_classes = np.unique(multiclass_mask[multiclass_mask > 0])
    print(f"Processing {len(unique_classes)} classes...")

    # Prepare the output mask
    output_mask = np.zeros_like(multiclass_mask, dtype=multiclass_mask.dtype)

    # Patch size based on structuring element
    patch_radius = radius_or_size+3

    # Process each classx
    for class_value in unique_classes:
        # Get sparse coordinates for the current class
        min_row, min_col, max_row, max_col = get_bounding_box(multiclass_mask == class_value)

        # Skip empty classes
        if not np.any([min_row, min_col, max_row, max_col]):
            continue

        # Define a patch around the current pixel

        x_min = np.max([0,min_col-patch_radius], axis=0)
        x_max=np.min([multiclass_mask.shape[1],max_col+ patch_radius + 1], axis=0)
        y_min = np.max([0, min_row - patch_radius], axis=0)
        y_max = np.min([multiclass_mask.shape[0], max_row + patch_radius + 1], axis=0)


        # Extract the patch
        patch = multiclass_mask[y_min:y_max,x_min:x_max]

        # Create a binary mask for the current class in the patch
        binary_patch = (patch == class_value).astype(np.uint8)

        processed_patch = morphological_operator(binary_patch, operation,
                                                    structuring_element,
                                                    radius_or_size)

        # Update the output mask with the processed patch
        output_mask[y_min:y_max,x_min:x_max][
            processed_patch & (output_mask[y_min:y_max,x_min:x_max] == 0)] = class_value

    return output_mask



def analyze_and_plot_histogram_with_nan(image):
    """
    Analyzes unique values in an image, including handling NaN, and plots their histogram.

    Parameters:
    - image: ndarray
        2D array representing the image.

    Returns:
    - unique_values: ndarray
        Array of unique values in the image, excluding NaN.
    - counts: ndarray
        Array of counts corresponding to each unique value.
    - nan_count: int
        Count of NaN values in the image.
    """
    # Flatten the image to a 1D array for analysis
    flat_image = image.flatten()

    # Count NaN values
    nan_count = np.sum(np.isnan(flat_image))

    # Remove NaN values for unique value analysis
    valid_pixels = flat_image[~np.isnan(flat_image)]

    # Find unique values and their counts
    unique_values, counts = np.unique(valid_pixels, return_counts=True)

    # Print unique values and their counts
    print("Unique Values and Their Counts (excluding NaN):")
    for value, count in zip(unique_values, counts):
        print(f"Value: {value}, Count: {count}")

    # Print count of NaN values
    print(f"NaN Count: {nan_count}")

    # Plot histogram
    plt.figure(figsize=(10, 6))
    plt.bar(unique_values, counts, color='skyblue', edgecolor='black', label='Valid Values')
    if nan_count > 0:
        plt.bar(-1, nan_count, color='red', edgecolor='black', label='NaN Count')  # Use -1 to represent NaN
    plt.xlabel('Pixel Values')
    plt.ylabel('Count')
    plt.title('Histogram of Pixel Values (Including NaN)')
    plt.xticks(np.append(unique_values, -1), rotation=45)  # Include -1 for NaN label
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend()
    plt.show()

    return unique_values, counts, nan_count


def analyze_and_plot_grouped_histogram(image, group_range=1, min_value=1):
    """
    Analyzes unique values in an image, groups them into bins, and plots a histogram.

    Parameters:
    - image: ndarray
        2D array representing the image.
    - group_range: float
        The range of values to group together (e.g., ±1 groups into bins of size 2).
    - min_value: float
        Minimum pixel value to include in the analysis.

    Returns:
    - grouped_values: ndarray
        The grouped values.
    - counts: ndarray
        The counts corresponding to each group.
    """
    # Flatten the image to a 1D array for analysis
    flat_image = image.flatten()

    # Filter out unwanted values (<= min_value)
    valid_pixels = flat_image[flat_image > min_value]

    # Group values within the specified range
    grouped_pixels = np.round(valid_pixels / group_range) * group_range

    # Find unique grouped values and their counts
    grouped_values, counts = np.unique(grouped_pixels, return_counts=True)

    # Print grouped values and their counts
    print("Grouped Values and Their Counts:")
    for value, count in zip(grouped_values, counts):
        print(f"Value: {value}, Count: {count}")

    # Plot histogram
    plt.figure(figsize=(10, 6))
    plt.bar(grouped_values, counts*PIXEL_SZ**2, width=group_range * 0.9, color='skyblue', edgecolor='black')
    plt.xlabel('PCI Values')
    plt.ylabel('Estimated area')
    plt.title('Histogram of PCI Values')
    plt.xticks(grouped_values, rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

    return grouped_values, counts


def extract_label_masks_with_instances(label_mask, hyperspectral_image):
    """
    Extracts individual masks for each instance of labels in the mask and uses them
    to extract the corresponding areas in a hyperspectral image.

    Parameters:
    - label_mask: ndarray
        A 2D array where each pixel's value represents a label.
    - hyperspectral_image: ndarray
        A 3D array (height x width x bands) representing the hyperspectral image.

    Returns:
    - instance_masks: list of ndarray
        List of binary masks for each label instance.
    - extracted_areas: list of ndarray
        List of extracted areas from the hyperspectral image for each mask.
    """
    # Label connected components in the mask
    labeled_mask, num_instances = label(label_mask > 0)

    # Initialize storage for individual masks and extracted areas
    instance_masks = []
    extracted_areas = []

    # Loop through each instance
    for instance_id in range(1, num_instances + 1):
        # Create a binary mask for the current instance
        instance_mask = (labeled_mask == instance_id).astype(np.uint8)
        instance_masks.append(instance_mask)

        # Use the mask to extract the corresponding area in the hyperspectral image
        # Retain all spectral bands
        extracted_area = hyperspectral_image * instance_mask[:, :, np.newaxis]
        extracted_areas.append(extracted_area)

    return instance_masks, extracted_areas

def extract_windows_from_mask(image, mask, window_length=16, window_width=3, overlap=0.5, debug=True):
    """
    Extract windows along a mask with variable width.

    Parameters:
    - image: ndarray
        The input 2D or multi-channel image.
    - mask: ndarray
        Binary mask where 1 indicates the region of interest.
    - window_length: int
        Length of each window along the mask's centerline.
    - window_width: int
        Width of each window perpendicular to the mask's centerline.
    - overlap: float
        Overlap percentage between consecutive windows (0 to 1).
    - debug: bool
        If True, visualize the extracted windows as overlays.

    Returns:
    - windows: list of ndarrays
        List of extracted windows (shape: window_length x window_width).
    - centers: list of tuples
        List of center coordinates for each window.
    """
    # Compute the distance transform of the mask to find the centerline
    dist_transform = distance_transform_edt(mask)
    # centerline = (dist_transform > (dist_transform.max() / 2)).astype(int)

    # Label connected components of the centerline
    labeled_centerline, num_features = label(dist_transform)

    windows = []
    centers = []
    debug_mask = np.zeros_like(mask, dtype=float) if debug else None

    for label_idx in range(1, num_features + 1):
        # Extract the coordinates of the current centerline
        coords = np.argwhere(labeled_centerline == label_idx)

        # Sort the coordinates along the centerline
        coords = coords[np.argsort(coords[:, 0])]

        step = int(window_length * (1 - overlap))
        for i in range(0, len(coords) - window_length, step):
            segment = coords[i: i + window_length]

            # Compute tangent for the segment
            tangents = np.diff(segment, axis=0)
            tangent = np.mean(tangents, axis=0)  # Average tangent direction
            tangent = tangent / np.linalg.norm(tangent)

            # Compute perpendicular direction
            perpendicular = np.array([-tangent[1], tangent[0]])

            # Extract the window
            window = np.zeros((window_length, window_width, image.shape[2]))
            valid_window = True

            for j, (cy, cx) in enumerate(segment):
                for k in range(-window_width // 2, window_width // 2 + 1):
                    offset = k * perpendicular
                    wy = int(cy + offset[0])
                    wx = int(cx + offset[1])

                    # Check bounds
                    if 0 <= wy < image.shape[0] and 0 <= wx < image.shape[1]:
                        window[j, k + window_width // 2,:] = image[wy, wx,:]
                        if debug:
                            debug_mask[wy, wx] = len(windows) + 1
                    else:
                        valid_window = False

            if valid_window:
                windows.append(window)
                centers.append(segment[len(segment) // 2])

    # Debug visualization
    if debug:
        plt.figure(figsize=(10, 10))
        plt.imshow(mask, cmap="gray", alpha=0.5)
        plt.imshow(nan_arr(debug_mask), cmap="prism", interpolation="nearest", alpha=0.8)
        plt.title("Windows Overlaid on Mask")
        plt.axis("off")
        plt.colorbar(label="Window ID")
        plt.show()

    return windows, centers


def process_labeled_image(hyperspectral_image, labeled_image, labels_lut=None, dilation_radius=2):
    """
    Process a labeled image to separate connected regions, apply dilation,
    and generate masks per region.

    Parameters:
    - hyperspectral_image: ndarray
        A 3D hyperspectral image (height x width x bands).
    - labeled_image: ndarray
        A 2D array where each pixel's value represents a label.
    - labels_lut: dict or None
        Lookup table mapping segment IDs to labels. If None, uses segment IDs directly.
    - dilation_radius: int
        Radius for the dilation operation.

    Returns:
    - mask_list: list of dict
        Each entry is a dictionary with keys:
        - 'mask': The binary mask for the region.
        - 'label': The label associated with the mask.
        - 'SegID': Segment ID of the region.
        - 'bounding_box': Bounding box (min_row, min_col, max_row, max_col).
    """
    # Define connectivity for connected component labeling
    structure = generate_binary_structure(2, 2)
    mask_list = []

    # Iterate through unique labels
    unique_labels = np.unique(labeled_image)
    for label_id in tqdm.tqdm(unique_labels, desc="Processing labels"):
        if label_id <= 0 or np.isnan(label_id):  # Skip background or invalid labels
            continue

        # Binary mask for current label
        label_mask = (labeled_image == label_id).astype(np.uint8)

        # Apply dilation
        label_mask_dilated = binary_dilation(label_mask, structure=disk(dilation_radius))

        # Connected component labeling
        connected_components, num_features = label(label_mask_dilated, structure=structure)

        # Process regions based on presence of LUT
        if labels_lut is None:
            # Treat each connected region as a separate segment
            for region_id in range(1, num_features + 1):
                mask_list.extend(
                    _process_region(hyperspectral_image, connected_components, region_id, label_id, dilation_radius)
                )
        else:
            # Use the whole segment ID
            mask_list.extend(
                _process_full_segment(
                    hyperspectral_image, label_mask, label_id, labels_lut, dilation_radius
                )
            )

    # Find the minimum mask size
    rois_sizes=[[entry['bounding_box'][2] - entry['bounding_box'][0], entry['bounding_box'][3] - entry['bounding_box'][0]] for
     entry in mask_list]
    min_mask_size_id = np.argmin(rois_sizes)
    print('min_mask_size_id:',min_mask_size_id)
    return mask_list

def _process_region(hyperspectral_image, connected_components, region_id, label_id, dilation_radius):
    """
    Process a single connected region in the labeled image.
    """
    # Binary mask for the region
    region_mask = (connected_components == region_id)

    # Get bounding box
    min_row, min_col, max_row, max_col = get_bounding_box(region_mask)

    # Extract ROI and mask
    roi = hyperspectral_image[min_row:max_row, min_col:max_col, :]
    region_mask_cropped = region_mask[min_row:max_row, min_col:max_col]
    masked_roi = roi * region_mask_cropped[:, :, np.newaxis]

    if np.all(np.isnan(roi)):
        return []

    return [{
        'mask': masked_roi,
        'label': label_id,
        'SegID': 'unknown',
        'bounding_box': (min_row, min_col, max_row, max_col)
    }]

def _process_full_segment(hyperspectral_image, label_mask, label_id, labels_lut, dilation_radius):
    """
    Process a full segment in the labeled image.
    """
    # Apply dilation to the label mask
    label_mask_dilated = binary_dilation(label_mask, structure=disk(dilation_radius))

    # Get bounding box
    min_row, min_col, max_row, max_col = get_bounding_box(label_mask_dilated)

    # Extract ROI and mask
    roi = hyperspectral_image[min_row:max_row, min_col:max_col, :]
    label_mask_cropped = label_mask[min_row:max_row, min_col:max_col]
    masked_roi = roi * label_mask_cropped[:, :, np.newaxis]

    if np.all(np.isnan(roi)):
        return []

    segID = str(int(label_id))
    label_value = labels_lut[segID] if segID in labels_lut else 'unknown'

    return [{
        'mask': masked_roi,
        'label': label_value,
        'SegID': segID,
        'bounding_box': (min_row, min_col, max_row, max_col)
    }]

def get_bounding_box(mask):
    """
    Calculate the bounding box of a binary mask.
    """
    rows, cols = np.where(mask)
    if rows.size == 0 or cols.size == 0:
        return 0, 0, 0, 0
    return rows.min(), cols.min(), rows.max() + 1, cols.max() + 1


def plot_centered_window(image, center_xy, half_window_size):
    h = half_window_size
    x,y = center_xy
    # Ensure the indices stay within image boundaries
    y_min = max(0, y - h)
    y_max = min(image.shape[0], y + h + 1)
    x_min = max(0, x - h)
    x_max = min(image.shape[1], x + h + 1)
    # Extract the window
    window = image[y_min:y_max, x_min:x_max]
    return window


def rgb_to_yuv(image_rgb):
    """
    Converts an RGB image to YUV color space manually using NumPy.

    Parameters:
    - image_rgb: np.ndarray
        Input image in RGB format with shape (H, W, 3).

    Returns:
    - image_yuv: np.ndarray
        Output image in YUV format with shape (H, W, 3).
    """
    # Define the RGB to YUV conversion matrix (BT.601)
    conversion_matrix = np.array([[0.299, 0.587, 0.114],
                                  [-0.14713, -0.28886, 0.436],
                                  [0.615, -0.51499, -0.10001]])

    # Reshape the image to a 2D array of pixels
    pixels = image_rgb.reshape(-1, 3)

    # Apply the conversion matrix
    yuv_pixels = np.dot(pixels, conversion_matrix.T)

    # Reshape back to the original image shape
    image_yuv = yuv_pixels.reshape(image_rgb.shape)

    # Clip the values to the valid range [0, 255] and convert to uint8
    # image_yuv = np.clip(image_yuv, 0, 255).astype(np.uint8)

    Y, U, V = cv2.split(image_yuv)
    return Y, U, V




def mean_filter(image, kernel_size=3):
    # Create a kernel of ones and normalize it
    kernel = np.ones((kernel_size, kernel_size)) / (kernel_size * kernel_size)

    # Apply the convolution
    filtered_image = convolve2d(image, kernel, mode='same', boundary='symm')

    return filtered_image


def sobel_gradient(image):
    # Sobel kernels
    image=np.nan_to_num(image, nan=np.nanmean(image))
    sobel_x = np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]])
    sobel_y = np.array([[-1, -2, -1],
                        [0, 0, 0],
                        [1, 2, 1]])
    # Convolve image with Sobel kernels
    gx = convolve2d(image, sobel_x, mode='same', boundary='symm')
    gy = convolve2d(image, sobel_y, mode='same', boundary='symm')
    # Gradient magnitude
    magnitude = np.sqrt(gx ** 2 + gy ** 2)
    return gx, gy, magnitude



if __name__ == "__main__":

    # Example point cloud with value dimension
    dummy_mask = np.zeros((10, 10), dtype=int)
    points_PCI = np.array([[1, 2, 20], [1.2, 2.2, 80], [2.2, 2.2, 25], [2.5, 2.5, 80], [3.5, 2.9, 50], [3.4, 2.8, 10],
                           [5, 5, 80], [4.9, 5.1, 90], [6.1, 2.8, 90], [6.0, 2.9, 99]])  # Example points
    points = points_PCI[:, :2]
    PCI = points_PCI[:, 2]
    print(len(points_PCI))
    points_merge_PCI = merge_close_points(points, PCI, 0.8)
    print(points_merge_PCI)
    print(len(points_merge_PCI))

    xy_pointcloud = points_PCI[:, :2]
    xy_pointcloud_merge = points_merge_PCI[:, :2]

    extended_mask, line_string = fill_mask_with_spline(dummy_mask, xy_pointcloud_merge,
                                                       combine_mask=False)  # this return mask in the pixels the spline line passes through

    # # Visualize the point cloud, spline, and binary mask
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    ax.plot(xy_pointcloud[:, 0], xy_pointcloud[:, 1], 'ko',markersize=10,markeredgecolor='black',markeredgewidth=1,markerfacecolor=('blue', 0.3), label='Point Cloud with lane duplicate')
    # x_new, y_new, _ = fit_spline_pc(xy_points)
    # ax.plot(x_new, y_new, 'b-', label='Spline Fit')
    # plt.plot(xy_points_merge[:, 0], xy_points_merge[:, 1], 'rx', label='Point Cloud new')

    # Now plot with merge point
    x_new, y_new, _ = fit_spline_pc(xy_pointcloud_merge)
    # plt.plot(x_new, y_new, 'c-', label='Spline Fit new')
    # plt.legend()
    # plt.show()
    #
    # # Plot the original mask, spline fit, and updated mask
    # fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    #
    # # Plot the original mask with the spline overlayed
    # ax.imshow(extended_mask, cmap='gray', origin='lower')
    # ax.plot(x_new, y_new, 'b-', label='Spline Fit')
    # ax = scatter_plot_with_annotations(points_merge_PCI,ax)
    # ax.set_title("Spline Fit")
    # ax.legend()
    # plt.show()

    segment_mask = fill_mask_with_line_point_values(line_string, points_merge_PCI, extended_mask.shape, radius=3.5)

    # Plot the original mask, spline fit, and updated mask
    # fig, ax = plt.subplots(1, 1, figsize=(12, 6))

    # Plot the original mask with the spline overlayed
    cmap_me = plt_utils.get_lighttraffic_colormap()
    im = ax.imshow(segment_mask, cmap=cmap_me, origin='lower')

    ax.plot(x_new, y_new, 'b-', label='Spline Fit')
    ax = plt_utils.scatter_plot_with_annotations(points_merge_PCI, ax)
    ax.set_title("Spline Fit")
    ax.legend()
    cbar = fig.colorbar(im, ticks=[0, 30,70, 100], orientation='vertical')
    cbar.ax.set_yticklabels(['severe','critical', 'moderate', 'good'])
    plt.show()

    a = 1


