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
from scipy.ndimage import binary_dilation,binary_erosion,binary_opening,binary_closing
from scipy.ndimage import distance_transform_edt, label
from scipy.spatial import distance
from collections import defaultdict, deque
from scipy.ndimage import map_coordinates
from scipy import stats
import cv2
import time
import functools
from GetRoadsModule import GetRoadsCoordinates
from sklearn.decomposition import PCA

from crop_runner import nan_arr

# ANSI escape codes for purple text with a black background
PURPLE_ON_BLACK = "\033[45;30m"
RESET = "\033[0m"

PIXEL_SZ= 4 # pixel size in meter

def log_execution_time(func):
    """
    A decorator that logs the function being called and its execution time.
    Logs are printed in purple with a black background.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time

        # Log function name and execution time in purple on black background
        print(f"{PURPLE_ON_BLACK}Function '{func.__name__}' was called and took {execution_time:.4f} seconds{RESET}")

        return result

    return wrapper


# Example usage
def get_lighttraffic_colormap():
    """
    Create a colormap that maps low values to red, mid values to yellow,
    and high values to green, based on the 'jet' colormap.

    Returns:
    - A customized LinearSegmentedColormap object.
    """
    from matplotlib.colors import ListedColormap, BoundaryNorm
    from matplotlib.colors import LinearSegmentedColormap
    # # Define custom colors
    # colors = ['#999999', '#ff0000', '#fbff00', '#32a852']
    # cmap_me = ListedColormap(colors)
    # bounds = [-0.5, 0.5, 30.5, 70.5, 100.5]
    # norm = BoundaryNorm(bounds, cmap_me.N)
    #

    # Get the 'jet' colormap
    jet = plt.get_cmap('jet')

    # Extract colors from 0.35 to 0.75 range (red to green)
    colors = jet(np.linspace(0.5, 1, 256))

    # Flip the colors to map high values to green and low to red
    colors = colors[::-1]

    # Create a new colormap from the sliced and flipped colors
    lighttraffic = LinearSegmentedColormap.from_list('lighttraffic', colors)

    return lighttraffic

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



# Example usage of fill_line_with_point_values in combination with pcolormesh
def plot_with_pcolormesh(X_cropped, Y_cropped, Z_cropped, mask_grid):
    plt.figure(figsize=(8, 6))

    # Plot the original pcolormesh grid
    plt.pcolormesh(X_cropped, Y_cropped, Z_cropped, cmap='viridis', shading='auto')
    plt.colorbar(label="Z_cropped values")

    # Overlay the mask using pcolormesh
    plt.pcolormesh(X_cropped, Y_cropped, mask_grid, cmap='Reds', shading='auto', alpha=0.6)
    plt.colorbar(label="Mask values")

    plt.title("Mask Overlay on Grid")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()


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


def scatter_plot_with_annotations(points, ax=None,**kwargs):
    """
    Overlay a scatter plot with annotated value dimension on the provided axis.

    Parameters:
    - points: numpy array of shape (n, 3), where columns represent (x, y, value).
    - ax: matplotlib axis, optional. If provided, the scatter plot will be drawn on this axis.

    Returns:
    - ax: The matplotlib axis object with the scatter plot and annotations.
    """
    # Extract x, y, and value dimensions
    x = points[:, 0]
    y = points[:, 1]
    values = points[:, 2]

    # If no axis is provided, create a new figure and axis
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.grid(True)

    # Scatter plot overlay
    cmap_me = get_lighttraffic_colormap()
    cmap_kwargs = kwargs.get('cmap', cmap_me)
    annotate_flag = kwargs.get('annotate_flag', True)
    markersize_kwargs = kwargs.get('markersize', 100)
    markeralpha_kwargs = kwargs.get('alpha', 0.5)
    linewidths_kwargs = kwargs.get('linewidths', 0.7)

    scatter = ax.scatter(x, y, c=values, s=markersize_kwargs, alpha=markeralpha_kwargs, edgecolor='black',linewidths=linewidths_kwargs,cmap=cmap_kwargs)

    if annotate_flag:
        # Annotate each point with its rounded value
        for i in range(len(points)):
            ax.annotate(f'{int(round(values[i]))}',
                        (x[i], y[i]),
                        textcoords="offset points",  # Position text relative to point
                        xytext=(10, 10),  # Offset the text slightly
                        ha='center',  # Horizontal alignment
                        fontsize=10,  # Font size for annotation
                        bbox=dict(boxstyle="round,pad=0.3", edgecolor='gray', facecolor='white'))

    # Return the axis object with the scatter plot
    return ax

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


def apply_masks_and_average(image, mask):
    """
    Apply a mask to the multi-channel image and compute the average pixel values for each channel.

    Parameters:
    - image: 3D numpy array representing the image (height, width, 12).
    - mask: 2D boolean numpy array representing the mask to be applied.

    Returns:
    - channel_values: A list containing the averaged pixel values for each channel.
    """
    # Apply the mask and extract the corresponding pixel values for each channel
    channel_values = [image[:, :, channel][mask] for channel in range(image.shape[2])]

    return channel_values


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
    data = np.copy(segmented_pci_spectral)

    # Number of data points
    spectral_bands, n_points = data.shape

    mean = np.nanmean(data,axis=1)
    std_dev = np.nanstd(data, ddof=1,axis=1)  # Use ddof=1 for sample standard deviation


    # Calculate the Standard Error of the Mean (SEM)
    sem = std_dev / np.sqrt(n_points)

    # Define the confidence level (e.g., 95%)
    confidence_level = 0.95

    # Calculate the t critical value (two-tailed)
    t_critical = stats.t.ppf((1 + confidence_level) / 2, df=n_points - 1)

    # Calculate the margin of error
    margin_of_error = t_critical * sem

    return (n_points,mean,std_dev,sem,margin_of_error)


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
    red = normalize_img(pca_image[:, :, 0])
    green = normalize_img(pca_image[:, :, 1])
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
    cmap_me = get_lighttraffic_colormap()
    im = ax.imshow(segment_mask, cmap=cmap_me, origin='lower')

    ax.plot(x_new, y_new, 'b-', label='Spline Fit')
    ax = scatter_plot_with_annotations(points_merge_PCI, ax)
    ax.set_title("Spline Fit")
    ax.legend()
    cbar = fig.colorbar(im, ticks=[0, 30,70, 100], orientation='vertical')
    cbar.ax.set_yticklabels(['severe','critical', 'moderate', 'good'])
    plt.show()

    a = 1


