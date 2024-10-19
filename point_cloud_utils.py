import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev,griddata
from skimage.draw import line, polygon
from scipy.spatial import Delaunay
from scipy.spatial.distance import pdist, squareform
from scipy.sparse.csgraph import minimum_spanning_tree
from shapely.geometry import LineString, Point
from scipy.spatial.distance import cdist


import time
import functools

# ANSI escape codes for purple text with a black background
PURPLE_ON_BLACK = "\033[45;30m"
RESET = "\033[0m"


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

def reorder_points_greedy(points):
    """
    Reorder points to form a continuous path by connecting each point to its nearest neighbor.
    Parameters:
    - points: numpy array of shape (n, 2), representing (x, y) coordinates.
    Returns:
    - ordered_points: numpy array of reordered points (x, y).
    """
    n_points = points.shape[0]
    visited = np.zeros(n_points, dtype=bool)

    # Start with the first point (arbitrary choice)
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
    # Reorder points using the greedy nearest-neighbor algorithm
    reordered_points = reorder_points_greedy(points)

    # Fit a spline through the reordered points
    tck, u = splprep(reordered_points.T, s=0)
    spline_res = kwargs.get('spline_res', 100)

    x_new, y_new = splev(np.linspace(0, 1, spline_res), tck)

    # Create a Shapely LineString from the spline points
    line_string = LineString(np.c_[x_new, y_new])

    return x_new, y_new, line_string
# -------------------------------
# Merge Close Points
# -------------------------------
def merge_close_points(points,PCI, threshold=0.551):
    # np.concatenate([points,PCI.reshape(-1, 1)],axis=1)
    points=np.c_[points,PCI]
    merged_points = []
    skip_indices = set()

    for i in range(len(points)):
        if i in skip_indices:
            continue
        close_indices = np.where(cdist([points[i, :2]], points[:, :2])[0] < threshold)[0]
        merged_x = np.mean(points[close_indices, 0])
        merged_y = np.mean(points[close_indices, 1])
        merged_value = np.min(points[close_indices, 2])  # Take minimum value

        merged_points.append([merged_x, merged_y, merged_value])
        skip_indices.update(close_indices)

    return np.array(merged_points)


def fill_mask_with_irregular_spline(xy_points, X_grid, Y_grid, binary_mask, radius=4.5,combine_mask=False):
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
    x_new, y_new, line_string = fit_spline_pc(xy_points)
    # Initialize the mask with NaN to indicate unfilled pixels
    mask_shape=binary_mask.shape[:2]
    # updated_mask = np.zeros_like(binary_mask)
    updated_mask = np.full(mask_shape, np.nan, dtype=float)

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

    return extended_mask,line_string



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
    markersize_kwargs = kwargs.get('markersize', 100)
    markeralpha_kwargs = kwargs.get('alpha', 0.5)
    linewidths_kwargs = kwargs.get('linewidths', 0.7)

    scatter = ax.scatter(x, y, c=values, s=markersize_kwargs, alpha=markeralpha_kwargs, edgecolor='black',linewidths=linewidths_kwargs,cmap=cmap_kwargs)

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
    mask_below_30 = segmented_image < 30
    mask_30_to_70 = (segmented_image >= 30) & (segmented_image <= 70)
    mask_above_85 = segmented_image > 85

    return mask_below_30, mask_30_to_70, mask_above_85







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
    cbar.ax.set_yticklabels(['severe','critical', 'moderate', 'ok'])
    plt.show()

    a = 1
