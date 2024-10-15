import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev
from skimage.draw import line, polygon
from shapely.geometry import LineString, Point
from scipy.spatial.distance import cdist


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


def fit_spline_pc(points):
    # Fit a spline through the point cloud
    tck, u = splprep(points.T, s=0)
    x_new, y_new = splev(np.linspace(0, 1, 100), tck)
    # Create a Shapely LineString from the spline points
    line_string = LineString(np.c_[x_new, y_new])
    return x_new,y_new,line_string


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


def scatter_plot_with_annotations(points, ax=None):
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
    scatter = ax.scatter(x, y, c=values, s=50, alpha=0.7, edgecolor='black')

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


def fill_line_with_point_values(line_string, points, mask_shape, radius=3.5):
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
        distances = cdist(pixel_coords, points[:, :2])

        # Find the closest point index for each pixel
        closest_indices = np.argmin(distances, axis=1)
        closest_distances = np.min(distances, axis=1)

        # Assign values only if the closest point is within the given radius
        for j, (r, c) in enumerate(zip(rr, cc)):
            if closest_distances[j] <= radius:
                segment_mask[r, c] = points[closest_indices[j], 2]

    return segment_mask


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




if __name__ == "__main__":

    # Example point cloud with value dimension
    binary_mask = np.zeros((10, 10), dtype=int)
    points_PCI = np.array([[1, 2, 20], [1.2, 2.2, 80], [2.1, 2.1, 25], [2.5, 2.5, 80], [3.5, 2.9, 50], [3.9, 2.8, 10],
                           [5, 5, 80], [4.5, 5.5, 90], [6.7, 2.8, 90], [6.0, 2.8, 99]])  # Example points
    points = points_PCI[:, :2]
    PCI = points_PCI[:, 2]
    print(len(points_PCI))
    points_merge_PCI = merge_close_points(points, PCI, 0.8)
    print(points_merge_PCI)
    print(len(points_merge_PCI))

    xy_points = points_PCI[:, :2]
    xy_points_merge = points_merge_PCI[:, :2]

    extended_mask, line_string = fill_mask_with_spline(binary_mask, xy_points_merge, combine_mask=False)

    # # Visualize the point cloud, spline, and binary mask
    # plt.plot(xy_points[:, 0], xy_points[:, 1], 'ro', label='Point Cloud')
    # x_new, y_new, _ = fit_spline_pc(xy_points)
    # plt.plot(x_new, y_new, 'b-', label='Spline Fit')
    # plt.plot(xy_points_merge[:, 0], xy_points_merge[:, 1], 'rx', label='Point Cloud new')
    #
    x_new, y_new, _ = fit_spline_pc(xy_points_merge)
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

    segment_mask = fill_line_with_point_values(line_string, points_merge_PCI, extended_mask.shape, radius=3.5)

    # Plot the original mask, spline fit, and updated mask
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))

    # Plot the original mask with the spline overlayed
    cmap_me = get_lighttraffic_colormap()
    ax.imshow(segment_mask, cmap=cmap_me, origin='lower')
    ax.plot(x_new, y_new, 'b-', label='Spline Fit')
    ax = scatter_plot_with_annotations(points_merge_PCI, ax)
    ax.set_title("Spline Fit")
    ax.legend()

    plt.show()

    a = 1
