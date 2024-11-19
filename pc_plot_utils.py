import numpy as np
import matplotlib.pyplot as plt
import cv2
import time
import functools



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

def plot_mask_over_gray_img(X_cropped, Y_cropped, hys_img,coinciding_mask,*args):
    fig_ani, ax_ani = plt.subplots()
    # Use pcolormesh to create the initial empty grid (binary mask)
    im_ax = ax_ani.pcolormesh(X_cropped, Y_cropped, hys_img[:, :, -1], cmap='gray')
    c_ax = ax_ani.pcolormesh(X_cropped, Y_cropped, coinciding_mask)
    pass


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


def plot_scatter_over_map(X_cropped,Y_cropped,hys_img,points_merge_PCI,x_new,y_new,coinciding_mask):
    # Plot the masked data using pcolormesh
    fig_roi, ax_roi = plt.subplots()
    im_ax = ax_roi.pcolormesh(X_cropped, Y_cropped, hys_img[:, :, -1], cmap='gray')
    scatter_plot_with_annotations(points_merge_PCI, ax_roi, markersize=100, linewidths=2, alpha=1)
    ax_roi.plot(x_new, y_new, 'b--', label='Spline Fit')
    ax_roi.pcolormesh(X_cropped, Y_cropped, coinciding_mask, alpha=0.2)
    pass

