"""
Data processing modules for APA.

Provides implementations for ROI processing, road extraction,
PCI segmentation, and data preprocessing.
"""

from typing import Any, Dict, Optional, Tuple
import numpy as np
import os
from pathlib import Path

from apa.common import (
    BaseDataProcessor,
    DataContainer,
    ProcessingResult,
    DataError,
)

# Try to import Dataset enum
try:
    from enums.datasets_enum import Dataset
except ImportError:
    print("Error importing enums.datasets_enum")

# Import pc_utils from local utils module
try:
    from apa.utils.point_cloud_utils import (
        normalize_hypersepctral_bands,
        merge_close_points,
        is_roi_within_bounds,
    )
    HAS_PC_UTILS = True
    # Create alias for compatibility
    class PCUtils:
        normalize_hypersepctral_bands = normalize_hypersepctral_bands
        merge_close_points = merge_close_points
        is_roi_within_bounds = is_roi_within_bounds
    pc_utils = PCUtils()
except ImportError:
    HAS_PC_UTILS = False
    pc_utils = None

# Import road mask loader
try:
    from apa.utils.road_mask_loader import (
        get_mask_from_roads_gdf,
        create_mask_from_roads_gdf,
    )
    HAS_ROAD_MASK_LOADER = True
except ImportError:
    HAS_ROAD_MASK_LOADER = False
    get_mask_from_roads_gdf = None
    create_mask_from_roads_gdf = None


class ROIProcessor(BaseDataProcessor):
    """
    Processor for regions of interest (ROI).
    
    Crops imagery to specified regions and handles coordinate transformations.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize ROI processor.
        
        Args:
            config: Configuration dictionary with keys:
                - roi_bounds: [min_lat, max_lat, min_lon, max_lon]
                - coordinate_system: 'latlon' or 'utm'
        """
        super().__init__("roi_processor", config)
        self.supported_data_types = ['hyperspectral', 'ground_truth_pci']
    
    def _process_impl(self, data: DataContainer, config: Dict[str, Any]) -> DataContainer:
        """
        Process ROI cropping.
        
        Crops hyperspectral image to specified ROI bounds and normalizes the data.
        Returns cropped image, coordinate matrices, and RGB image.
        
        Args:
            data: Input data container with dictionary containing:
                - 'image': Hyperspectral image array (height, width, bands)
                - 'lon_mat': Longitude matrix (height, width)
                - 'lat_mat': Latitude matrix (height, width)
            config: Processing configuration with:
                - 'roi_bounds': [xmin_cut, xmax_cut, ymin_cut, ymax_cut] (lon_min, lon_max, lat_min, lat_max)
                - 'enum_data_source': Dataset enum value (optional, from metadata if not provided)
            
        Returns:
            Processed data container with cropped data
        """
        roi_bounds = config.get('roi_bounds')
        if roi_bounds is None:
            # No ROI specified, return original data
            return data
        
        # Validate input data structure
        if not isinstance(data.data, dict):
            raise DataError("ROI processing requires data as dictionary with 'image', 'lon_mat', 'lat_mat'")
        
        image = data.data.get('image')
        lon_mat = data.data.get('lon_mat')
        lat_mat = data.data.get('lat_mat')
        
        if image is None or lon_mat is None or lat_mat is None:
            raise DataError("ROI processing requires 'image', 'lon_mat', and 'lat_mat' in data dictionary")
        
        # Get enum_data_source from config or metadata
        enum_data_source = config.get('enum_data_source')
        if enum_data_source is None:
            enum_data_source = data.metadata.get('enum_data_source')
        if enum_data_source is None:
            enum_data_source = data.metadata.get('dataset')
        
        if enum_data_source is None:
            # Default to venus_Detroit if not specified
            enum_data_source = 1
        
        # Convert to Dataset enum if needed
        if isinstance(enum_data_source, int):
            dataset = Dataset(enum_data_source)
        elif isinstance(enum_data_source, Dataset):
            dataset = enum_data_source
        else:
            dataset = Dataset.venus_Detroit  # Default
        
        # Crop ROI using the original cropROI_Venus_image logic
        X_cropped, Y_cropped, cropped_MSP_img, RGB_enhanced, cropped_rect = self._crop_roi_venus_image(
            roi_bounds, lon_mat, lat_mat, image, dataset
        )
        
        # Extract optical center (first two values of cropped_rect)
        OC_xy = cropped_rect[:2]
        print(f'Optical center ROI in xy[column][row]: {OC_xy}\n')
        
        # Prepare output data dictionary
        processed_data = {
            'image': cropped_MSP_img,
            'lon_mat': Y_cropped,  # Y is longitude
            'lat_mat': X_cropped,  # X is latitude
            'RGB_enhanced': RGB_enhanced,
        }
        
        metadata = {
            **data.metadata,
            'roi_bounds': roi_bounds,
            'cropped_rect': cropped_rect,
            'optical_center': OC_xy,
            'enum_data_source': enum_data_source,
            'processor': self.name,
        }
        
        return DataContainer(
            data=processed_data,
            metadata=metadata,
            data_type=data.data_type
        )
    
    def _crop_roi_venus_image(self, roi: list, lon_mat: np.ndarray, lat_mat: np.ndarray,
                              MSP_Image: np.ndarray, dataset: Dataset) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Tuple[int, int, int, int]]:
        """
        Crop ROI from Venus/Airbus image.
        
        Implements the original cropROI_Venus_image function logic.
        
        Args:
            roi: ROI bounds as [xmin_cut, xmax_cut, ymin_cut, ymax_cut] (lon_min, lon_max, lat_min, lat_max)
            lon_mat: Longitude matrix
            lat_mat: Latitude matrix
            MSP_Image: Multispectral/hyperspectral image
            dataset: Dataset enum value
        
        Returns:
            Tuple of (X_cropped, Y_cropped, cropped_MSP_img, RGB_enhanced, cropped_rect)
        """
        xmin_cut, xmax_cut, ymin_cut, ymax_cut = roi
        
        # Get the indices corresponding to the cut boundaries
        # Note: roi format is [xmin_cut, xmax_cut, ymin_cut, ymax_cut]
        # where xmin_cut, xmax_cut are longitude bounds (checked against lon_mat)
        # and ymin_cut, ymax_cut are latitude bounds (checked against lat_mat)
        # This matches the user's original logic: lon_mat > ymin_cut, lat_mat > xmin_cut
        idx_roi = np.argwhere((lon_mat > ymin_cut) & (lon_mat < ymax_cut) &
                              (lat_mat > xmin_cut) & (lat_mat < xmax_cut))
        
        if len(idx_roi) == 0:
            raise DataError(f"No pixels found within ROI bounds: {roi}")
        
        # Get the bounding box indices
        x_ind_min, x_ind_max = np.min(idx_roi[:, 1]), np.max(idx_roi[:, 1])
        y_ind_min, y_ind_max = np.min(idx_roi[:, 0]), np.max(idx_roi[:, 0])
        
        # Cut the image based on indices
        cropped_MSP_img = MSP_Image[y_ind_min:y_ind_max, x_ind_min:x_ind_max, :].astype(float)
        
        # Create RGB image based on dataset type
        if dataset == Dataset.venus_Detroit:
            RGB_Img = cropped_MSP_img[:, :, [6, 3, 1]].astype(float)
        elif dataset == Dataset.airbus_HSP_Detroit:
            RGB_Img = cropped_MSP_img[:, :, 3:].astype(float)
        elif dataset == Dataset.airbus_Pan_Detroit:
            # Panchromatic: repeat single band to create RGB
            RGB_Img = np.repeat(cropped_MSP_img, repeats=3, axis=2)
        else:
            # Default: use first 3 bands
            RGB_Img = cropped_MSP_img[:, :, :3].astype(float)
        
        # Normalize RGB image
        RGB_Img[RGB_Img <= 0] = np.nan
        norm_vec = np.nanpercentile(RGB_Img, q=95, axis=(0, 1)).astype(float)
        
        for normBandIdx in range(len(norm_vec)):
            img = RGB_Img[:, :, normBandIdx]
            if norm_vec[normBandIdx] > 0:
                RGB_Img[:, :, normBandIdx] = img / norm_vec[normBandIdx]
        
        # Crop coordinate matrices
        lon_mat_roi = lon_mat[y_ind_min:y_ind_max, x_ind_min:x_ind_max]
        lat_mat_roi = lat_mat[y_ind_min:y_ind_max, x_ind_min:x_ind_max]
        
        # Normalize hyperspectral image bands
        cropped_MSP_img = self._normalize_hyperspectral_bands(cropped_MSP_img)
        
        # Return cropped data
        X_cropped = lat_mat_roi
        Y_cropped = lon_mat_roi
        cropped_rect = (x_ind_min, y_ind_min, x_ind_max, y_ind_max)
        
        return X_cropped, Y_cropped, cropped_MSP_img, RGB_Img, cropped_rect
    
    def _normalize_hyperspectral_bands(self, hys_img: np.ndarray) -> np.ndarray:
        """
        Normalize hyperspectral image bands.
        
        Normalizes each band by dividing by the maximum value and sets values <= 0 to NaN.
        
        Args:
            hys_img: Hyperspectral image array (height, width, bands)
        
        Returns:
            Normalized hyperspectral image
        """
        # Try to use pc_utils if available
        if HAS_PC_UTILS and hasattr(pc_utils, 'normalize_hypersepctral_bands'):
            return pc_utils.normalize_hypersepctral_bands(hys_img)
        
        # Fallback: manual normalization
        hys_img_norm = np.zeros_like(hys_img)
        
        for kk in range(hys_img.shape[-1]):
            hys_img_1chn = hys_img[:, :, kk]
            max_val = np.nanmax(hys_img_1chn)
            if max_val > 0:
                hys_img_1chn = hys_img_1chn / max_val
            hys_img_1chn[hys_img_1chn <= 0] = np.nan
            hys_img_norm[:, :, kk] = hys_img_1chn
        
        return hys_img_norm


class RoadExtractor(BaseDataProcessor):
    """
    Processor for extracting road networks from imagery.
    
    Creates binary road masks from satellite imagery.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize road extractor.
        
        Args:
            config: Configuration dictionary with keys:
                - method: 'osm', 'segmentation', etc.
                - threshold: Threshold for road detection
        """
        super().__init__("road_extractor", config)
        self.supported_data_types = ['hyperspectral', 'road_network']
    
    def _process_impl(self, data: DataContainer, config: Dict[str, Any]) -> DataContainer:
        """
        Extract road network from hyperspectral imagery using OpenStreetMap data.
        
        Main function for doing geo-reference between PCI data and HSI images.
        Extracts road mask from OpenStreetMap and aligns it with the hyperspectral data.
        
        Args:
            data: Input data container with dictionary containing:
                - 'image': Hyperspectral image array (height, width, bands)
                - 'lon_mat': Longitude matrix (height, width)
                - 'lat_mat': Latitude matrix (height, width)
            config: Processing configuration with:
                - Full config structure with 'data' and 'preprocessing' sections
                - 'data': Contains 'enum_data_source', 'zone', 'rois'
                - 'preprocessing': Contains 'georeferencing' with 'osx_map_mask_path'
            
        Returns:
            DataContainer with road mask (binary array where 1 = road, 0 = non-road)
        """
        if not HAS_ROAD_MASK_LOADER:
            raise DataError("road_mask_loader module is required for road extraction")
        
        # Validate input data structure
        if not isinstance(data.data, dict):
            raise DataError("Road extraction requires data as dictionary with 'image', 'lon_mat', 'lat_mat'")
        
        image = data.data.get('image')
        lon_mat = data.data.get('lon_mat')
        lat_mat = data.data.get('lat_mat')
        
        if image is None or lon_mat is None or lat_mat is None:
            raise DataError("Road extraction requires 'image', 'lon_mat', and 'lat_mat' in data dictionary")
        
        # Extract config sections - handle both flat and nested config
        if 'data' in config:
            data_config = config['data']
        else:
            data_config = config
        
        if 'preprocessing' in config and 'georeferencing' in config['preprocessing']:
            georef_config = config['preprocessing']['georeferencing']
        else:
            georef_config = config.get('georeferencing', {})
        
        # Get enum_data_source
        enum_data_source = data_config.get('enum_data_source', 1)
        
        # Get ROI from config or metadata
        roi = None
        if 'rois' in data_config and data_config['rois']:
            roi = data_config['rois'][0]  # Use first ROI
        elif 'roi_bounds' in config:
            roi = config['roi_bounds']
        elif 'roi_bounds' in data.metadata:
            roi = data.metadata['roi_bounds']
        else:
            # Calculate ROI from coordinate matrices
            roi = [
                np.min(lon_mat),  # xmin_cut (lon_min)
                np.max(lon_mat),  # xmax_cut (lon_max)
                np.min(lat_mat),  # ymin_cut (lat_min)
                np.max(lat_mat),  # ymax_cut (lat_max)
            ]
        
        # Get cropped data - check if already cropped (from ROIProcessor)
        if 'cropped_rect' in data.metadata:
            # Data already cropped by ROIProcessor
            X_cropped = lat_mat
            Y_cropped = lon_mat
            cropped_rect = data.metadata['cropped_rect']
        else:
            # Need to crop ROI first
            # Convert enum_data_source to Dataset enum
            if isinstance(enum_data_source, int):
                dataset = Dataset(enum_data_source)
            elif isinstance(enum_data_source, Dataset):
                dataset = enum_data_source
            else:
                dataset = Dataset.venus_Detroit  # Default
            
            # Crop ROI using the same logic as ROIProcessor
            X_cropped, Y_cropped, cropped_image, RGB_enhanced, cropped_rect = self._crop_roi_for_roads(
                roi, lon_mat, lat_mat, image, dataset
            )
        
        # Get road mask from OpenStreetMap
        # Determine NPZ filename
        if 'osx_map_mask_path' in georef_config:
            npz_filename = georef_config['osx_map_mask_path']
        else:
            npz_filename = 'data/Detroit/masks_OpenStreetMap/Detroit_OpenSteet_roads_mask.npz'
        
        # Add enum_data_source suffix
        if npz_filename.endswith('.npz'):
            npz_filename = npz_filename[:-4] + str(enum_data_source) + '.npz'
        else:
            npz_filename = npz_filename + str(enum_data_source) + '.npz'
        
        # Get REPO_ROOT (project root)
        repo_root = self._get_repo_root()
        npz_filename = os.path.join(repo_root, npz_filename)
        
        # Prepare mask data
        mask_data = {
            "roi": roi,
            "X_cropped": X_cropped,
            "Y_cropped": Y_cropped
        }
        
        # Get road mask
        coinciding_mask = get_mask_from_roads_gdf(npz_filename, cropped_rect, mask_data)
        
        metadata = {
            **data.metadata,
            'method': 'osm',
            'processor': self.name,
            'npz_filename': npz_filename,
            'roi': roi,
            'cropped_rect': cropped_rect,
        }
        
        return DataContainer(
            data=coinciding_mask,
            metadata=metadata,
            data_type='road_mask'
        )
    
    def _crop_roi_for_roads(self, roi: list, lon_mat: np.ndarray, lat_mat: np.ndarray,
                           MSP_Image: np.ndarray, dataset: Dataset) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Tuple[int, int, int, int]]:
        """
        Crop ROI from image for road extraction.
        
        Simplified version of cropROI_Venus_image that doesn't create RGB or normalize.
        Only crops the image and coordinates.
        
        Args:
            roi: ROI bounds as [xmin_cut, xmax_cut, ymin_cut, ymax_cut] (lon_min, lon_max, lat_min, lat_max)
            lon_mat: Longitude matrix
            lat_mat: Latitude matrix
            MSP_Image: Multispectral/hyperspectral image
            dataset: Dataset enum value
        
        Returns:
            Tuple of (X_cropped, Y_cropped, cropped_MSP_img, RGB_enhanced, cropped_rect)
        """
        xmin_cut, xmax_cut, ymin_cut, ymax_cut = roi
        
        # Get the indices corresponding to the cut boundaries
        idx_roi = np.argwhere((lon_mat > ymin_cut) & (lon_mat < ymax_cut) &
                              (lat_mat > xmin_cut) & (lat_mat < xmax_cut))
        
        if len(idx_roi) == 0:
            raise DataError(f"No pixels found within ROI bounds: {roi}")
        
        # Get the bounding box indices
        x_ind_min, x_ind_max = np.min(idx_roi[:, 1]), np.max(idx_roi[:, 1])
        y_ind_min, y_ind_max = np.min(idx_roi[:, 0]), np.max(idx_roi[:, 0])
        
        # Cut the image based on indices
        cropped_MSP_img = MSP_Image[y_ind_min:y_ind_max, x_ind_min:x_ind_max, :].astype(float)
        
        # Create RGB image based on dataset type (for compatibility, but not used for road extraction)
        if dataset == Dataset.venus_Detroit:
            RGB_Img = cropped_MSP_img[:, :, [6, 3, 1]].astype(float)
        elif dataset == Dataset.airbus_HSP_Detroit:
            RGB_Img = cropped_MSP_img[:, :, 3:].astype(float)
        elif dataset == Dataset.airbus_Pan_Detroit:
            RGB_Img = np.repeat(cropped_MSP_img, repeats=3, axis=2)
        else:
            RGB_Img = cropped_MSP_img[:, :, :3].astype(float)
        
        # Crop coordinate matrices
        lon_mat_roi = lon_mat[y_ind_min:y_ind_max, x_ind_min:x_ind_max]
        lat_mat_roi = lat_mat[y_ind_min:y_ind_max, x_ind_min:x_ind_max]
        
        # Return cropped data
        X_cropped = lat_mat_roi
        Y_cropped = lon_mat_roi
        cropped_rect = (x_ind_min, y_ind_min, x_ind_max, y_ind_max)
        
        return X_cropped, Y_cropped, cropped_MSP_img, RGB_Img, cropped_rect
    
    def _get_repo_root(self) -> str:
        """
        Get repository root directory.
        
        Looks for common markers like 'src', 'configs', 'setup.py' to find project root.
        
        Returns:
            Path to repository root
        """
        cwd = os.getcwd()
        
        # Look for project root markers
        for root_marker in ['setup.py', 'src', 'configs', '.git']:
            current = cwd
            for _ in range(5):  # Go up max 5 levels
                marker_path = os.path.join(current, root_marker)
                if os.path.exists(marker_path):
                    return current
                parent = os.path.dirname(current)
                if parent == current:  # Reached root
                    break
                current = parent
        
        # Fallback to current working directory
        return cwd


class PCISegmenter(BaseDataProcessor):
    """
    Processor for PCI segmentation.
    
    Assigns PCI values to road segments using ground truth data.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize PCI segmenter.
        
        Args:
            config: Configuration dictionary with keys:
                - algorithm: 'dijkstra', 'nearest_neighbor', etc.
                - pci_values: List of valid PCI values
        """
        super().__init__("pci_segmenter", config)
        self.supported_data_types = ['road_mask', 'ground_truth_pci']
    
    def _process_impl(self, data: DataContainer, config: Dict[str, Any]) -> DataContainer:
        """
        Segment roads with PCI values.
        
        Args:
            data: Input data container (road mask or ground truth)
            config: Processing configuration
            
        Returns:
            DataContainer with PCI-segmented roads
        """
        algorithm = config.get('algorithm', 'dijkstra')
        
        # TODO: Implement PCI segmentation logic
        # This is a template - replace with actual implementation
        # Example:
        # if algorithm == 'dijkstra':
        #     pci_segmented = dijkstra_segmentation(data.data, config)
        # else:
        #     pci_segmented = nearest_neighbor_segmentation(data.data, config)
        
        # Placeholder: create dummy segmentation
        if isinstance(data.data, np.ndarray):
            pci_segmented = data.data.copy()
        else:
            pci_segmented = np.array([0])  # Placeholder
        
        metadata = {
            **data.metadata,
            'algorithm': algorithm,
            'processor': self.name,
        }
        
        return DataContainer(
            data=pci_segmented,
            metadata=metadata,
            data_type='pci_segmented'
        )


class DataPreprocessor(BaseDataProcessor):
    """
    Processor for data preprocessing.
    
    Handles normalization, augmentation, and preparation for neural networks.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize data preprocessor.
        
        Args:
            config: Configuration dictionary with keys:
                - normalize: bool, whether to normalize data
                - augmentation: bool, whether to apply augmentation
                - patch_size: Size of patches for neural network input
        """
        super().__init__("data_preprocessor", config)
        self.supported_data_types = ['hyperspectral', 'pci_segmented']
    
    def _process_impl(self, data: DataContainer, config: Dict[str, Any]) -> DataContainer:
        """
        Preprocess data for neural network training.
        
        Args:
            data: Input data container
            config: Processing configuration
            
        Returns:
            Preprocessed data container
        """
        normalize = config.get('normalize', True)
        augmentation = config.get('augmentation', False)
        patch_size = config.get('patch_size', (32, 32))
        
        processed_data = data.data.copy() if hasattr(data.data, 'copy') else data.data
        
        # TODO: Implement preprocessing logic
        # This is a template - replace with actual implementation
        # Example:
        # if normalize:
        #     processed_data = normalize_data(processed_data)
        # if augmentation:
        #     processed_data = apply_augmentation(processed_data)
        # processed_data = create_patches(processed_data, patch_size)
        
        metadata = {
            **data.metadata,
            'normalize': normalize,
            'augmentation': augmentation,
            'patch_size': patch_size,
            'processor': self.name,
        }
        
        return DataContainer(
            data=processed_data,
            metadata=metadata,
            data_type='preprocessed'
        )

