"""
Data processing modules for APA.

Provides implementations for ROI processing, road extraction,
PCI segmentation, and data preprocessing.
"""

from typing import Any, Dict, Optional, Tuple
import numpy as np
import os
import sys
import h5py
import pickle
from pathlib import Path
from scipy.interpolate import griddata

from apa.common import (
    BaseDataProcessor,
    DataContainer,
    ProcessingResult,
    DataError,
    ValidationError,
)

# Import Dataset enum from centralized location
# Add project root to path if not already there
_project_root = Path(__file__).parent.parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

try:
    from enums.datasets_enum import Dataset
except ImportError:
    # Fallback: try direct path import
    try:
        import importlib.util
        enum_path = _project_root / "enums" / "datasets_enum.py"
        spec = importlib.util.spec_from_file_location("datasets_enum", enum_path)
        datasets_enum = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(datasets_enum)
        Dataset = datasets_enum.Dataset
    except Exception as e:
        raise ImportError(f"Failed to import Dataset enum from {enum_path}: {e}")

# Import pc_utils from local utils module
# This module-level import makes pc_utils available to ALL processors in this file:
# - ROIProcessor: uses normalize_hypersepctral_bands
# - RoadExtractor: has access (currently unused, but available if needed)
# - PCISegmenter: uses many functions (merge_close_points, create_proximity_mask, morphological_operator, etc.)
# - DataPreprocessor: uses normalize_hypersepctral_bands and process_labeled_image
# Try to import the full module to access all functions
HAS_PC_UTILS = False
pc_utils = None

try:
    import sys
    # Get project root - handle case where __file__ might not be available (e.g., in some import contexts)
    try:
        _project_root_pc = Path(__file__).parent.parent.parent.parent
    except NameError:
        # __file__ not available, try to find project root from current working directory
        _project_root_pc = Path.cwd()
        # Look for src directory
        if not (_project_root_pc / "src" / "apa" / "utils" / "point_cloud_utils.py").exists():
            # Try going up directories
            for _ in range(5):
                if (_project_root_pc / "src" / "apa" / "utils" / "point_cloud_utils.py").exists():
                    break
                _project_root_pc = _project_root_pc.parent
    
    pc_utils_path = _project_root_pc / "src" / "apa" / "utils" / "point_cloud_utils.py"
    
    if pc_utils_path.exists():
        try:
            import importlib.util
            spec = importlib.util.spec_from_file_location("point_cloud_utils", pc_utils_path)
            pc_utils_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(pc_utils_module)
            pc_utils = pc_utils_module
            HAS_PC_UTILS = True
        except Exception as import_error:
            # Module exists but failed to import (likely missing dependencies like scipy)
            # Try package import as fallback
            try:
                from apa.utils import point_cloud_utils as pc_utils
                HAS_PC_UTILS = True
            except ImportError:
                # Package import also failed, try minimal imports
                try:
                    from apa.utils.point_cloud_utils import (
                        normalize_hypersepctral_bands,
                        merge_close_points,
                        is_roi_within_bounds,
                    )
                    HAS_PC_UTILS = True
                    # Create minimal wrapper
                    class PCUtils:
                        normalize_hypersepctral_bands = normalize_hypersepctral_bands
                        merge_close_points = merge_close_points
                        is_roi_within_bounds = is_roi_within_bounds
                    pc_utils = PCUtils()
                except ImportError:
                    HAS_PC_UTILS = False
                    pc_utils = None
    else:
        # File doesn't exist, try package import
        try:
            from apa.utils import point_cloud_utils as pc_utils
            HAS_PC_UTILS = True
        except ImportError:
            # Try minimal imports
            try:
                from apa.utils.point_cloud_utils import (
                    normalize_hypersepctral_bands,
                    merge_close_points,
                    is_roi_within_bounds,
                )
                HAS_PC_UTILS = True
                # Create minimal wrapper
                class PCUtils:
                    normalize_hypersepctral_bands = normalize_hypersepctral_bands
                    merge_close_points = merge_close_points
                    is_roi_within_bounds = is_roi_within_bounds
                pc_utils = PCUtils()
            except ImportError:
                HAS_PC_UTILS = False
                pc_utils = None
except Exception as e:
    # Last resort: try minimal imports
    try:
        from apa.utils.point_cloud_utils import (
            normalize_hypersepctral_bands,
            merge_close_points,
            is_roi_within_bounds,
        )
        HAS_PC_UTILS = True
        # Create minimal wrapper
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

# Import ground truth loader
try:
    from apa.utils.ground_truth_loader import (
        get_GT_xy_PCI,
        get_PCI_ROI,
    )
    HAS_GROUND_TRUTH_LOADER = True
except ImportError:
    HAS_GROUND_TRUTH_LOADER = False
    get_GT_xy_PCI = None
    get_PCI_ROI = None

# Import PrepareDataForNN_module (pp)
try:
    import sys
    _project_root_pp = Path(__file__).parent.parent.parent.parent
    pp_path = _project_root_pp / "src" / "apa" / "utils" / "PrepareDataForNN_module.py"
    if pp_path.exists():
        import importlib.util
        spec = importlib.util.spec_from_file_location("PrepareDataForNN_module", pp_path)
        pp = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(pp)
        HAS_PP = True
    else:
        HAS_PP = False
        pp = None
except Exception:
    HAS_PP = False
    pp = None

# Import GetOptimalRoadOffset
try:
    from apa.geo_reference import GetOptimalRoadOffset
    HAS_GET_OPTIMAL_OFFSET = True
except ImportError:
    try:
        from src.geo_reference import GetOptimalRoadOffset
        HAS_GET_OPTIMAL_OFFSET = True
    except ImportError:
        HAS_GET_OPTIMAL_OFFSET = False
        GetOptimalRoadOffset = None


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
        self.supported_data_types = ['road_mask', 'ground_truth_pci', 'pci_segmentation_input']
    
    def validate_input(self, data: DataContainer) -> bool:
        """
        Validate input data for PCI segmentation.
        
        For 'pci_segmentation_input' type, validates that the dictionary contains
        the required data containers: 'hyperspectral_data', 'ground_truth_data', 'road_data'.
        
        Args:
            data: Input data container to validate
            
        Returns:
            True if valid
            
        Raises:
            ValidationError if invalid
        """
        # Call base validation first
        super().validate_input(data)
        
        # Additional validation for pci_segmentation_input type
        if data.data_type == 'pci_segmentation_input':
            if not isinstance(data.data, dict):
                raise ValidationError(
                    "For 'pci_segmentation_input' type, data must be a dictionary with "
                    "'hyperspectral_data', 'ground_truth_data', and 'road_data' keys"
                )
            
            required_keys = ['hyperspectral_data', 'ground_truth_data', 'road_data']
            missing_keys = [key for key in required_keys if key not in data.data]
            if missing_keys:
                raise ValidationError(
                    f"Missing required keys in pci_segmentation_input: {missing_keys}. "
                    f"Required keys: {required_keys}"
                )
            
            # Validate that each value is a DataContainer
            for key in required_keys:
                value = data.data[key]
                if not isinstance(value, DataContainer):
                    raise ValidationError(
                        f"Value for '{key}' must be a DataContainer, got {type(value)}"
                    )
        
        return True
    
    def _process_impl(self, data: DataContainer, config: Dict[str, Any]) -> DataContainer:
        """
        Segment roads with PCI values using ground truth data.
        
        Main function for doing geo-reference between PCI data and HSI images.
        Creates segmented mask based on PCI values and applies enhancements.
        
        Args:
            data: Input data container - expects dictionary with:
                - 'hyperspectral_data': DataContainer with 'image', 'lon_mat', 'lat_mat' (from ROIProcessor)
                - 'ground_truth_data': DataContainer with PCI data (from GroundTruthDataImporter)
                - 'road_data': DataContainer with road mask (from RoadExtractor)
            config: Full configuration with 'data' and 'preprocessing' sections
            
        Returns:
            DataContainer with PCI-segmented roads and processed data
        """
        # Validate required modules
        if not HAS_PC_UTILS or pc_utils is None:
            raise DataError(
                "pc_utils module is required for PCI segmentation. "
                "The point_cloud_utils.py file exists but could not be imported. "
                "This is likely due to missing dependencies (e.g., scipy). "
                "Please install required dependencies: pip install scipy"
            )
        if not HAS_GROUND_TRUTH_LOADER:
            raise DataError("ground_truth_loader module is required for PCI segmentation")
        if not HAS_ROAD_MASK_LOADER:
            raise DataError("road_mask_loader module is required for PCI segmentation")
        
        # Extract config sections
        if 'data' in config:
            data_config = config['data']
        else:
            data_config = config
        
        if 'preprocessing' in config:
            preprocessing_config = config['preprocessing']
        else:
            preprocessing_config = config.get('preprocessing', {})
        
        # Get inputs from data container
        # Expect data to be a dictionary with keys: 'hyperspectral_data', 'ground_truth_data', 'road_data'
        if isinstance(data.data, dict):
            hyperspectral_data = data.data.get('hyperspectral_data')
            ground_truth_data = data.data.get('ground_truth_data')
            road_data = data.data.get('road_data')
        else:
            # Try to get from metadata or assume data is the hyperspectral data
            hyperspectral_data = data if isinstance(data, DataContainer) else None
            ground_truth_data = data.metadata.get('ground_truth_data')
            road_data = data.metadata.get('road_data')
        
        if hyperspectral_data is None or not isinstance(hyperspectral_data, DataContainer):
            raise DataError("hyperspectral_data DataContainer is required")
        if ground_truth_data is None or not isinstance(ground_truth_data, DataContainer):
            raise DataError("ground_truth_data DataContainer is required")
        if road_data is None or not isinstance(road_data, DataContainer):
            raise DataError("road_data DataContainer is required")
        
        # Extract data from containers
        hys_data_dict = hyperspectral_data.data if isinstance(hyperspectral_data.data, dict) else {}
        image = hys_data_dict.get('image')
        lon_mat = hys_data_dict.get('lon_mat')
        lat_mat = hys_data_dict.get('lat_mat')
        
        if image is None or lon_mat is None or lat_mat is None:
            raise DataError("hyperspectral_data must contain 'image', 'lon_mat', 'lat_mat'")
        
        # Get ground truth PCI data
        gt_data = ground_truth_data.data
        if isinstance(gt_data, dict):
            # Ground truth data is stored as separate arrays: 'pci', 'lon', 'lat', 'seg_id'
            pci_vec = gt_data.get('pci')  # 1D array of PCI values
            lon_vec = gt_data.get('lon')  # 1D array of longitude values
            lat_vec = gt_data.get('lat')  # 1D array of latitude values
            seg_id = gt_data.get('seg_id', [])
            
            # Combine into 2D array with shape (N, 3) where each row is [lon, lat, pci]
            if pci_vec is not None and lon_vec is not None and lat_vec is not None:
                # Ensure all arrays have the same length
                min_len = min(len(pci_vec), len(lon_vec), len(lat_vec))
                if min_len > 0:
                    points_PCI = np.column_stack([
                        lon_vec[:min_len],
                        lat_vec[:min_len],
                        pci_vec[:min_len]
                    ])
                else:
                    points_PCI = np.array([]).reshape(0, 3)
            else:
                points_PCI = np.array([]).reshape(0, 3)
        else:
            # Assume it's already a 2D array with shape (N, 3) containing [lon, lat, pci]
            points_PCI = gt_data if isinstance(gt_data, np.ndarray) else np.array([]).reshape(0, 3)
            seg_id = []
        
        # Get road mask
        coinciding_mask = road_data.data if isinstance(road_data.data, np.ndarray) else None
        if coinciding_mask is None:
            raise DataError("road_data must contain numpy array mask")
        
        # Get ROI from config or metadata
        roi = None
        if 'rois' in data_config and data_config['rois']:
            roi = data_config['rois'][0]
        elif 'roi_bounds' in config:
            roi = config['roi_bounds']
        elif 'roi_bounds' in hyperspectral_data.metadata:
            roi = hyperspectral_data.metadata['roi_bounds']
        else:
            # Calculate from coordinates
            roi = [
                np.min(lon_mat), np.max(lon_mat),
                np.min(lat_mat), np.max(lat_mat)
            ]
        
        # Get excel_path for ground truth (if needed for reloading)
        excel_path = data_config.get('label_file') or data_config.get('PCI_path', '')
        
        # Get enum_data_source
        enum_data_source = data_config.get('enum_data_source', 1)
        
        # Get cropped data from ROIProcessor output
        # ROIProcessor should have already cropped and created RGB_enhanced
        if 'cropped_rect' in hyperspectral_data.metadata:
            # Data already cropped by ROIProcessor
            X_cropped = lat_mat
            Y_cropped = lon_mat
            hys_img = image
            RGB_enhanced = hys_data_dict.get('RGB_enhanced')
            cropped_rect = hyperspectral_data.metadata['cropped_rect']
            
            if RGB_enhanced is None:
                raise DataError("RGB_enhanced not found in hyperspectral_data. "
                              "Please ensure data has been processed by ROIProcessor first.")
        else:
            raise DataError("cropped_rect not found in hyperspectral_data metadata. "
                          "Please ensure data has been processed by ROIProcessor first.")
        
        # Process PCI segmentation
        X_cropped, Y_cropped, hys_img, points_merge_PCI, coinciding_mask, segment_mask, lut = \
            self._process_pci_segmentation(
                config, X_cropped, Y_cropped, hys_img, RGB_enhanced, coinciding_mask,
                points_PCI, seg_id, roi, enum_data_source, cropped_rect
            )
        
        # Note: Segment processing and saving is now handled by DataPreprocessor
        # Do not call _process_and_save_segments here
        
        # Prepare output data
        output_data = {
            'segment_mask': segment_mask,
            'coinciding_mask': coinciding_mask,
            'points_merge_PCI': points_merge_PCI,
            'X_cropped': X_cropped,
            'Y_cropped': Y_cropped,
            'hys_img': hys_img,
            'lut': lut,
        }
        
        metadata = {
            **data.metadata,
            'algorithm': 'dijkstra' if len(seg_id) > 0 else 'proximity',
            'processor': self.name,
            'roi': roi,
            'cropped_rect': cropped_rect,
            'enum_data_source': enum_data_source,
        }
        
        return DataContainer(
            data=output_data,
            metadata=metadata,
            data_type='pci_segmented'
        )
    
    def _process_pci_segmentation(self, config: Dict[str, Any], X_cropped: np.ndarray, Y_cropped: np.ndarray,
                                  hys_img: np.ndarray, RGB_enhanced: np.ndarray, coinciding_mask: np.ndarray,
                                  points_PCI: np.ndarray, ROI_seg: np.ndarray, roi: list,
                                  enum_data_source: int, cropped_rect: Tuple[int, int, int, int]) -> Tuple:
        """
        Process PCI segmentation logic.
        
        Creates segmented mask based on PCI data using either proximity mask or Dijkstra merging.
        
        Args:
            config: Full configuration
            X_cropped: Cropped latitude matrix
            Y_cropped: Cropped longitude matrix
            hys_img: Cropped hyperspectral image
            RGB_enhanced: Enhanced RGB image
            coinciding_mask: Road mask from OpenStreetMap
            points_PCI: PCI point data as 2D array with shape (N, 3) where each row is [lon, lat, pci]
            ROI_seg: Segment IDs for ROI points
            roi: ROI bounds
            enum_data_source: Dataset enum value
            cropped_rect: Crop rectangle (x_ind_min, y_ind_min, x_ind_max, y_ind_max)
        
        Returns:
            Tuple of (X_cropped, Y_cropped, hys_img, points_merge_PCI, coinciding_mask, segment_mask, lut)
        """
        lut = None
        
        # Merge close PCI points
        # points_PCI should be a 2D array with shape (N, 3) where each row is [lon, lat, pci]
        if points_PCI is not None and len(points_PCI) > 0:
            if HAS_PC_UTILS and hasattr(pc_utils, 'merge_close_points'):
                points_merge_PCI = pc_utils.merge_close_points(
                    points_PCI[:, :2], points_PCI[:, 2], 50e-5
                )
            else:
                points_merge_PCI = points_PCI
            xy_points_merge = points_merge_PCI[:, :2]
        else:
            points_merge_PCI = np.array([]).reshape(0, 3)
            xy_points_merge = np.array([]).reshape(0, 2)
        
        # Determine segmentation method based on segment IDs
        if len(ROI_seg) == 0:
            # No segment IDs - use proximity mask method
            print("No segment IDs found - using proximity mask method")
            
            # Create proximity mask
            if HAS_PC_UTILS and hasattr(pc_utils, 'create_proximity_mask'):
                extended_mask = pc_utils.create_proximity_mask(xy_points_merge, X_cropped, Y_cropped)
            else:
                # Fallback: create simple distance-based mask
                extended_mask = self._create_proximity_mask_fallback(xy_points_merge, X_cropped, Y_cropped)
            
            # Apply morphological operators
            if HAS_PC_UTILS and hasattr(pc_utils, 'morphological_operator'):
                combine_mask_roads = pc_utils.morphological_operator(
                    extended_mask, 'dilation', 'square', 1
                ) * coinciding_mask
                combine_mask_roads = pc_utils.morphological_operator(
                    combine_mask_roads, 'closing', 'disk', 3
                )
            else:
                combine_mask_roads = extended_mask * coinciding_mask
            
            # Interpolate PCI values to grid
            if len(points_PCI) > 0 and points_PCI.shape[1] >= 3:
                grid_value = griddata(
                    points_PCI[:, :2], points_PCI[:, 2],
                    (X_cropped, Y_cropped), method='nearest'
                )
                classified_roads_mask = grid_value * combine_mask_roads
            else:
                classified_roads_mask = combine_mask_roads
        else:
            # Has segment IDs - use Dijkstra method
            print("Segment IDs found - using Dijkstra merging method")
            
            # Get Dijkstra mask path
            preprocessing_config = config.get('preprocessing', {})
            georef_config = preprocessing_config.get('georeferencing', {})
            
            if 'dijkstra_map_mask_path' in georef_config:
                npz_filename = georef_config['dijkstra_map_mask_path']
            else:
                npz_filename = 'data/Detroit/masks_OpenStreetMap/Detroit_dijkstra_roads_mask.npz'
            
            # Add enum_data_source suffix
            if npz_filename.endswith('.npz'):
                npz_filename = npz_filename[:-4] + str(enum_data_source) + '.npz'
            else:
                npz_filename = npz_filename + str(enum_data_source) + '.npz'
            
            # Get full path
            repo_root = self._get_repo_root()
            npz_filename = os.path.join(repo_root, npz_filename)
            
            # Merge points using Dijkstra
            if HAS_PC_UTILS and hasattr(pc_utils, 'merge_points_dijkstra'):
                merge_points_dijkstra, lut = pc_utils.merge_points_dijkstra(
                    npz_filename, X_cropped, Y_cropped, coinciding_mask,
                    points_PCI, ROI_seg
                )
                classified_roads_mask = merge_points_dijkstra
            else:
                raise DataError("merge_points_dijkstra function not available in pc_utils")
        
        # Apply enhancement based on thresholds
        preprocessing_config = config.get('preprocessing', {})
        wt = preprocessing_config.get("white_threshold", None)
        gyt = preprocessing_config.get("gray_threshold", None)
        gdt = preprocessing_config.get("grad_threshold", None)
        
        if all(x is None for x in (wt, gyt, gdt)):
            segment_mask = classified_roads_mask
        else:
            # Apply enhancement
            title_dict = {"wt": wt, "gyt": gyt, "gdt": gdt}
            print(f"Applying enhancement with thresholds: {title_dict}")
            
            enhance_morph_operator_type = preprocessing_config.get("enhance_morph_operator_type", "dilation")
            enhance_morph_operator_size = preprocessing_config.get("enhance_morph_operator_size", 10)
            
            if enhance_morph_operator_type is not None and enhance_morph_operator_size > 0:
                if HAS_PC_UTILS and hasattr(pc_utils, 'morphological_operator_multiclass_mask'):
                    segment_mask_nan = self._nan_arr(
                        pc_utils.morphological_operator_multiclass_mask(
                            classified_roads_mask, enhance_morph_operator_type, 'square', enhance_morph_operator_size
                        )
                    )
                else:
                    segment_mask_nan = self._nan_arr(classified_roads_mask)
            else:
                segment_mask_nan = self._nan_arr(classified_roads_mask)
            
            if gyt >= 0:
                # Apply gray-based enhancement
                gray_color_enhanced, x_off, y_off = self._enhance_gray_based_on_RGB(
                    config, RGB_enhanced, segment_mask_nan
                )
                segment_mask_nan = self._nan_arr(gray_color_enhanced)
                
                # Apply gradient-based enhancement
                segment_mask_nan = self._enhance_mask_grad(
                    gdt, classified_roads_mask, RGB_enhanced, segment_mask_nan
                )
                segment_mask = segment_mask_nan
                
                # Apply offset to all variables
                ix_x_start = max(0, -x_off)
                ix_x_end = segment_mask.shape[0] - max(0, x_off)
                ix_y_start = max(0, -y_off)
                ix_y_end = segment_mask.shape[1] - max(0, y_off)
                
                X_cropped = X_cropped[ix_x_start:ix_x_end, ix_y_start:ix_y_end]
                Y_cropped = Y_cropped[ix_x_start:ix_x_end, ix_y_start:ix_y_end]
                hys_img = hys_img[ix_x_start:ix_x_end, ix_y_start:ix_y_end, :]
                coinciding_mask = coinciding_mask[ix_x_start:ix_x_end, ix_y_start:ix_y_end]
                segment_mask = segment_mask[ix_x_start:ix_x_end, ix_y_start:ix_y_end]
            else:
                segment_mask = segment_mask_nan
        
        return X_cropped, Y_cropped, hys_img, points_merge_PCI, coinciding_mask, segment_mask, lut
    
    def _enhance_mask_grad(self, grad_threshold: float, classified_roads_mask: np.ndarray,
                          RGB_enhanced: np.ndarray, segment_mask_nan: np.ndarray) -> np.ndarray:
        """
        Enhance mask by removing outliers based on gradient magnitude.
        
        Uses Sobel gradient over Y channel to detect objects and remove them from mask.
        
        Args:
            grad_threshold: Gradient threshold for object detection
            classified_roads_mask: Original classified roads mask
            RGB_enhanced: Enhanced RGB image
            segment_mask_nan: Current segment mask with NaN values
        
        Returns:
            Enhanced segment mask with objects removed
        """
        if not HAS_PC_UTILS:
            return segment_mask_nan
        
        # Convert RGB to YUV and get Y channel
        if hasattr(pc_utils, 'rgb_to_yuv'):
            Y, _, _ = pc_utils.rgb_to_yuv(RGB_enhanced)
        else:
            # Fallback: use grayscale
            Y = np.mean(RGB_enhanced, axis=2)
        
        # Apply mean filter and Sobel gradient
        if hasattr(pc_utils, 'mean_filter') and hasattr(pc_utils, 'sobel_gradient'):
            _, _, mag = pc_utils.sobel_gradient(pc_utils.mean_filter(Y, 3))
        else:
            # Fallback: simple gradient
            from scipy import ndimage
            mag = np.abs(ndimage.sobel(Y))
        
        # Detect objects using gradient threshold
        if hasattr(pc_utils, 'morphological_operator_multiclass_mask'):
            dilated_mask = pc_utils.morphological_operator_multiclass_mask(
                classified_roads_mask, 'dilation', 'square', 3
            )
        else:
            dilated_mask = classified_roads_mask
        
        objects_detected_im_mask = np.where(dilated_mask > 0, mag, 0) > grad_threshold
        
        # Remove detected objects from mask
        segment_mask_obj_removed = np.where(objects_detected_im_mask, np.nan, segment_mask_nan)
        
        print(
            f'After object detection, Not nan pixels in new mask: {np.sum(~np.isnan(segment_mask_obj_removed))} pixels\n'
            f'Removed {100 * (1 - np.sum(~np.isnan(segment_mask_obj_removed)) / np.sum(~np.isnan(segment_mask_nan))):.2f}% '
            f'of pixels from original mask\n\n'
        )
        
        return segment_mask_obj_removed
    
    def _enhance_gray_based_on_RGB(self, config: Dict[str, Any], RGB_enhanced: np.ndarray,
                                   dilated_mask: np.ndarray) -> Tuple[np.ndarray, int, int]:
        """
        Enhance mask based on gray color detection in RGB image.
        
        Identifies gray (asphalt) regions and computes spatial offset using cross-correlation.
        
        Args:
            config: Configuration dictionary
            RGB_enhanced: Enhanced RGB image
            dilated_mask: Dilated segment mask
        
        Returns:
            Tuple of (combined_mask, x_offset, y_offset)
        """
        preprocessing_config = config.get('preprocessing', {})
        gray_threshold = preprocessing_config.get("gray_threshold", 0.1)
        white_threshold = preprocessing_config.get("white_threshold", 0.92)
        
        # Calculate differences between RGB channels
        diff_rg = np.abs(RGB_enhanced[:, :, 0] - RGB_enhanced[:, :, 1])  # Red vs Green
        diff_rb = np.abs(RGB_enhanced[:, :, 0] - RGB_enhanced[:, :, 2])  # Red vs Blue
        diff_gb = np.abs(RGB_enhanced[:, :, 1] - RGB_enhanced[:, :, 2])  # Green vs Blue
        
        # Identify gray regions (all channel differences within tolerance)
        gray_color_mask = (diff_rg <= gray_threshold) & (diff_rb <= gray_threshold) & (diff_gb <= gray_threshold)
        
        # Identify gray but not white regions
        intensity = np.mean(RGB_enhanced, axis=2)
        not_white_mask = intensity < white_threshold
        gray_but_not_white_mask = gray_color_mask & not_white_mask
        
        # Compute spatial offset using cross-correlation
        georef_config = preprocessing_config.get('georeferencing', {})
        int_max_shift = georef_config.get("max_reg_offset", 20)
        
        if HAS_GET_OPTIMAL_OFFSET and GetOptimalRoadOffset:
            (x_off, y_off) = GetOptimalRoadOffset.find_local_road_offset_from_arrays(
                ~np.isnan(dilated_mask),
                gray_but_not_white_mask,
                max_shift=int_max_shift
            )
        else:
            # Fallback: no offset
            x_off, y_off = 0, 0
            print("Warning: GetOptimalRoadOffset not available, using zero offset")
        
        # Apply offset to mask
        bin_offset_dialeted_mask = np.roll(~np.isnan(dilated_mask), shift=(y_off, x_off), axis=(0, 1))
        combined_mask = np.where(bin_offset_dialeted_mask & gray_but_not_white_mask, dilated_mask, 0)
        
        return combined_mask, x_off, y_off
    
    def _create_proximity_mask_fallback(self, xy_points: np.ndarray, X_cropped: np.ndarray,
                                       Y_cropped: np.ndarray) -> np.ndarray:
        """
        Fallback proximity mask creation if pc_utils.create_proximity_mask is not available.
        
        Args:
            xy_points: Point coordinates (N, 2)
            X_cropped: Latitude matrix
            Y_cropped: Longitude matrix
        
        Returns:
            Proximity mask array
        """
        if len(xy_points) == 0:
            return np.zeros(X_cropped.shape, dtype=bool)
        
        # Create distance-based mask
        mask = np.zeros(X_cropped.shape, dtype=bool)
        threshold = 50e-5  # Default threshold
        
        for point in xy_points:
            lon, lat = point
            distances = np.sqrt((X_cropped - lat)**2 + (Y_cropped - lon)**2)
            mask |= distances < threshold
        
        return mask.astype(float)
    
    def _nan_arr(self, arr: np.ndarray) -> np.ndarray:
        """
        Convert array to NaN array format (helper for pc_utils.nan_arr compatibility).
        
        Args:
            arr: Input array
        
        Returns:
            Array with same values (or NaN conversion if needed)
        """
        if HAS_PC_UTILS and hasattr(pc_utils, 'nan_arr'):
            return pc_utils.nan_arr(arr)
        else:
            # Fallback: return as-is
            return arr
    
    
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


class DataPreprocessor(BaseDataProcessor):
    """
    Processor for data preprocessing.
    
    Handles normalization, augmentation, and preparation for neural networks.
    Also processes PCI segments and saves them to HDF5 files.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize data preprocessor.
        
        Args:
            config: Configuration dictionary with keys:
                - normalize: bool, whether to normalize data
                - augmentation: bool, whether to apply augmentation
                - patch_size: Size of patches for neural network input
                - save_segments: bool, whether to save processed segments to HDF5
        """
        super().__init__("data_preprocessor", config)
        self.supported_data_types = ['hyperspectral', 'pci_segmented']
    
    def _process_impl(self, data: DataContainer, config: Dict[str, Any]) -> DataContainer:
        """
        Preprocess data for neural network training.
        
        Processes PCI-segmented data and saves segments to HDF5 files.
        
        Args:
            data: Input data container with PCI-segmented data (from PCISegmenter)
                Expected structure: dictionary with keys:
                - 'segment_mask': Segmented mask
                - 'hys_img': Cropped hyperspectral image
                - 'lut': Lookup table for segment IDs (optional)
            config: Full configuration with 'data', 'preprocessing', and 'cnn_model' sections
            
        Returns:
            Preprocessed data container
        """
        # Extract config sections
        if 'data' in config:
            data_config = config['data']
        else:
            data_config = config
        
        if 'preprocessing' in config:
            preprocessing_config = config['preprocessing']
        else:
            preprocessing_config = config.get('preprocessing', {})
        
        # Get input data
        if not isinstance(data.data, dict):
            raise DataError("DataPreprocessor expects data as dictionary with 'segment_mask', 'hys_img', etc.")
        
        segment_mask = data.data.get('segment_mask')
        hys_img = data.data.get('hys_img')
        segID_PCI_LUT = data.data.get('lut')
        
        if segment_mask is None or hys_img is None:
            raise DataError("DataPreprocessor requires 'segment_mask' and 'hys_img' in data dictionary")
        
        # Get metadata
        roi = data.metadata.get('roi')
        cropped_rect = data.metadata.get('cropped_rect')
        enum_data_source = data.metadata.get('enum_data_source', 1)
        
        if roi is None:
            raise DataError("ROI bounds not found in metadata")
        
        # Process and save segments
        self._process_and_save_segments(
            config, hys_img, segment_mask, segID_PCI_LUT, cropped_rect, roi, enum_data_source
        )
        
        # Apply normalization if requested
        normalize = preprocessing_config.get('normalize', True)
        if normalize:
            # Normalize hyperspectral image if needed
            if HAS_PC_UTILS and hasattr(pc_utils, 'normalize_hypersepctral_bands'):
                hys_img = pc_utils.normalize_hypersepctral_bands(hys_img)
        
        # Prepare output data
        processed_data = {
            'segment_mask': segment_mask,
            'hys_img': hys_img,
            'lut': segID_PCI_LUT,
        }
        
        metadata = {
            **data.metadata,
            'normalize': normalize,
            'processor': self.name,
            'segments_saved': True,
        }
        
        return DataContainer(
            data=processed_data,
            metadata=metadata,
            data_type=data.data_type
        )
    
    def _process_and_save_segments(self, config: Dict[str, Any], cropped_msp_img: np.ndarray,
                                   segment_mask: np.ndarray, segID_PCI_LUT: Optional[Dict],
                                   cropped_rect: Optional[Tuple[int, int, int, int]], roi: list,
                                   enum_data_source: int) -> None:
        """
        Process segments and save data to HDF5 files.
        
        This method handles all the segment processing logic including:
        - Creating/loading bounding box lists
        - Cropping images to segments
        - Normalizing masks
        - Saving to HDF5 files
        
        Args:
            config: Full configuration
            cropped_msp_img: Cropped hyperspectral image
            segment_mask: Segmented mask
            segID_PCI_LUT: Lookup table for segment IDs
            cropped_rect: Crop rectangle (optional)
            roi: ROI bounds
            enum_data_source: Dataset enum value
        """
        if not HAS_PP or pp is None:
            print("Warning: PrepareDataForNN_module not available, skipping segment processing")
            return
        
        preprocessing_config = config.get('preprocessing', {})
        
        # Segment ID filename
        strBoundingFilename = f"boudningbox_list_labeled_image_source_{enum_data_source}.h5"
        
        if segID_PCI_LUT is not None:
            mask_null_fill_value = preprocessing_config.get("mask_null_fill_value", 0)
            
            # Create mapping from keys to unique integers
            unique_key_map = {key: idx for idx, key in enumerate(segID_PCI_LUT.keys()) if not isinstance(key, int)}
            key_to_int_map = {int(key): int(segID_PCI_LUT[key]) for key in segID_PCI_LUT.keys()}
            
            # Convert dictionary to numpy array
            numerical_segID_PCI_LUT = np.array([(int(key), value) for key, value in segID_PCI_LUT.items()])
            
            filled_with = np.nan_to_num(segment_mask, nan=mask_null_fill_value)
            
            # Process labeled image
            if HAS_PC_UTILS and hasattr(pc_utils, 'process_labeled_image'):
                boudningbox_list_labeled_image = pc_utils.process_labeled_image(
                    cropped_msp_img, segment_mask, segID_PCI_LUT, dilation_radius=1
                )
            else:
                print("Warning: process_labeled_image not available, skipping")
                return
            
            # Replace mask values using lookup table
            replaced_mask = np.vectorize(key_to_int_map.get)(filled_with.astype('int'))
            segment_mask = np.where(replaced_mask == None, mask_null_fill_value, replaced_mask).astype('int')
            
            # Save bounding box list
            repo_root = self._get_repo_root()
            strBoundingFilename = os.path.join(repo_root, strBoundingFilename)
            with h5py.File(strBoundingFilename, "w") as f:
                f.create_dataset(strBoundingFilename, data=np.void(pickle.dumps(boudningbox_list_labeled_image)))
        else:
            # Load existing bounding box list
            repo_root = self._get_repo_root()
            strBoundingFilename = os.path.join(repo_root, strBoundingFilename)
            if os.path.exists(strBoundingFilename):
                with h5py.File(strBoundingFilename, "r") as f:
                    boudningbox_list_labeled_image = pickle.loads(bytes(f[strBoundingFilename][()]))
            else:
                print(f"Warning: Bounding box file not found: {strBoundingFilename}")
                return
        
        # Analyze pixel value ranges (if function available)
        # TODO: Import analyze_pixel_value_ranges if available
        
        # Get spectral bands info (if function available)
        # TODO: Import get_spectral_bands if available
        
        # Create binary segment mask
        binary_seg_mask = (segment_mask > 0) * 1
        road_hys_filter = np.reshape(binary_seg_mask, list(segment_mask.shape) + [1])
        
        # Get roads in general
        num_of_channels = cropped_msp_img.shape[-1]
        hys_roads = np.repeat(road_hys_filter, num_of_channels, -1) * cropped_msp_img
        
        # Crop image to segments
        NN_inputs = pp.crop_image_to_segments(config, hys_roads, image_dim=num_of_channels)
        NN_inputs[np.isnan(NN_inputs)] = 0
        
        # Get only labeled roads
        labeled_road_mask = np.ones(binary_seg_mask.shape)
        labeled_road_mask[np.isnan(segment_mask)] = 0
        labeled_road_mask = np.reshape(labeled_road_mask * binary_seg_mask, list(labeled_road_mask.shape) + [1])
        hys_labeled_roads = np.repeat(labeled_road_mask, num_of_channels, -1) * cropped_msp_img
        NN_labeled_inputs = pp.crop_image_to_segments(config, hys_labeled_roads, image_dim=num_of_channels)
        NN_labeled_inputs[np.isnan(NN_labeled_inputs)] = 0
        
        # Create true labels
        true_labels_full_image = np.reshape(segment_mask, list(segment_mask.shape) + [1]) * labeled_road_mask
        true_labels_full_image[np.isnan(true_labels_full_image)] = 0
        true_labels = pp.crop_image_to_segments(config, true_labels_full_image, image_dim=1)
        
        # Remove frames with zeros only
        non_zero_idx = np.argwhere(np.sum(np.sum(np.sum(true_labels, -1), -1), -1) > 0)
        fin_NN_inputs = NN_inputs[non_zero_idx[:, 0], :, :, :]
        fin_true_labels = true_labels[non_zero_idx[:, 0], :, :, :]
        fin_NN_labeled_inputs = NN_labeled_inputs[non_zero_idx[:, 0], :, :, :]
        
        # Save data
        data_config = config.get('data', {})
        output_dirname = data_config.get('output_path', 'preprocessed_database')
        output_path = Path(output_dirname)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Create formatted string for filenames
        formatted_string = "_".join(map(lambda x: str(round(x)), roi)) + \
                          "_DataSource_" + str(enum_data_source)
        
        # Get crop size from config
        cnn_config = config.get('cnn_model', {})
        crop_size = cnn_config.get('input_shape', [64, 64, 12])[0]  # Assume symmetric
        
        # Normalize masks
        normalized_masks = []
        normalized_masks_labels = []
        
        for i in range(len(boudningbox_list_labeled_image)):
            # Replace nans with zeros
            nan_idx = np.isnan(boudningbox_list_labeled_image[i]['mask'])
            boudningbox_list_labeled_image[i]['mask'][nan_idx] = 0
            
            normalized_masks.append(
                pp.normalize_mask(boudningbox_list_labeled_image[i]['mask'], crop_size)
            )
            
            label_mat = np.zeros(normalized_masks[i].shape[0:2])
            label_mat[normalized_masks[i][:, :, 0] != 0] = boudningbox_list_labeled_image[i]['label']
            normalized_masks_labels.append(label_mat.reshape(list(label_mat.shape) + [1]))
        
        normalized_masks_labels = np.asarray(normalized_masks_labels)
        normalized_masks = np.asarray(normalized_masks)
        
        # Save segment data
        pp.save_cropped_segments_to_h5(normalized_masks, output_path / f"BoudingBoxList{formatted_string}.h5")
        pp.save_cropped_segments_to_h5(normalized_masks_labels, output_path / f"BoudingBoxLabel{formatted_string}.h5")
        
        # Save old format data
        pp.save_cropped_segments_to_h5(fin_NN_inputs, output_path / f"All_Road_{formatted_string}.h5")
        pp.save_cropped_segments_to_h5(fin_true_labels, output_path / f"PCI_labels_{formatted_string}.h5")
        pp.save_cropped_segments_to_h5(fin_NN_labeled_inputs, output_path / f"Labeld_Roads_{formatted_string}.h5")
        
        print('Segment processing and saving completed.')
    
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

