"""
Data importer modules for APA.

Provides implementations for importing hyperspectral imagery,
ground truth PCI data, and road network data from various sources.
"""

from typing import Any, Dict, Optional, Tuple, List
from pathlib import Path
import numpy as np
import h5py
import os

from apa.common import (
    BaseModule,
    DataContainer,
    DataError,
    ValidationError,
)
from apa.common.interfaces import DataInterface
try:
    from enums.datasets_enum import Dataset
except ImportError:
    # Fallback if enums module is not available
    from enum import Enum
    class Dataset(Enum):
        venus_IL = 0
        venus_Detroit = 1
        airbus_HSP_Detroit = 2
        airbus_Pan_Detroit = 3

# Try to import required modules for data loading
try:
    import rasterio
    from rasterio.transform import xy
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False

try:
    import utm
    HAS_UTM = True
except ImportError:
    HAS_UTM = False

# Try to import custom modules - adjust paths as needed
try:
    from ImportVenusModule import getVenusData
    HAS_IMPORT_VENUS = True
except ImportError:
    HAS_IMPORT_VENUS = False
    getVenusData = None

# Import getLatLon_fromTiff from local utils module
try:
    from apa.utils.getLatLon_fromTiff import convert_raster_to_geocoords
    HAS_GET_LATLON = True
    # Store the function directly for use
    _convert_raster_to_geocoords_func = convert_raster_to_geocoords
except ImportError:
    HAS_GET_LATLON = False
    _convert_raster_to_geocoords_func = None

# Try to import CovertITM2LatLon from apa.geo_reference
try:
    from apa.geo_reference import CovertITM2LatLon
    HAS_COVERT_ITM = True
except ImportError:
    try:
        # Fallback: try direct import (if installed as package)
        import CovertITM2LatLon
        HAS_COVERT_ITM = True
    except ImportError:
        try:
            # Fallback: try src.geo_reference path
            from src.geo_reference import CovertITM2LatLon
            HAS_COVERT_ITM = True
        except ImportError:
            HAS_COVERT_ITM = False


class HyperspectralDataImporter(BaseModule, DataInterface):
    """
    Importer for hyperspectral satellite imagery.
    
    Supports multiple data sources:
    - VENUS (Israel - Kiryat Ata)
    - VENUS (Detroit)
    - Airbus HSP (Detroit)
    - Airbus Pan (Detroit)
    
    Returns DataContainer with data as a dictionary containing:
    - 'image': The hyperspectral image array (height, width, bands)
    - 'lon_mat': Longitude matrix (height, width)
    - 'lat_mat': Latitude matrix (height, width)
    
    Supports both flat and nested config structures:
    - Flat: config['input_path'], config['dataset']
    - Nested: config['data']['input_path'], config['data']['enum_data_source']
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize hyperspectral data importer.
        
        Args:
            config: Configuration dictionary with keys:
                - input_path: Path to data directory
                - filename_NED: Name of NED (hyperspectral) file
                - filename_RGB: Name of RGB file
                - dataset: Dataset enum value
        """
        super().__init__("hyperspectral_importer", config)
        self.required_config_keys = ['input_path']
        self.supported_datasets = [
            Dataset.venus_IL,
            Dataset.venus_Detroit,
            Dataset.airbus_HSP_Detroit,
            Dataset.airbus_Pan_Detroit,
        ]
    
    def _extract_config(self, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Extract configuration from either flat or nested structure.
        
        Args:
            config: Optional configuration (overrides instance config)
            
        Returns:
            Flattened configuration dictionary
        """
        load_config = {**self.config, **(config or {})}
        
        # Check if config has nested 'data' structure
        if 'data' in load_config and isinstance(load_config['data'], dict):
            data_config = load_config['data']
            # Merge data config into main config (data config takes precedence)
            flat_config = {**load_config, **data_config}
            # Remove the nested 'data' key
            flat_config.pop('data', None)
        else:
            flat_config = load_config
        
        return flat_config
    
    def load_data(self, config: Optional[Dict[str, Any]] = None) -> DataContainer:
        """
        Load hyperspectral imagery data.
        
        Returns DataContainer with data as a dictionary containing:
        - 'image': The hyperspectral image array
        - 'lon_mat': Longitude matrix
        - 'lat_mat': Latitude matrix
        
        Args:
            config: Optional configuration (overrides instance config)
                Supports both flat and nested structures (config['data'] dict)
            
        Returns:
            DataContainer with hyperspectral data as dict with keys: image, lon_mat, lat_mat
        """
        flat_config = self._extract_config(config)
        
        try:
            input_path = flat_config.get('input_path')
            if not input_path:
                raise DataError("input_path is required in configuration")
            
            # Get dataset enum - support both 'dataset' and 'enum_data_source'
            dataset_value = flat_config.get('enum_data_source') or flat_config.get('dataset')
            
            if dataset_value is None:
                dataset = Dataset.venus_Detroit  # Default
            elif isinstance(dataset_value, Dataset):
                dataset = dataset_value
            elif isinstance(dataset_value, int):
                # Convert int to Dataset enum
                dataset = Dataset(dataset_value)
            else:
                # Try to match by name
                dataset = Dataset[dataset_value] if isinstance(dataset_value, str) else Dataset.venus_Detroit
            
            # Load data based on dataset type
            if dataset == Dataset.venus_IL:
                data_dict = self._load_venus_israel(input_path, flat_config)
            elif dataset == Dataset.venus_Detroit:
                data_dict = self._load_venus_detroit(input_path, flat_config)
            elif dataset == Dataset.airbus_HSP_Detroit:
                data_dict = self._load_airbus_detroit(input_path, flat_config)
            elif dataset == Dataset.airbus_Pan_Detroit:
                data_dict = self._load_airbus_detroit(input_path, flat_config)
            else:
                raise DataError(f"Unsupported dataset: {dataset}")
            
            # Handle ROIs - from old importer logic
            if "rois" in flat_config:
                rois = flat_config["rois"]
            else:
                # Use all data as default ROI
                lat_mat = data_dict['lat_mat']
                lon_mat = data_dict['lon_mat']
                roi = [np.min(lat_mat), np.max(lat_mat), np.min(lon_mat), np.max(lon_mat)]
                rois = [roi]
            
            metadata = {
                'dataset': dataset.name if hasattr(dataset, 'name') else str(dataset),
                'input_path': input_path,
                'data_type': 'hyperspectral',
                'rois': rois,
                'filename_NED': flat_config.get('filename_NED', ''),
                'filename_RGB': flat_config.get('filename_RGB', ''),
            }
            
            return DataContainer(
                data=data_dict,  # Dictionary with image, lon_mat, lat_mat
                metadata=metadata,
                data_type='hyperspectral'
            )
        
        except Exception as e:
            raise DataError(f"Failed to load hyperspectral data: {str(e)}") from e
    
    def _load_venus_israel(self, input_path: str, config: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """Load VENUS Israel data."""
        # For now, use same logic as Detroit - can be customized later
        return self._load_venus_detroit(input_path, config)
    
    def _load_venus_detroit(self, input_path: str, config: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """
        Load VENUS Detroit data using get_multi_spectral_imaginery_Venus logic.
        
        Returns dictionary with keys: 'image', 'lon_mat', 'lat_mat'
        """
        data_filename = config.get('filename_NED', '')
        data_dirname = input_path
        config_data = config
        
        # Check if we should use the multi-file approach (big_tiff == False)
        zone = config_data.get('zone', 'Detroit')
        big_tiff = config_data.get('big_tiff', False)
        
        if zone == "Detroit" and big_tiff == False:
            # Use the multi-band file approach
            if not HAS_IMPORT_VENUS:
                raise DataError("ImportVenusModule is required. Please ensure it's available.")
            
            bands = range(1, 13)
            VenusImage_ls = []
            original_filename = data_filename
            
            for b in bands:
                # Modify filename for each band
                filename_parts = original_filename.split('.')
                num2replace = len(str(b-1))
                filename_parts[0] = filename_parts[0][:-num2replace] + str(b)
                band_filename = filename_parts[0] + '.' + filename_parts[1]
                
                VenusImage_band = getVenusData(data_dirname, band_filename)
                VenusImage_ls.append(VenusImage_band)
            
            VenusImage = np.asarray(VenusImage_ls)
            VenusImage = np.transpose(VenusImage, axes=(1, 2, 0))
            
            # Get lat/lon from the last band file (band 12)
            filename_parts = original_filename.split('.')
            num2replace = len(str(11))  # For band 12, previous band is 11
            filename_parts[0] = filename_parts[0][:-num2replace] + str(12)
            last_band_filename = filename_parts[0] + '.' + filename_parts[1]
            tiff_path = os.path.join(data_dirname, last_band_filename)
            
            # Convert raster to geocoords
            latlon_mat = self._convert_raster_to_geocoords(tiff_path, zone_number=17, zone_letter='T')
            
            # Unpack lat/lon
            lon_mat = latlon_mat[:, :, 0]
            lat_mat = latlon_mat[:, :, 1]
            
        else:
            # Use single file approach (big_tiff == True or other zones)
            # This would need to be implemented based on your specific file structure
            raise DataError("big_tiff=True or non-Detroit zones not yet implemented. Please use big_tiff=False for Detroit.")
        
        return {
            'image': VenusImage,
            'lon_mat': lon_mat,
            'lat_mat': lat_mat
        }
    
    def _try_path_locations(self, input_path: str, filename: str) -> Tuple[Optional[str], List[str]]:
        """Try multiple path resolution strategies and return the first existing file path."""
        tried_paths: List[str] = []
        if not filename:
            return None, tried_paths
        
        if input_path is None:
            input_path = ''
        
        def record_and_check(candidate: Optional[str]) -> Optional[str]:
            if not candidate:
                return None
            candidate_norm = os.path.normpath(candidate)
            if candidate_norm not in tried_paths:
                tried_paths.append(candidate_norm)
            if os.path.exists(candidate_norm):
                return candidate_norm
            return None
        
        # Strategy 1: If input_path is absolute, use it directly
        if input_path and os.path.isabs(input_path):
            result = record_and_check(os.path.join(input_path, filename))
            if result:
                return result, tried_paths
        
        # Strategy 2: Try relative to current working directory
        result = record_and_check(os.path.join(os.getcwd(), input_path, filename))
        if result:
            return result, tried_paths
        
        # Strategy 3: Try relative path as-is (relative to cwd)
        result = record_and_check(os.path.join(input_path, filename))
        if result:
            return result, tried_paths
        
        # Strategy 4: Try relative to the notebook/script location (if we can detect it)
        try:
            import inspect
            frame = inspect.currentframe()
            if frame and frame.f_back:
                caller_file = frame.f_back.f_globals.get('__file__', '')
                if caller_file:
                    caller_dir = os.path.dirname(os.path.abspath(caller_file))
                    candidate = os.path.join(caller_dir, '..', input_path, filename)
                    result = record_and_check(candidate)
                    if result:
                        return result, tried_paths
        except Exception:
            pass
        
        # Strategy 5: Try relative to project root (look for common markers)
        cwd = os.getcwd()
        for root_marker in ['setup.py', 'src', 'configs']:
            current = cwd
            for _ in range(5):  # Go up max 5 levels
                marker_path = os.path.join(current, root_marker)
                if os.path.exists(marker_path):
                    candidate = os.path.join(current, input_path, filename)
                    result = record_and_check(candidate)
                    if result:
                        return result, tried_paths
                parent = os.path.dirname(current)
                if parent == current:  # Reached root
                    break
                current = parent
        
        return None, tried_paths
    
    def _convert_raster_to_geocoords(self, file_path: str, zone_number: int = 36, zone_letter: str = 'U') -> np.ndarray:
        """
        Reads a raster file, extracts pixel data, converts pixel coordinates to geographic coordinates 
        (latitude and longitude), and returns the lat/lon matrix.
        
        Parameters:
        - file_path: str, the path to the raster file (TIF format).
        - zone_number: int, UTM zone number (default 36)
        - zone_letter: str, UTM zone letter (default 'U')
        
        Returns:
        - latlon_mat: numpy array, a matrix containing latitude and longitude coordinates for each pixel.
        """
        if not HAS_RASTERIO:
            raise DataError("rasterio is required for GeoTIFF loading. Install with: pip install rasterio")
        
        with rasterio.open(file_path) as src:
            # Read the data (assuming single band raster)
            data = src.read(1)
            
            # Extract metadata
            bounds = src.bounds
            transform = src.transform
            crs = src.crs
        
        # Generate row and column indices for the given data
        rows, cols = np.indices(data.shape)
        
        # Convert pixel coordinates to geospatial coordinates (easting and northing)
        xs, ys = xy(transform, rows.ravel(), cols.ravel())
        
        # Convert lists to numpy arrays for easier manipulation
        xs = np.array(xs).reshape(rows.shape)
        ys = np.array(ys).reshape(rows.shape)
        
        # Convert UTM coordinates to latitude and longitude
        if HAS_COVERT_ITM and CovertITM2LatLon:
            # Use custom conversion module if available
            lat_mat, lon_mat = CovertITM2LatLon.UTM2WGS(xs, ys, zone_number, zone_letter)
        elif HAS_UTM:
            # Use utm library as fallback
            lat_mat = np.zeros_like(xs)
            lon_mat = np.zeros_like(ys)
            for i in range(xs.shape[0]):
                for j in range(xs.shape[1]):
                    lat, lon = utm.to_latlon(xs[i, j], ys[i, j], zone_number, zone_letter)
                    lat_mat[i, j] = lat
                    lon_mat[i, j] = lon
        else:
            raise DataError("Either CovertITM2LatLon module or utm library is required for coordinate conversion.")
        
        # Reshape latitude and longitude matrices to match the shape of the input data
        lat_mat = np.array(lat_mat).reshape(list(np.shape(xs)) + [1])
        lon_mat = np.array(lon_mat).reshape(list(np.shape(ys)) + [1])
        
        # Combine latitude and longitude into a single matrix
        latlon_mat = np.append(lat_mat, lon_mat, -1)
        
        return latlon_mat
    
    def _load_airbus_detroit(self, input_path: str, config: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """
        Load Airbus Detroit data (HSP or Pan).
        
        Returns dictionary with keys: 'image', 'lon_mat', 'lat_mat'
        """
        filename = config.get('filename_NED', config.get('filename_HSP', ''))
        if not filename:
            raise DataError("filename_NED or filename_HSP is required for Airbus data")
        
        # Check for both NED and RGB filenames
        filename_NED = config.get('filename_NED', '')
        filename_RGB = config.get('filename_RGB', '')
        
        # Determine which files to load (only non-empty filenames)
        files_to_load = []
        if filename_NED and filename_NED.strip():
            files_to_load.append(('NED', filename_NED))
        if filename_RGB and filename_RGB.strip():
            files_to_load.append(('RGB', filename_RGB))
        
        if not files_to_load:
            # Fallback to original filename if no NED/RGB specified
            if filename and filename.strip():
                files_to_load = [('NED', filename)]
            else:
                raise DataError("At least one of filename_NED or filename_RGB is required for Airbus data")
        
        # Check if any file is a TIF file (GeoTIFF)
        is_geotiff = any(fname.lower().endswith(('.tif', '.tiff')) for _, fname in files_to_load)
        
        if is_geotiff:
            # Load all GeoTIFF files and concatenate along band dimension
            loaded_data = []
            ned_index = None
            
            for file_type, fname in files_to_load:
                filepath_file, tried_paths_file = self._try_path_locations(input_path, fname)
                
                if filepath_file:
                    data_dict = self._load_geotiff_airbus(filepath_file, config)
                    loaded_data.append(data_dict)
                    if file_type.upper() == 'NED':
                        ned_index = len(loaded_data) - 1
                else:
                    all_tried = tried_paths_file if tried_paths_file else [os.path.join(input_path, fname)]
                    raise DataError(
                        f"GeoTIFF file ({file_type}) not found: {fname}\n"
                        f"Tried paths:\n" + "\n".join(f"  - {p}" for p in all_tried)
                    )
            
            # Concatenate images along band dimension
            if len(loaded_data) == 1:
                return loaded_data[0]
            else:
                images = [data['image'] for data in loaded_data]
                spatial_shapes = [(img.shape[0], img.shape[1]) for img in images]
                if len(set(spatial_shapes)) > 1:
                    raise DataError(
                        f"Cannot concatenate images with different spatial dimensions: "
                        f"{spatial_shapes}"
                    )
                
                concatenated_image = np.concatenate(images, axis=2)
                
                if ned_index is not None and ned_index < len(loaded_data):
                    ned_data = loaded_data[ned_index]
                else:
                    ned_data = loaded_data[0]
                
                return {
                    'image': concatenated_image,
                    'lon_mat': ned_data['lon_mat'],
                    'lat_mat': ned_data['lat_mat'],
                }
        else:
            # Not GeoTIFF - handle HDF5 or other formats (original logic)
            if files_to_load:
                filename = files_to_load[0][1]
            filepath, tried_paths = self._try_path_locations(input_path, filename)
            
            if filepath is None:
                error_msg = f"Data file not found: {filename}\n"
                error_msg += f"Input path: {input_path}\n"
                error_msg += f"Current working directory: {os.getcwd()}\n"
                error_msg += f"Tried paths:\n"
                for path in tried_paths:
                    exists = "✓ EXISTS" if os.path.exists(path) else "✗ NOT FOUND"
                    error_msg += f"  {exists}: {path}\n"
                raise DataError(error_msg)
        
        try:
            with h5py.File(filepath, 'r') as f:
                # Try to get image data
                if 'data' in f:
                    image_data = f['data'][:]
                elif 'hyperspectral' in f:
                    image_data = f['hyperspectral'][:]
                elif 'image' in f:
                    image_data = f['image'][:]
                else:
                    raise DataError("Image data not found in HDF5 file")
                
                # Try to get lon/lat matrices
                if 'lon_mat' in f and 'lat_mat' in f:
                    lon_mat = f['lon_mat'][:]
                    lat_mat = f['lat_mat'][:]
                else:
                    # If not available, try to generate from georeferencing
                    # This would need to be implemented based on your file structure
                    raise DataError("lon_mat and lat_mat not found in file. Please ensure GeoTIFF format or include coordinate matrices.")
            
            return {
                'image': image_data,
                'lon_mat': lon_mat,
                'lat_mat': lat_mat
            }
        except Exception as e:
            raise DataError(f"Failed to load HDF5 file {filepath}: {str(e)}") from e
    
    def _load_geotiff_airbus(self, filepath: str, config: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """
        Load Airbus data from GeoTIFF file.
        
        Uses getLatLon_fromTiff.convert_raster_to_geocoords to get correct lat/lon coordinates
        with UTM to WGS84 conversion.
        
        Args:
            filepath: Path to GeoTIFF file
            config: Configuration dictionary with 'zone' information
            
        Returns:
            Dictionary with keys: 'image', 'lon_mat', 'lat_mat'
        """
        if not HAS_RASTERIO:
            raise DataError("rasterio is required for GeoTIFF loading. Install with: pip install rasterio")
        
        # Normalize the path
        filepath = os.path.normpath(filepath)
        
        # Check if file exists
        if not os.path.exists(filepath):
            # Try as absolute path if it's relative
            if not os.path.isabs(filepath):
                abs_filepath = os.path.abspath(filepath)
                if os.path.exists(abs_filepath):
                    filepath = abs_filepath
                else:
                    # Try relative to current working directory
                    cwd_filepath = os.path.join(os.getcwd(), filepath)
                    if os.path.exists(cwd_filepath):
                        filepath = cwd_filepath
                    else:
                        raise DataError(
                            f"GeoTIFF file not found: {filepath}\n"
                            f"Current working directory: {os.getcwd()}\n"
                            f"Absolute path tried: {abs_filepath}\n"
                            f"CWD path tried: {cwd_filepath}"
                        )
            else:
                raise DataError(
                    f"GeoTIFF file not found: {filepath}\n"
                    f"Current working directory: {os.getcwd()}"
                )
        
        # Read image data with rasterio
        with rasterio.open(filepath) as src:
            # Read image data
            image_data = src.read()  # Shape: (bands, height, width)
            # Transpose to (height, width, bands)
            if len(image_data.shape) == 3:
                image_data = np.transpose(image_data, (1, 2, 0))
            elif len(image_data.shape) == 2:
                # Single band - add channel dimension
                image_data = image_data[:, :, np.newaxis]
        
        # Get lat/lon coordinates using convert_raster_to_geocoords
        # Determine zone_number and zone_letter from config
        zone = config.get('zone', 'Detroit')
        
        # Map zone names to UTM zone numbers and letters
        # Detroit is in UTM zone 17T
        zone_mapping = {
            'Detroit': (17, 'T'),
            'detroit': (17, 'T'),
            'Israel': (36, 'R'),
            'israel': (36, 'R'),
        }
        
        if zone in zone_mapping:
            zone_number, zone_letter = zone_mapping[zone]
        else:
            # Default to Detroit zone
            zone_number, zone_letter = 17, 'T'
            print(f"Warning: Unknown zone '{zone}', using default zone {zone_number}{zone_letter}")
        
        # Try to use convert_raster_to_geocoords function if available
        if HAS_GET_LATLON and _convert_raster_to_geocoords_func is not None:
            try:
                # Use the user's convert_raster_to_geocoords function directly
                # Function signature: convert_raster_to_geocoords(file_path, zone_number=36, zone_letter='U')
                # Ensure filepath is a string (not tuple/list)
                if not isinstance(filepath, str):
                    filepath = str(filepath)
                
                # Call with all arguments as keyword arguments to avoid any conflicts
                # Use the zone_number and zone_letter determined by the importer from config['zone']
                latlon_mat = _convert_raster_to_geocoords_func(
                    file_path=filepath,  # Explicit keyword argument
                    zone_number=zone_number,  # Keyword argument (uses importer's zone_number from zone mapping)
                    zone_letter=zone_letter   # Keyword argument (uses importer's zone_letter from zone mapping)
                )
                
                # Unpack lat/lon from latlon_mat
                # Function returns: latlon_mat = np.append(lat_mat, lon_mat, -1)
                # So: latlon_mat[:, :, 0] = lat_mat, latlon_mat[:, :, 1] = lon_mat
                lon_mat = latlon_mat[:, :, 0]
                lat_mat = latlon_mat[:, :, 1]
                
            except Exception as e:
                raise DataError(f"Failed to convert raster to geocoords using getLatLon_fromTiff: {str(e)}") from e
        else:
            # Fallback: use rasterio directly (less accurate, no UTM conversion)
            if not HAS_COVERT_ITM:
                raise DataError(
                    "getLatLon_fromTiff module is required for accurate coordinate conversion. "
                    "Please ensure getLatLon_fromTiff is available, or install CovertITM2LatLon module."
                )
            
            # Use the existing _convert_raster_to_geocoords method as fallback
            latlon_mat = self._convert_raster_to_geocoords(filepath, zone_number=zone_number, zone_letter=zone_letter)
            
            # Unpack lat/lon from latlon_mat
            # Match the same pattern as getLatLon_fromTiff function
            if len(latlon_mat.shape) == 3 and latlon_mat.shape[2] == 2:
                lat_mat = latlon_mat[:, :, 0]
                lon_mat = latlon_mat[:, :, 1]
            else:
                # If shape is different, try to extract
                raise DataError(f"Unexpected latlon_mat shape: {latlon_mat.shape}")
        
        return {
            'image': image_data,
            'lon_mat': lon_mat,
            'lat_mat': lat_mat
        }
    
    def validate_data(self, data: DataContainer) -> bool:
        """
        Validate loaded hyperspectral data.
        
        Args:
            data: DataContainer to validate
            
        Returns:
            True if valid
            
        Raises:
            ValidationError if invalid
        """
        if not isinstance(data, DataContainer):
            raise ValidationError("Expected DataContainer")
        
        if data.data is None:
            raise ValidationError("Data cannot be None")
        
        # Data should be a dictionary with image, lon_mat, lat_mat
        if not isinstance(data.data, dict):
            raise ValidationError("Data must be a dictionary with keys: 'image', 'lon_mat', 'lat_mat'")
        
        required_keys = ['image', 'lon_mat', 'lat_mat']
        for key in required_keys:
            if key not in data.data:
                raise ValidationError(f"Data dictionary missing required key: {key}")
            
            if not isinstance(data.data[key], np.ndarray):
                raise ValidationError(f"Data['{key}'] must be a numpy array")
        
        # Validate image shape
        image = data.data['image']
        if len(image.shape) < 2:
            raise ValidationError("Image data must be at least 2D (height, width, bands)")
        
        # Validate lon/lat shapes match image spatial dimensions
        lon_mat = data.data['lon_mat']
        lat_mat = data.data['lat_mat']
        if lon_mat.shape[:2] != image.shape[:2] or lat_mat.shape[:2] != image.shape[:2]:
            raise ValidationError("lon_mat and lat_mat spatial dimensions must match image dimensions")
        
        return True


class GroundTruthDataImporter(BaseModule, DataInterface):
    """
    Importer for ground truth PCI data.
    
    Loads PCI labels from various formats (CSV, HDF5, etc.)
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize ground truth data importer.
        
        Args:
            config: Configuration dictionary with keys:
                - input_path: Path to PCI data file
                - file_format: 'csv', 'h5', etc.
        """
        super().__init__("ground_truth_importer", config)
        self.required_config_keys = ['input_path']
    
    def load_data(self, config: Optional[Dict[str, Any]] = None) -> DataContainer:
        """
        Load ground truth PCI data.
        
        Supports CSV and Excel files with PCI data, coordinates, and segment IDs.
        Returns DataContainer with data as a dictionary containing:
        - 'pci': PCI values array
        - 'lon': Longitude values array
        - 'lat': Latitude values array
        - 'seg_id': Segment IDs array (if available)
        
        Args:
            config: Optional configuration (overrides instance config)
                - input_path: Path to PCI data file
                - file_format: 'csv', 'xls', 'xlsx', or 'h5'
                - isLatLon: If True, coordinates are already in lat/lon (default: False, assumes ITM)
        
        Returns:
            DataContainer with PCI data as dict with keys: pci, lon, lat, seg_id
        """
        load_config = {**self.config, **(config or {})}
        
        try:
            input_path = load_config['input_path']
            file_format = load_config.get('file_format', 'auto')  # Auto-detect from extension
            isLatLon = load_config.get('isLatLon', False)
            
            # Auto-detect file format from extension if not specified
            if file_format == 'auto':
                file_ext = Path(input_path).suffix.lower()
                if file_ext == '.csv':
                    file_format = 'csv'
                elif file_ext in ['.xls', '.xlsx']:
                    file_format = 'xls'
                elif file_ext in ['.h5', '.hdf5']:
                    file_format = 'h5'
                else:
                    # Default to CSV if unknown
                    file_format = 'csv'
            
            if file_format == 'h5':
                data = self._load_h5(input_path)
            elif file_format in ['csv', 'xls', 'xlsx']:
                data = self._load_csv_or_excel(input_path, isLatLon=isLatLon)
            else:
                raise DataError(f"Unsupported file format: {file_format}")
            
            metadata = {
                'input_path': input_path,
                'file_format': file_format,
                'data_type': 'ground_truth_pci',
                'isLatLon': isLatLon,
            }
            
            return DataContainer(
                data=data,
                metadata=metadata,
                data_type='ground_truth_pci'
            )
        
        except Exception as e:
            raise DataError(f"Failed to load ground truth data: {str(e)}") from e
    
    def _load_h5(self, filepath: str) -> np.ndarray:
        """Load from HDF5 file."""
        if not os.path.exists(filepath):
            raise DataError(f"Data file not found: {filepath}")
        
        with h5py.File(filepath, 'r') as f:
            # TODO: Adjust keys based on actual HDF5 structure
            data = f['pci_labels'][:] if 'pci_labels' in f else f['data'][:]
        
        return data
    
    def _load_csv_or_excel(self, filepath: str, isLatLon: bool = False) -> Dict[str, np.ndarray]:
        """
        Load PCI data from CSV or Excel file using get_GT_xy_PCI logic.
        
        Args:
            filepath: Path to CSV or Excel file
            isLatLon: If True, coordinates are already in lat/lon format.
                     If False, assumes ITM coordinates and converts to WGS84.
        
        Returns:
            Dictionary with keys: 'pci', 'lon', 'lat', 'seg_id'
        """
        from apa.utils.ground_truth_loader import get_GT_xy_PCI
        
        # Check if file exists
        if not os.path.exists(filepath):
            raise DataError(f"Data file not found: {filepath}")
        
        # Use the utility function to load data
        lon_vec, lat_vec, pci_vec, seg_id = get_GT_xy_PCI(filepath, isLatLon=isLatLon)
        
        # Return as dictionary
        return {
            'pci': pci_vec,
            'lon': lon_vec.flatten() if len(lon_vec.shape) > 1 else lon_vec,
            'lat': lat_vec.flatten() if len(lat_vec.shape) > 1 else lat_vec,
            'seg_id': seg_id if len(seg_id) > 0 else None
        }
    
    def validate_data(self, data: DataContainer) -> bool:
        """
        Validate loaded ground truth data.
        
        Args:
            data: DataContainer to validate
            
        Returns:
            True if valid
            
        Raises:
            ValidationError if invalid
        """
        if not isinstance(data, DataContainer):
            raise ValidationError("Expected DataContainer")
        
        if data.data is None:
            raise ValidationError("Data cannot be None")
        
        # Data should be a dictionary with pci, lon, lat
        if isinstance(data.data, dict):
            required_keys = ['pci', 'lon', 'lat']
            for key in required_keys:
                if key not in data.data:
                    raise ValidationError(f"Data dictionary missing required key: {key}")
                
                if not isinstance(data.data[key], np.ndarray):
                    raise ValidationError(f"Data['{key}'] must be a numpy array")
            
            # Validate that arrays have same length
            pci_len = len(data.data['pci'])
            lon_len = len(data.data['lon'])
            lat_len = len(data.data['lat'])
            
            if not (pci_len == lon_len == lat_len):
                raise ValidationError(f"PCI, lon, and lat arrays must have same length. Got: pci={pci_len}, lon={lon_len}, lat={lat_len}")
        
        return True


class RoadDataImporter(BaseModule, DataInterface):
    """
    Importer for road network data.
    
    Loads road masks from NPZ files (if they exist) or creates them from OpenStreetMap.
    Uses the same logic as the original implementation:
    - Checks for existing .npz file with enum_data_source suffix
    - If not found, creates mask from OpenStreetMap and saves it
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize road data importer.
        
        Args:
            config: Configuration dictionary with keys:
                - Can use full config with 'data' and 'preprocessing' sections
                - Or minimal config with 'osx_map_mask_path' and 'enum_data_source'
        """
        super().__init__("road_importer", config)
        self.required_config_keys = []  # Config can come from full config structure
    
    def load_data(self, config: Optional[Dict[str, Any]] = None, 
                  hyperspectral_data: Optional[DataContainer] = None,
                  repo_root: Optional[str] = None) -> DataContainer:
        """
        Load road network mask.
        
        Args:
            config: Optional configuration (overrides instance config)
                Can be full config dict with 'data' and 'preprocessing' sections
            hyperspectral_data: Optional DataContainer with hyperspectral data containing:
                - data['lon_mat']: Longitude matrix
                - data['lat_mat']: Latitude matrix
                - metadata['rois']: ROI list
            repo_root: Optional repository root path (defaults to current working directory)
        
        Returns:
            DataContainer with road mask as numpy array
        """
        load_config = {**self.config, **(config or {})}
        
        try:
            from apa.utils.road_mask_loader import get_mask_from_roads_gdf
            
            # Extract config values - handle both flat and nested config structures
            # Check if config has 'config' wrapper (from ConfigManager)
            if 'config' in load_config:
                full_config = load_config['config']
            else:
                full_config = load_config
            
            # Get enum_data_source
            if 'data' in full_config and 'enum_data_source' in full_config['data']:
                enum_data_source = full_config['data']['enum_data_source']
            elif 'enum_data_source' in full_config:
                enum_data_source = full_config['enum_data_source']
            else:
                raise DataError("enum_data_source not found in config. "
                              "Please provide config with 'data.enum_data_source' or 'enum_data_source'")
            
            # Get npz filename path
            if ('preprocessing' in full_config and 
                'georeferencing' in full_config['preprocessing'] and
                'osx_map_mask_path' in full_config['preprocessing']['georeferencing']):
                npz_filename = full_config['preprocessing']['georeferencing']['osx_map_mask_path']
            else:
                # Default path
                npz_filename = 'data/Detroit/masks_OpenStreetMap/Detroit_OpenSteet_roads_mask.npz'
            
            # Add enum_data_source suffix before .npz extension
            if '.npz' in npz_filename:
                npz_filename = npz_filename[:npz_filename.find('.npz')] + str(enum_data_source) + '.npz'
            
            # Get repository root
            if repo_root is None:
                repo_root = os.getcwd()
            
            # Construct full path
            npz_filename = os.path.join(repo_root, npz_filename)
            
            # Get hyperspectral data if not provided
            if hyperspectral_data is None:
                raise DataError("hyperspectral_data is required to create road mask. "
                              "Please provide DataContainer with lon_mat, lat_mat, and rois.")
            
            # Extract data from hyperspectral container
            if not isinstance(hyperspectral_data.data, dict):
                raise DataError("hyperspectral_data.data must be a dictionary with 'lon_mat' and 'lat_mat'")
            
            lon_mat = hyperspectral_data.data.get('lon_mat')
            lat_mat = hyperspectral_data.data.get('lat_mat')
            rois = hyperspectral_data.metadata.get('rois', [])
            
            if lon_mat is None or lat_mat is None:
                raise DataError("hyperspectral_data must contain 'lon_mat' and 'lat_mat' in data dictionary")
            
            if not rois:
                raise DataError("hyperspectral_data must contain 'rois' in metadata")
            
            # Use first ROI
            roi = rois[0]  # Format: [xmin_cut, xmax_cut, ymin_cut, ymax_cut]
            
            # Calculate crop_rect from lon_mat and lat_mat shape
            # crop_rect is (x_ind_min, y_ind_min, x_ind_max, y_ind_max)
            # For full image, use entire matrix
            height, width = lon_mat.shape[:2]
            crop_rect = (0, 0, width, height)
            
            # Prepare data dictionary for get_mask_from_roads_gdf
            mask_data = {
                "roi": roi,
                "X_cropped": lat_mat,  # X is latitude
                "Y_cropped": lon_mat,   # Y is longitude
            }
            
            # Get or create mask
            coinciding_mask = get_mask_from_roads_gdf(npz_filename, crop_rect, mask_data)
            
            metadata = {
                'npz_filename': npz_filename,
                'enum_data_source': enum_data_source,
                'roi': roi,
                'crop_rect': crop_rect,
                'data_type': 'road_mask',
            }
            
            return DataContainer(
                data=coinciding_mask,
                metadata=metadata,
                data_type='road_mask'
            )
        
        except Exception as e:
            raise DataError(f"Failed to load road data: {str(e)}") from e
    
    def validate_data(self, data: DataContainer) -> bool:
        """
        Validate loaded road data.
        
        Args:
            data: DataContainer to validate
            
        Returns:
            True if valid
            
        Raises:
            ValidationError if invalid
        """
        if not isinstance(data, DataContainer):
            raise ValidationError("Expected DataContainer")
        
        if data.data is None:
            raise ValidationError("Data cannot be None")
        
        # Data should be a numpy array (road mask)
        if not isinstance(data.data, np.ndarray):
            raise ValidationError("Road mask data must be a numpy array")
        
        # Should be 2D array
        if len(data.data.shape) != 2:
            raise ValidationError(f"Road mask should be 2D array, got shape: {data.data.shape}")
        
        return True

