import os.path
import json
import numpy as np
from pathlib import Path
import yaml
import json

# ANSI color codes
reset = "\033[0m"
white_bg_green_font = "\033[107m\033[32m"  # White background, green font
gray_bg_light_green_font = "\033[100m\033[92m"  # Gray background, light green font


default_config = {
    "paths": {
        "input_dir": "data/input",
        "output_dir": "data/output"
    },
    "feature_extraction": {
        "window_size": 256,
        "overlap": 128
    },
    "preprocessing": {
        "augmentations": {'crop_size': [256, 256]},
        "preprocessing": {
            "normalization": True,
            "normalization_type": "min-max",
            "spectral_smoothing": False,
            "georeferencing": {
                "merge_threshold": [0.05],
                "merge_method": "mean_min",
            }
        }

    },
    "cnn_model": {
        "overlap": 0.2,
        "crop_size": 64,

    }
}

def read_yaml_config(config_path):
    # Load the configuration from a YAML file
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
        return config

######

def merge_config(user_config, default_config):
    """
    Recursively merge user-provided config into default config.
    Adds missing keys from default_config while preserving user-defined keys.

    Parameters
    ----------
    user_config : dict
        User-provided configuration (may be partial).
    default_config : dict
        Default configuration.

    Returns
    -------
    dict
        Merged configuration.
    """
    merged = {}
    for key, default_value in default_config.items():
        if key in user_config:
            user_value = user_config[key]
            if isinstance(default_value, dict) and isinstance(user_value, dict):
                # Recursively merge nested dictionaries
                merged[key] = merge_config(user_value, default_value)
            else:
                # Use the user-provided value
                merged[key] = user_value
        else:
            # If key is missing in user_config, use the default value
            merged[key] = default_value

    # Add any extra keys from user_config that are not in default_config
    for key in user_config:
        if key not in default_config:
            user_value = user_config[key]
            merged[key] = user_value  # Add the extra key from user_config

    return merged


def pretty_print_config(final_config, default_config, user_config, indent=0):
    """
    Pretty-print the final merged configuration.

    - User-provided values: White background with green font.
    - Default-added values: Gray background with light green font.

    Parameters
    ----------
    final_config : dict
        Merged configuration.
    default_config : dict
        Default configuration.
    user_config : dict
        User-provided configuration.
    indent : int
        Indentation level for nested dictionaries.
    """
    indent_str = '  ' * indent

    for key, value in final_config.items():
        if isinstance(value, dict):
            # Recursively handle nested dictionaries
            print(f"{indent_str}{key}:")
            pretty_print_config(
                value,
                default_config.get(key, {}),
                user_config.get(key, {}),
                indent + 1,
            )
        else:
            if key in user_config and user_config[key] == value:
                # User-provided value: White background with green font
                print(f"{indent_str}{white_bg_green_font}{key}: {value}{reset}")
            else:
                # Default-added value: Gray background with light green font
                print(f"{indent_str}{gray_bg_light_green_font}{key}: {value}{reset}")

######

def values_equal(v1, v2):
    """Helper function to compare values (recursively for dicts)"""
    if isinstance(v1, dict) and isinstance(v2, dict):
        if v1.keys() != v2.keys():
            return False
        for k in v1:
            if not values_equal(v1[k], v2[k]):
                return False
        return True
    else:
        return v1 == v2

def fill_with_defaults(user_config):
    config_merge=merge_config(user_config, default_config)
    print('*' * 20, 'CONFIG LOG COLORMAP', '*'* 20)
    print(f"{white_bg_green_font}USER DEFINED{reset}")
    print(f"{gray_bg_light_green_font}DEFAULT ADDED{reset}")
    pretty_print_config(config_merge,default_config, user_config)
    print('*' * 60)
    return config_merge


def parse_metadata(file_path):
    """
    Reads and parses a JSON metadata file.

    Args:
        file_path (str): Path to the JSON metadata file.

    Returns:
        dict: Parsed metadata as a dictionary.
    """
    try:
        # Step 1: Load the JSON file
        with open(file_path, "r") as file:
            metadata = json.load(file)

        # Step 2: Parse specific fields (example for detailed parsing)
        parsed_metadata = {
            "file_information": metadata.get("file_information", {}),
            "acquisition_information": metadata.get("acquisition_information", {}),
            "image_properties": metadata.get("image_properties", {}),
            "data_quality": metadata.get("data_quality", {}),
            "processing_information": metadata.get("processing_information", {}),
        }

        # Optional: Print summary of the metadata
        print("Metadata Summary:")
        print(f"  Filename: {parsed_metadata['file_information'].get('filename', 'N/A')}")
        print(f"  Format: {parsed_metadata['file_information'].get('file_format', 'N/A')}")
        print(f"  Acquisition Date/Time: {parsed_metadata['acquisition_information'].get('date_time', 'N/A')}")
        print(f"  Sensor Name: {parsed_metadata['acquisition_information'].get('sensor', {}).get('name', 'N/A')}")
        print(f"  Platform Type: {parsed_metadata['acquisition_information'].get('platform', {}).get('type', 'N/A')}")
        print(f"  Spectral Range: {parsed_metadata['image_properties'].get('spectral_range', {})}")
        print(f"  Cloud Coverage: {parsed_metadata['data_quality'].get('cloud_coverage', 'N/A')}")
        print(
            f"  Processing Software: {parsed_metadata['processing_information'].get('software', {}).get('name', 'N/A')}")

        return parsed_metadata

    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON format - {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
