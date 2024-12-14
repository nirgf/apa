import os.path
import json
import numpy as np
from pathlib import Path
import yaml
import json

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
        "augmentations": {'crop_size': [256, 256]}
    }
}

def read_yaml_config(config_path):
    # Load the configuration from a YAML file
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
        return config



def fill_with_defaults(user_config=None,log=True):
    # Define the default config inside the function
    default_cfg = {
        "thresholds": {
            "low": 0.1,
            "high": 0.9
        },
        "paths": {
            "input_dir": "data/input",
            "output_dir": "data/output"
        },
        "feature_extraction": {
            "window_size": 256,
            "overlap": 128
        }
    }

    if user_config is None:
        user_config = {}

    def recursive_merge(user, default):
        merged = default.copy()
        for k, v in user.items():
            if k in merged and isinstance(merged[k], dict) and isinstance(v, dict):
                merged[k] = recursive_merge(v, merged[k])
            else:
                merged[k] = v
        return merged

    return recursive_merge(user_config, default_config)

def pretty_print_config(final_config, default_config, indent=0):
    """
    Pretty-print the final_config compared to default_config.
    - Unchanged keys: Yellow background for key/value
    - Changed keys: Key in red, value in blue background
    """
    indent_str = '  ' * indent
    reset = "\033[0m"
    red = "\033[31m"
    yellow_bg = "\033[43m"
    blue_bg = "\033[44m"

    for key, val in final_config.items():
        # Determine if this key is changed or unchanged
        # If the key does not exist in default_config or values differ, it's changed
        if key not in default_config:
            # Treat as changed if it's not even in defaults
            changed = True
        else:
            # Check if value differs
            changed = not values_equal(val, default_config[key])

        if isinstance(val, dict):
            # Recursively handle dictionaries
            if changed:
                # Print the key in red only
                print(f"{indent_str}{red}{key}:{reset}")
            else:
                # Unchanged key: print with yellow background
                print(f"{indent_str}{yellow_bg}{key}:{reset}")
            pretty_print_config(val, default_config.get(key, {}), indent + 1)
        else:
            # Leaf node (non-dict value)
            if changed:
                # Print key in red, value in blue background
                print(f"{indent_str}{red}{key}{reset}: {blue_bg}{val}{reset}")
            else:
                # Unchanged: key/value on yellow background
                print(f"{indent_str}{yellow_bg}{key}:{val}{reset}")


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
