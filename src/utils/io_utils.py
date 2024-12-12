import os.path
import json
import numpy as np
from pathlib import Path
import yaml


def read_yaml_config(config_path):
    # Load the configuration from a YAML file
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
        return config


import json


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
