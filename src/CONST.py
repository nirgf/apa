# central wavelength and bandwidth are in nm
from enums.datasets_enum import Dataset as enum_Dataset
import pprint


def get_spectral_bands(config):
    ie_datasource = config['data']['enum_data_source']
    is_venus = (enum_Dataset(ie_datasource) == enum_Dataset.venus_Detroit) or \
               (enum_Dataset(ie_datasource) == enum_Dataset.venus_IL)

    is_airbus = (enum_Dataset(ie_datasource) == enum_Dataset.airbus_HSP_Detroit) or \
                (enum_Dataset(ie_datasource) == enum_Dataset.airbus_Pan_Detroit)
    
    if is_venus:
            bands_dict = {
                0: {"wavelength": 423.9, "bandwidth": 40, "main_driver": "Atmospheric correction"},
                1: {"wavelength": 446.9, "bandwidth": 40, "main_driver": "Aerosol, clouds"},
                2: {"wavelength": 491.9, "bandwidth": 40, "main_driver": "Atmospheric correction"},
                3: {"wavelength": 555.0, "bandwidth": 40, "main_driver": "Land"},
                4: {"wavelength": 619.7, "bandwidth": 40, "main_driver": "Land"},
                5: {"wavelength": 619.5, "bandwidth": 40, "main_driver": "DEM, image quality"},
                6: {"wavelength": 666.2, "bandwidth": 30, "main_driver": "Land"},
                7: {"wavelength": 702.0, "bandwidth": 24, "main_driver": "Land"},
                8: {"wavelength": 741.1, "bandwidth": 16, "main_driver": "Land"},
                9: {"wavelength": 782.2, "bandwidth": 16, "main_driver": "Land"},
                10: {"wavelength": 861.1, "bandwidth": 40, "main_driver": "Land"},
                11: {"wavelength": 908.7, "bandwidth": 20, "main_driver": "Water vapor"}
            }
    elif is_airbus:
            bands_dict = {
                0: {"wavelength": 654.5, "bandwidth": 71, "main_driver": ""}, # Red
                1: {"wavelength": 562,   "bandwidth": 58, "main_driver": ""}, # Green
                2: {"wavelength": 483,   "bandwidth": 74, "main_driver": ""}, # Blue
                3: {"wavelength": 828,   "bandwidth": 120,"main_driver": ""}, # NIR 
                4: {"wavelength": 723.5, "bandwidth": 53, "main_driver": ""}, # Red Edge
                5: {"wavelength": 436.5, "bandwidth": 40, "main_driver": ""}, # Deep Blue
            }
    else:        
        bands_dict = {}
        pprint.pprint('no spectral information for datasource enum. please check config file.')
    
    pprint.pprint(bands_dict)
    return bands_dict
                