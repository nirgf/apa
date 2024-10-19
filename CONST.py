# central wavelength and bandwidth are in nm
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

if __name__ == "__main__":
    # Displaying the dictionary
    import pprint
    pprint.pprint(bands_dict)