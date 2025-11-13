from tifffile import imread
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd

def getVenusData(dirname, filename):
    
    # Read the data, needs to be automated if multipile frames are used
    image = imread(os.path.join(dirname,filename))
    
    return image


def getVenusMetaData(dirname, filename):
    
    # Read the data, needs to be automated if multipile frames are used
    metadata_pd = pd.read_csv(os.path.join(dirname,filename))
    
    return metadata_pd
