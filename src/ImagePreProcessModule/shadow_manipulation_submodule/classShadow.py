#%%
import os, sys

# To properly import ylim
# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Add it to sys.path if it's not already there
if script_dir not in sys.path:
    sys.path.append(script_dir)
    

import ylim
import matplotlib.pyplot as plt
import cv2
import numpy as np
import matplotlib.pyplot as plt

class removeShadow:
    def __init__(self, image, brightness, contrast, treshold=0, repeat=False):
        self.B = brightness
        self.C = contrast
        self.treshold = treshold
        self.repeat = repeat
        self.image = image

        # Ensure the image is in the correct format
        if self.image.dtype != np.uint8:
            self.image = cv2.cvtColor((self.image * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)  # Convert to BGR for OpenCV processing

    def rgb(self, image2rgb):
        return cv2.cvtColor(image2rgb, cv2.COLOR_BGR2RGB)
    
    def AliModel(self, toAdjust):
        # Adjust brightness and contrast
        adjustedImage = cv2.convertScaleAbs(toAdjust, alpha=self.C, beta=self.B)
        return adjustedImage
    
    def remove(self):
        # Process the image using Ylim's model
        _, _, imageClear = ylim.processImageFile(self.image, ab_threshold=self.treshold)
        imageBright = self.AliModel(toAdjust=imageClear)

        # Repeat ylims model after adjusting brightness and contrast if specified 
        if self.repeat:
            _, _, imageClear = ylim.processImageFile(imageBright, ab_threshold=self.treshold)
            return imageClear
        return imageBright
#%%
if __name__ == "__main__":
    # Load the cropped image data
    cropped_msp_img = np.load("/home/ariep/Hyperspectral Road/brach_from_github/test_hys_image.npy")
    # cropped_msp_img = data["cropped"]

    # Zoom into the area of interest
    x1, x2 = 1500, 3000
    y1, y2 = 1500, 3000

    # Convert to RGB and zoom in
    cropped_img = cropped_msp_img[:, :, 3:][y1:y2, x1:x2]

    # Display the cropped image
    plt.imshow(cropped_img)

    # Remove shadows
    shadow = removeShadow(image=cropped_img,
                                  brightness=70, contrast=2,
                                  treshold=0, repeat=True)
    imageClear = shadow.remove()
    
    # Display the result
    plt.imshow(shadow.rgb(imageClear), cmap='viridis')
    plt.axis('off')
    plt.show()
        
        
    
    


# %%
