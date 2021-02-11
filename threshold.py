import sys
import numpy as np
import skimage.color
import skimage.filters
import skimage.io
import skimage.morphology


def mask_manual(image, i_sigma = 2.2, t= 0.8):
    if(len(image.shape) == 3):
        blur = skimage.color.rgb2gray(image)
    else:
        blur = image
    blur = skimage.filters.gaussian(blur, sigma=i_sigma)
    mask = blur > t
    return mask



def mask_otsu(image, i_sigma = 2.2):
    if(len(image.shape) == 3):
        blur = skimage.color.rgb2gray(image)
    else:
        blur = image
    blur = skimage.filters.gaussian(blur, sigma=i_sigma)
    t = skimage.filters.threshold_otsu(blur)
    mask = blur > t
    mask = skimage.morphology.remove_small_objects(mask, 50)
    return mask


def mask_yen(image, i_sigma = 2.2):
    if(len(image.shape) == 3):
        blur = skimage.color.rgb2gray(image)
    else:
        blur = image
    blur = skimage.filters.gaussian(blur, sigma=i_sigma)
    t = skimage.filters.threshold_yen(blur)
    mask = blur > t
    mask = skimage.morphology.remove_small_objects(mask, 2500)
    return mask

def try_all(image):
    blur = skimage.color.rgb2gray(image)
    _, _ = skimage.filters.try_all_threshold(blur, figsize=(10, 8), verbose=False)
    
    
def apply(image, mask):
    sel = np.zeros_like(image)
    sel[mask] = image[mask]

    return sel