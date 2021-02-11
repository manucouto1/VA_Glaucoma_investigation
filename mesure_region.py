"""
https://scikit-image.org/docs/stable/auto_examples/segmentation/plot_regionprops.html#sphx-glr-auto-examples-segmentation-plot-regionprops-py
"""
from scipy import stats
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import threshold as th
from skimage.measure import label, regionprops, regionprops_table
import visualize as vi

def props(img, mask):
    label_img = label(mask)
    regions = regionprops(label_img)

    return regions

def bigest_area(regions):
    max_mean = 0.0
    prop_max = None
    for prop in regions:
        if max_mean < prop.area:
            max_mean = prop.area
            prop_max = prop

    return prop_max 

def higest_non_zero_mean(mask):
    return  


def closest_prop(img, mask):
    (lbls, num) = label(mask, return_num=True)
    max_mean = 0.0
    max_mask = None

    if num > 1:
        for lb in range(num + 1):
            current_mean = 0
            current = th.apply(img, lbls == lb)
            current_mean = np.sum(current) / np.count_nonzero(current)
            if max_mean < current_mean:
                max_mean = current_mean
                max_mask = current

        return max_mask>0
    else:
        return mask

