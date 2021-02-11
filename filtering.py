from skimage.filters import meijering, sato, frangi, hessian
import skimage.morphology
from skimage.morphology import disk, binary_opening, binary_closing, binary_dilation
import numpy as np

def frango(image):
    return frangi(image)

def hess(image):
    return hessian(image)

def meije(image):
    return meije(image)

def sat(image):
    return sato(image)

def target_set_mean(image, percent = 10):
    array = np.array(image).flatten()
    highest = np.sort(array)
    highest = highest[::-1]
    chusen = int(len(highest) * percent /100)
    target = np.mean(highest[1:chusen])
    return target

def hand_made(image, percent = 10, SE = 3, smallest=1500):
    target = target_set_mean(image, percent)
    mask = image >= target
    mask = binary_opening(mask, disk(SE))
    mask = skimage.morphology.remove_small_objects(mask, smallest)
    mask = binary_closing(mask, disk(SE))

    return mask
