from sklearn.cluster import KMeans
import numpy as np
import visualize as vi
import skimage.color
import skimage.filters
import numpy as np
import matplotlib.pyplot as plt
from skimage import data, img_as_float
from skimage.segmentation import (morphological_chan_vese,
                                  morphological_geodesic_active_contour,
                                  inverse_gaussian_gradient,
                                  checkerboard_level_set)

def segments(image, k, i_sigma = 2.2):
    blur = skimage.color.rgb2gray(image)
    blur = skimage.filters.gaussian(blur, sigma=i_sigma)

    image_gray= blur.reshape(blur.shape[0] * blur.shape[1], 1)
    

    kmeans = KMeans(n_clusters=k, random_state=0).fit(image_gray)

    clustered = kmeans.cluster_centers_[kmeans.labels_]
    labels = kmeans.labels_

    for i in range(k):
        image_cluster = []
        for i in range(len(labels)):
            if(labels[i]) == k:
                image_cluster.append(float(clustered[i]))
            else:
                image_cluster.append(1)
    
    if(k==1):
        image_fix= np.array(image_cluster).reshape(blur.shape)

    reshape_clustered = np.array(image_cluster).reshape(blur.shape)

    vi.pltImage(blur)
    vi.show()


def store_evolution_in(lst):
    def _store(x):
        lst.append(np.copy(x))

    return _store

def morphological(image):

    # Morphological ACWE
    img = img_as_float(skimage.color.rgb2gray(image))
    # Initial level set
    init_ls = checkerboard_level_set(img.shape, 6)
    # List with intermediate results for plotting the evolution
    evolution = []
    callback = store_evolution_in(evolution)
    ls = morphological_chan_vese(img, 200, init_level_set=init_ls, smoothing=4,
                                iter_callback=callback)

    fig, axes = plt.subplots(2, 2, figsize=(8, 8))
    ax = axes.flatten()

    ax[0].imshow(img, cmap="gray")
    ax[0].set_axis_off()
    ax[0].contour(ls, [0.5], colors='r')
    ax[0].set_title("Morphological ACWE segmentation", fontsize=12)

    ax[1].imshow(ls, cmap="gray")
    ax[1].set_axis_off()
    contour = ax[1].contour(evolution[2], [0.5], colors='g')
    contour.collections[0].set_label("Iteration 2")
    contour = ax[1].contour(evolution[7], [0.5], colors='y')
    contour.collections[0].set_label("Iteration 7")
    contour = ax[1].contour(evolution[-1], [0.5], colors='r')
    contour.collections[0].set_label("Iteration 35")
    ax[1].legend(loc="upper right")
    title = "Morphological ACWE evolution"
    ax[1].set_title(title, fontsize=12)


    # Morphological GAC
    img = img_as_float(skimage.color.rgb2gray(image))
    gimage = inverse_gaussian_gradient(img)

    # Initial level set
    init_ls = np.zeros(img.shape, dtype=np.int8)
    init_ls[10:-10, 10:-10] = 1
    # List with intermediate results for plotting the evolution
    evolution = []
    callback = store_evolution_in(evolution)
    ls = morphological_geodesic_active_contour(gimage, 230, init_ls,
                                            smoothing=5, balloon=-1,
                                            threshold=0.5,
                                            iter_callback=callback)

    ax[2].imshow(img, cmap="gray")
    ax[2].set_axis_off()
    ax[2].contour(ls, [0.5], colors='r')
    ax[2].set_title("Morphological GAC segmentation", fontsize=12)

    ax[3].imshow(ls, cmap="gray")
    ax[3].set_axis_off()
    contour = ax[3].contour(evolution[0], [0.5], colors='g')
    contour.collections[0].set_label("Iteration 0")
    contour = ax[3].contour(evolution[100], [0.5], colors='y')
    contour.collections[0].set_label("Iteration 100")
    contour = ax[3].contour(evolution[-1], [0.5], colors='r')
    contour.collections[0].set_label("Iteration 230")
    ax[3].legend(loc="upper right")
    title = "Morphological GAC evolution"
    ax[3].set_title(img, fontsize=12)

    fig.tight_layout()
    plt.show()
 
