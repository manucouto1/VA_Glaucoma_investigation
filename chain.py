import visualize as vi
import threshold as th
import mesure_region as msr
import numpy as np
import math
import skimage.color
from skimage.draw import ellipse
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import filtering as f
from skimage.filters import sobel
from skimage.filters.rank import mean
from skimage import segmentation
import skimage.morphology
from skimage.filters.rank import median
from skimage.morphology import disk, ball, binary_dilation, binary_erosion, binary_closing, binary_opening
from skimage.filters.rank import enhance_contrast
from skimage.filters.rank import enhance_contrast_percentile
from skimage.filters import difference_of_gaussians, window
from skimage import filters
from skimage.filters.rank import otsu
from skimage.filters import threshold_otsu
from skimage import exposure
from skimage.morphology import erosion, dilation, opening, closing, white_tophat
from skimage.feature import canny
from skimage.transform import hough_ellipse
from skimage.draw import ellipse_perimeter
from skimage.transform import hough_circle, hough_circle_peaks
from skimage.filters import meijering, sato, frangi, hessian
from skimage.transform import rotate
from skimage.exposure import rescale_intensity
from sklearn.feature_extraction import image
from sklearn.cluster import spectral_clustering
from skimage import data, segmentation, color
from skimage.future import graph
from skimage.filters import gaussian
from skimage.segmentation import active_contour
from skimage.segmentation import flood, flood_fill
from skimage.segmentation import chan_vese

def handmade_p_tile_method(img, op="disc", test=False):
    if op == "disc" or op == "cup":
        to_plot = []
        img_red = img[:,:,0]

        if test :
            to_plot.append(("Red Channel",img_red))

        img_red = dilation(img_red, disk(3))

        if test :
            to_plot.append(("Gaussing",img_red))

        img_red = enhance_contrast(img_red, disk(10))

        if test :
            to_plot.append(("Enhace_contrast",img_red))

        mask = f.hand_made(img_red, 10, 2)

        if test :
            to_plot.append(("P-tile",th.apply(img_red, mask)))

        mask = binary_opening(mask, disk(7))
        mask = skimage.morphology.remove_small_objects(mask, 1700)

        if test :
            to_plot.append(("Opening and remove small objects",th.apply(img_red, mask)))

        props = msr.props(img_red, mask)[0]

        minr, minc, maxr, maxc = props.bbox
        img_cut = img[minr:maxr, minc:maxc]
        
        if test :
            to_plot.append(("Binary opening",th.apply(img_red, mask)))
        if test:
            vi.plot_multy(to_plot,2,3, 'P-tile')
        
        
    if op == "cup":
        minr, minc, maxr, maxc = props.bbox
        img_green = img[minr:maxr, minc:maxc, 1]
        
        to_plot = []

        if test :
            to_plot.append(("green channel",img_green))

        v_mask , _= segment_veins(img_green, test)
        
        img_aux = closing(img_green, disk(6))
        img_aux = dilation(img_aux, disk(6))
        img_aux2 = dilation(img_green, disk(3))
        
        if test :
            to_plot.append(("closed + dilated",img_aux))
            to_plot.append(("dilated",img_aux2))

        img_v_closed = th.apply(img_aux, v_mask)
        img_t_dilated = th.apply(img_aux2, v_mask==False)

        if test :
            to_plot.append(("veins part",img_v_closed))
            to_plot.append(("target part",img_t_dilated))

        img_green = img_v_closed + img_t_dilated

        if test :
            to_plot.append(("without veins",img_green))

        img_green = dilation(img_green, disk(6))

        img_green = enhance_contrast(img_green, disk(10))

        if test :
            to_plot.append(("dilation + contrast",img_green))
        
        mask = f.hand_made(img_green, 10, 2,smallest=0)

        if test :
            to_plot.append(("P-tile",th.apply(img_green, mask)))

        mask1 = mask
        mask = np.zeros(img[:,:, 1].shape)
        mask[minr:maxr, minc:maxc] = mask1

        if test:
            vi.plot_multy(to_plot,4,4, 'Otsu Local chain')


    return (mask, img_cut, props)

def sobel_watershed_method(img, op="disc", veins=False, test=False):
    if op == "disc" or op == "cup":
        to_plot = []
        img_red = img[:,:,0]
        
        if test :
            to_plot.append(("Red Channel",img_red))

        img_red = skimage.util.img_as_ubyte(img_red)

        img_red = skimage.filters.gaussian(img_red, 0.1)
        
        if test :
            to_plot.append(("Gaussian Filter",img_red))

        img_red = enhance_contrast(img_red, disk(6))

        if test :
            to_plot.append(("Enhace Contrast",img_red))


        elevation_map = sobel(img_red)

        if test :
            to_plot.append(("gradientes",elevation_map))

        markers = np.zeros_like(img_red)

        s2 = f.target_set_mean(img_red, 8.5)

        markers[img_red < 150] = 1
        markers[img_red > s2] = 2
        
        seg_img = segmentation.watershed(elevation_map, markers)

        mask = (seg_img-1)>0

        if test :
            to_plot.append(("Sobel + WaterShed",seg_img))

        mask = binary_opening(mask, disk(2))
        mask = skimage.morphology.remove_small_objects(mask, 400)

        if test :
            to_plot.append(("Removing small objects",th.apply(img_red,mask)))

        mask = binary_closing(mask, disk(6))
        
        if test :
            to_plot.append(("Closing Region",th.apply(img[:,:,0],mask)))

        mask = skimage.morphology.remove_small_objects(mask, 1700)

        if test :
            to_plot.append(("Removing Big Objects",th.apply(img[:,:,0],mask)))

        mask = binary_closing(mask, disk(6))
        mask = msr.closest_prop(img[:,:,0], mask)

        if test :
            to_plot.append(("Removing non brighter region",th.apply(img[:,:,0],mask)))

        mask = binary_dilation(mask, disk(3))
        mask = binary_closing(mask, disk(12))

        if test :
            to_plot.append(("Dilate result region",th.apply(img[:,:,0],mask)))

        props = msr.props(img_red, mask)[0]

        minr, minc, maxr, maxc = props.bbox
        img_cut = img[minr:maxr, minc:maxc]

        if test:
            vi.plot_multy(to_plot,3,4, 'Sobel Watershed')


    if op == "cup":
        minr, minc, maxr, maxc = props.bbox
        img_green = img[minr:maxr, minc:maxc, 1]
        
        to_plot = []
        columns = 1

        if test :
            to_plot.append(("green channel",img_green))

        if not veins:
            columns = 4
            v_mask , _= segment_veins(img_green, test)
        
            img_aux = closing(img_green, disk(6))
            img_aux = dilation(img_aux, disk(6))
            img_aux2 = dilation(img_green, disk(3))
            
            if test :
                to_plot.append(("closed + dilated",img_aux))
                to_plot.append(("dilated",img_aux2))

            img_v_closed = th.apply(img_aux, v_mask)
            img_t_dilated = th.apply(img_aux2, v_mask==False)

            if test :
                to_plot.append(("veins part",img_v_closed))
                to_plot.append(("target part",img_t_dilated))

            img_green = img_v_closed + img_t_dilated

            if test :
                to_plot.append(("without veins",img_green))


        img_green = dilation(img_green, disk(6))
        img_green = enhance_contrast(img_green, disk(10))
        
        elevation_map = sobel(img_green)
        markers = np.zeros_like(img_green)

        s2 = f.target_set_mean(img_green, 8.5)

        markers[img_green < 150] = 1
        markers[img_green > s2] = 2
        
        seg_img = segmentation.watershed(elevation_map, markers)

        mask = (seg_img-1)>0

        if test :
            to_plot.append(("P-tile",th.apply(img_green, mask)))

        mask1 = mask
        mask = np.zeros(img[:,:, 1].shape)
        mask[minr:maxr, minc:maxc] = mask1

        if test:
            vi.plot_multy(to_plot,2,columns, 'cup')

    return (mask, img_cut, props)


#Este va muy bien
def otsu_local_method(img, op="disc", test = False):
    if op == "disc" or op == "cup":
        to_plot = []
        img_red = img[:,:,0]

        img_red = skimage.util.img_as_ubyte(img_red)
        if test :
            to_plot.append(("red_chan",img_red))

        img_red = skimage.filters.gaussian(img_red, 4.2)
        
        if test :
            to_plot.append(("gaussian f",img_red))

        img_red = enhance_contrast(img_red, disk(3))

        if test :
            to_plot.append(("enhace_contrast",img_red))

        
        """ Local Otsu """
        t_loc_otsu = otsu(img_red, disk(20))
        mask = img_red >= t_loc_otsu

        if test :
           to_plot.append(("Otsu local",th.apply(img_red, mask)))

        mask = binary_opening(mask, disk(2))
        mask = binary_closing(mask, disk(4))

        if test :
            to_plot.append(("binary open-close",th.apply(img_red, mask)))

        
        """ Chusing brightest region """
        mask = msr.closest_prop(img_red, mask)

        if test :
            to_plot.append(("brightest region",th.apply(img_red, mask)))

        maskfinal = binary_closing(mask, disk(20))
        
        if test :
           to_plot.append(("binary closing", th.apply(img_red, maskfinal)))

        img_red = th.apply(img_red, maskfinal)

        props = msr.props(img_red, maskfinal)[0]

        minr, minc, maxr, maxc = props.bbox
        img_cut = img[minr:maxr, minc:maxc]

        if test:
            vi.plot_multy(to_plot,3,3, 'Otsu Local chain')

        
    if op == "cup":
        minr, minc, maxr, maxc = props.bbox
        img_green = img[minr:maxr, minc:maxc, 1]
        
        to_plot = []

        if test :
            to_plot.append(("green channel",img_green))

        v_mask , _= segment_veins(img_green, test)
        
        img_aux = closing(img_green, disk(6))
        img_aux = dilation(img_aux, disk(6))
        img_aux2 = dilation(img_green, disk(3))
        
        if test :
            to_plot.append(("closed + dilated",img_aux))
            to_plot.append(("dilated",img_aux2))

        img_v_closed = th.apply(img_aux, v_mask)
        img_t_dilated = th.apply(img_aux2, v_mask==False)

        if test :
            to_plot.append(("veins part",img_v_closed))
            to_plot.append(("target part",img_t_dilated))

        img_green = img_v_closed + img_t_dilated

        if test :
            to_plot.append(("img_green",img_green))

        """ Local Otsu """
        t_loc_otsu = otsu(img_green, disk(20))
        mask = img_green >= t_loc_otsu
        mask = msr.closest_prop(img_green, mask)

        if test :
            to_plot.append(("Otsu local",th.apply(img_green, mask)))

        mask1 = mask
        mask = np.zeros(img[:,:, 1].shape)
        mask[minr:maxr, minc:maxc] = mask1

        if test:
            vi.plot_multy(to_plot,2,4, 'cup')
        

    return (mask, img_cut, props)

def segment_veins(img, test=False):
    to_plot = []
    """ Segmentamos venas """
    img_inverted = skimage.util.invert(img)

    if test :
        to_plot.append(("inverted green channel", img_inverted))

    img_inverted = exposure.equalize_adapthist(img_inverted, clip_limit=0.5)

    if test :
        to_plot.append(("high contrast", img_inverted))

    mask_invert = th.mask_otsu(img_inverted)

    if test :
        to_plot.append(("Global Otsu", th.apply(img, mask_invert)))

    """ creamos mascara para eliminar fondo """
    (ancho, alto) = img.shape

    rr, cc = ellipse(ancho/2, alto/2, (ancho/2)-5, (alto/2)-5)
    mask_background = np.zeros((ancho, alto))>0
    mask_background[rr, cc] = 1

    mask_invert = np.logical_and(mask_invert, mask_background)
    mask_invert = skimage.morphology.remove_small_objects(mask_invert, 200)

    if test :
        to_plot.append(("Mask without background", mask_invert))
    
    if test:
        vi.plot_multy(to_plot,1,4, 'Veins segmentation')

    return mask_invert, mask_background


def clustering_RAG(img, op="disc", test = False):
    
    to_plot = []

    img_red =img[:,:,0]
    
    if test :
        to_plot.append(("red_chan",img))

    (ancho, alto) = img_red.shape

    rr, cc = ellipse(ancho/2, alto/2, (ancho/2), (alto/2))
    mask_background = np.zeros((ancho, alto))>0
    mask_background[rr, cc] = 1
    
    """ Clustering k-means type """
    if op == "disc" :
        labels1 = segmentation.slic(img, mask = mask_background, n_segments=250, compactness=15, sigma=1, start_label=1)
        out1 = color.label2rgb(labels1, img)

        if test :
            to_plot.append(("Cluster1",out1))

        g = graph.rag_mean_color(img, labels1,  mode='similarity')
        labels2 = graph.cut_normalized(labels1, g)

    if op == "cup":
        labels1 = segmentation.slic(img, mask = mask_background, n_segments=100, compactness=10, sigma=1, start_label=1)
        out1 = color.label2rgb(labels1, img)

        if test :
            to_plot.append(("Cluster2",out1))

        g = graph.rag_mean_color(img, labels1)
        g = graph.rag_mean_color(img, labels1,  mode='similarity')
        labels2 = graph.cut_threshold(labels1, g, 500)



    out2 = color.label2rgb(labels2, img)

    if test :
        to_plot.append(("RAGs",out2))
    
    if test:
        vi.plot_multy(to_plot,1,3, 'K-means + RAGs')
        
def snakes(img, op="disc", test = False):

    if op == "disc":
        img_red = img[:,:,0]
        img_red = enhance_contrast(img_red, disk(10))
        img_red = gaussian(img_red, 5)

        mask = f.hand_made(img_red, 10, 2)
        mask = binary_opening(mask, disk(7))
        mask = skimage.morphology.remove_small_objects(mask, 1700)
        mask = msr.closest_prop(img_red, mask)

        if msr.props(img_red, mask): 
            props = msr.props(img_red, mask)[0]
            y0, x0 = props.centroid
            x02 = x0-props.minor_axis_length/4
            y_axis = x_axis = props.major_axis_length/2

        else :
            (x_axis, y_axis) = img_red.shape
            x0 = x02 = x_axis/2 -1
            y0 = y_axis/2 -1

        s = np.linspace(0, 2*np.pi, 400)
        r = y0 + y_axis*np.sin(s)
        c = x02 + x_axis*np.cos(s)
        init = np.array([r, c]).T
        
        snake = active_contour(img_red, init, boundary_condition='fixed',
                        alpha=0.1, beta=5, gamma=0.001)

    if op == "cup":
        img_red = img[:,:,1]
        img_red = enhance_contrast(img_red, disk(4))
        img_red = gaussian(img_red, 1)

        mask = f.hand_made(img_red, 10, 2)
        mask = binary_opening(mask, disk(7))
        mask = skimage.morphology.remove_small_objects(mask, 1700)
        mask = msr.closest_prop(img_red, mask)

        if msr.props(img_red, mask): 
            props = msr.props(img_red, mask)[0]
            y0, x0 = props.centroid
            x02 = x0-props.minor_axis_length/4
            y_axis = x_axis = props.major_axis_length/2

        else :
            (y_axis, x_axis) = img_red.shape
            x0 = x02 = x_axis/2
            y0 = y_axis/2 
            x_axis = x_axis/3 - 1  
            y_axis = y_axis/2 - 10
            

        s = np.linspace(0, 2*np.pi, 400)
        r = y0 + y_axis*np.sin(s)
        c = x02 + x_axis*np.cos(s)
        init = np.array([r, c]).T
        
        snake = active_contour(img_red, init, boundary_condition='fixed',
                        alpha=0.4, beta=10, gamma=0.001)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.imshow(img_red, cmap="gray")
    ax.plot(init[:, 1], init[:, 0], '--r', lw=3)
    ax.plot(snake[:, 1], snake[:, 0], '-b', lw=3)
    ax.set_xticks([]), ax.set_yticks([])
    ax.axis([0, img.shape[1], img.shape[0], 0])

    plt.show()
