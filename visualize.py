import matplotlib.pyplot as plt
import numpy as np
import math

def histogram(image, i_bins=32, i_range=[0,256]):
    fig, ax = plt.subplots(1, 1)
    ax.hist(image.ravel(), bins=i_bins, range=i_range)
    ax.set_xlim(0, 256)
    return fig, ax

def plot_props(img, regions):
    _, ax = plt.subplots()
    ax.imshow(img)

    for props in regions:
        y0, x0 = props.centroid
        orientation = props.orientation

        x1 = x0 + 0.5 * props.minor_axis_length
        y1 = y0 - 0.5 * props.minor_axis_length
        x2 = x0 - 0.5 * props.major_axis_length
        y2 = y0 - 0.5 * props.major_axis_length

        ax.plot((x0, x1), (y0, y1), '-r', linewidth=1.5)
        ax.plot((x0, x2), (y0, y2), '-r', linewidth=1.5)
        ax.plot(x0, y0, '.g', markersize=3)

        minr, minc, maxr, maxc = props.bbox
        bx = (minc, maxc, maxc, minc, minc)
        by = (minr, minr, maxr, maxr, minr)
        ax.plot(bx, by, '-b', linewidth=1.5)

def pltSnake(img, init, snake):
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(img, cmap="gray")
    ax.plot(init[:, 1], init[:, 0], '--r', lw=3)
    ax.plot(snake[:, 1], snake[:, 0], '-b', lw=3)
    ax.set_xticks([]), ax.set_yticks([])
    ax.axis([0, img.shape[1], img.shape[0], 0])

def plot_multy(figures, r, c, title):
    fig = plt.figure(figsize=(80,80))
    fig.suptitle(title, fontsize=16)

    ax = []
    count = 0
    for (stitle,img) in figures:
        ax.append( fig.add_subplot(r, c, count+1) )
        ax[count].set_title(stitle)  # set title
        ax[count].imshow(img, cmap='gray')
        count = count + 1
            
def plot_scatter(algs, r, c, title):
    fig = plt.figure(figsize=(80,80))
    fig.suptitle(title, fontsize=16)

    ax = []
    count = 0
    for ((alg, (x_title, x),(y_title, y))) in algs:
        ax.append(fig.add_subplot(r, c, count+1))
        ax[count].set_title(alg.__name__)
        plt.xlabel(x_title)
        plt.ylabel(y_title)
        ax[count].scatter(x,y)
        count = count + 1

def plotJet(image):
    plt.imshow(image, cmap='jet')
    plt.axis('off')

def pltImage(img) -> None:
    plt.figure()
    plt.imshow(img, cmap='gray')

def pltImages(imgs) -> None:
    for img in imgs:
        pltImage(img)

def show() -> None:
    plt.show()

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.144])

