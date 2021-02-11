
import os
import re
import imageio
from skimage import data
import numpy as np

import threading 
import time 


class DataTool:
    def __init__(self, dir):
        self.dir = dir
        self.data = dict()
        self.names = np.array([])

    def loadData(self, pat1):
        files = os.listdir(self.dir)
        r1 = re.compile(pat1)
        list1 = list(filter(r1.search, files))
        aux = [ DataTool.loadImages(x, self.dir+"/",".png", "_cup.png", "_disc.png") for x in list1]
        self.data = dict(zip(list1, aux))
        self.names = list1

    @staticmethod
    def loadImages(image,path,sep='.png', ext1='', ext2='', gray = True):
        
        try:
            img = imageio.imread(path+image)

            if ext1 != '':
                cup = path+image.split(sep,1)[0]+"_cup.png"
                cup_img = imageio.imread(cup)
            if ext2 != '':
                disc = path+image.split(sep,1)[0]+"_disc.png"
                disc_img = imageio.imread(disc)
            return {"img":img, "cup":cup_img, "disc":disc_img}
        except Exception:
            print("File not found")
        return None

