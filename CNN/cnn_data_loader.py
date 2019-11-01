import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from tqdm import tqdm

DATADIR = "D:\Images"

CATEGORIES = ["Fire", "Smoke"]

for category in CATEGORIES:  
    path = os.path.join(DATADIR,category)  # create path to categories
    for img in os.listdir(path):  # iterate over each image per dogs and cats
        img_array = cv2.imread(os.path.join(path,img) ,cv2.IMREAD_GRAYSCALE)  # convert to array
        plt.imshow(img_array, cmap='gray')  # graph it
        plt.show()  # display!

        #break  # we just want one for now so break
    break  #...and one more!