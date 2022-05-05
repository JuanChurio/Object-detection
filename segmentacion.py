import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread("D:\\imagenes\\prueba\\veiled.jpeg", 0)
mask = np.zeros(img.shape, dtype='float32')
dimensions = [img.shape[0],img.shape[1]]

numElement = 0

for i in range(1,dimensions[0]-1):
    for j in range(1,dimensions[1]-1):
        up = img[i][j+1]
        down = img[i][j-1]
        r = img[i+1][j]
        l = img[i-1][j]
        if(img == up and img == down and img == r and img == l and img != 1 and mask  == ):


