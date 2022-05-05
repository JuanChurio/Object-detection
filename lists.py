from math import ceil
import cv2
from cv2 import magnitude
from cv2 import normalize
import numpy as np
from matplotlib import pyplot as plt

class Image:
    def __init__(self):
        self.img = cv2.imread("D:\\imagenes\\prueba\\veiled.jpeg", 0)
        self.aux = cv2.imread("D:\\imagenes\\prueba\\veiled.jpeg", 0)
        self.dimensions = [self.img.shape[0],self.img.shape[1]]
        self.center = [self.dimensions[0]/2,self.dimensions[1]/2]
        self.histogram = []
        self.aux_hist = np.zeros(256)
        for i in range(0,self.dimensions[0]):
            for j in range(0,self.dimensions[1]):
                self.histogram.append(self.img[i][j])
                self.aux_hist[self.img[i][j]] += 1


    def showHistogram(self):

        plt.hist(self.histogram,edgecolor = 'black',bins=25)
        plt.title("Histogram")
        plt.xlabel("value")
        plt.xlabel("#")
        plt.show()



    def setDimensions(self):
        self.m = int(input("input m: "))
        self.n = int(input("input n: "))
        self.ResultSizeM = self.dimensions[0]-self.m+1
        self.ResultSizeN = self.dimensions[1]-self.n+1
        print("m",self.ResultSizeN)
        print("n",self.ResultSizeN)

    def average(self):
        print("funciona")
        ay = self.img.copy()
        self.setDimensions()
        for i in range(0,self.ResultSizeM):
            for j in range(0,self.ResultSizeN):

                counter = 0
                for z in range(0,self.m):
                    for j in range(0,self.n):
                        counter += self.img[i][j]
            ay[i][j] = counter/(self.m*self.n)
        cv2.imshow("gauss",ay)
        
    def fourier(self,option):

        dft = cv2.dft(np.float32(self.img), flags=cv2.DFT_COMPLEX_OUTPUT)
        dft_shift = np.fft.fftshift(dft)
        self.magnitude = dft_shift.copy()

        for i in range(0,self.dimensions[0]):
            for j in range(0,self.dimensions[1]):
                self.magnitude[i][j] = 20*np.log(pow((pow(dft_shift[i][j][0],2)+pow(dft_shift[i][j][1],2)),0.5))/255
        self.real,trash = cv2.split(self.magnitude)
        del trash
        self.mask = dft_shift.copy()
        cv2.imshow("frecuency",self.real)
        if(option == 1):
            self.lowPass()
        if(option == 2):
            self.highPass()
        if(option == 3):
            self.bandPass()
        if(option == 4):
            self.bandReject()
        if(option == 5):
            self.guassianFilter()
        #print(self.mask)
        fshift   = self.mask
        f_ishift = np.fft.ifftshift(fshift)
        img_back = cv2.idft(f_ishift)
        img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])
        plt.imshow(img_back,cmap="gray")
        plt.show()
        
        
    def highPass(self):
        r = int(input("input r: "))
        for i in range(0,self.dimensions[0]):
            for j in range(0,self.dimensions[1]):

                if( pow(i-self.center[0],2)+pow(j-self.center[1],2) <= pow(r,2) ):
                    self.real[i][j] = 0
                    self.mask[i][j] = 0
                    self.magnitude[i][j] = 0

    def lowPass(self):
        r = int(input("input r: "))
        for i in range(0,self.dimensions[0]):
            for j in range(0,self.dimensions[1]):
                if( pow(i-self.center[0],2)+pow(j-self.center[1],2) >= pow(r,2) ):

                    self.real[i][j] = 0
                    self.mask[i][j] = 0
                    self.magnitude[i][j] = 0

    def bandPass(self):
        r1 = int(input("input r1: "))
        r2 = int(input("input r2: "))
        for i in range(0,self.dimensions[0]):
            for j in range(0,self.dimensions[1]):
                if( pow(i-self.center[0],2)+pow(j-self.center[1],2) >= pow(r1,2) and pow(i-self.center[0],2)+pow(j-self.center[1],2) <= pow(r2,2) ):

                    self.real[i][j] = 0
                    self.magnitude[i][j] = 0
                    self.mask[i][j] = 0
    
    def bandReject(self):
        r1 = int(input("input r1: "))
        r2 = int(input("input r2: "))
        for i in range(0,self.dimensions[0]):
            for j in range(0,self.dimensions[1]):

                if( pow(i-self.center[0],2)+pow(j-self.center[1],2) <= pow(r1,2) and pow(i-self.center[0],2)+pow(j-self.center[1],2) >= pow(r2,2) ):
                    self.real[i][j] = 0
                    self.mask[i][j] = 0
                    self.magnitude[i][j] = 0

    def guassianFilter(self):
        deviation = float(input("input deviation: "))
        matrix = np.zeros((self.dimensions[0],self.dimensions[1]))
        for i in range(0,self.dimensions[0]):
            for j in range(0,self.dimensions[1]):
                matrix[i][j] = (1/(deviation*pow(44/7,0.5)))*pow(np.e,-((pow(i-0.5*self.dimensions[0],2)+pow(j-0.5*self.dimensions[1],2))/(44/7)*pow(deviation,2)))
                matrix[i][j] = np.abs(matrix[i][j])
                self.mask[i][j] = self.mask[i][j] * matrix[i][j] 
                
        plt.imshow(matrix)
        plt.show()

        copy = self.mask
        sum = self.mask.sum()

        for i in range(0,self.dimensions[0]):
            for j in range(0,self.dimensions[1]):
                copy[i][j]= self.mask[i][j]/sum
        self.mask = copy.copy()

    def otsu(self):
        otsu_list = np.zeros(256)
        normalize_aux = 0
        background = [0,0]
        foreground = [0,0]
        dis = 0
        histogram_aux = []
        for i in range(1,256):
            dis = i
            for j in range(0,dis):
                background[0] += self.aux_hist[i]
                background[1] += self.aux_hist[i]*j
            for j in range(dis,256):
                foreground[0] += self.aux_hist[i]
                foreground[1] += self.aux_hist[i]*j

            background[1] = background[1]/background[0]
            foreground[1] = foreground[1]/foreground[0]     
             ##   
            foreground[0] = foreground[0]/(self.dimensions[0]*self.dimensions[1])
            background[0] = background[0]/(self.dimensions[0]*self.dimensions[1])
            otsu_list[i] = foreground[0]*background[0]*pow(background[1]-foreground[1],2)
            normalize_aux += otsu_list[i]


        
        otsu_list = otsu_list /normalize_aux 
        criticalV = np.argmax(otsu_list)
        print(criticalV)

        
        for i in range(0,self.dimensions[0]):
            for j in range(0,self.dimensions[1]):
                if(self.aux[i][j]  <= criticalV+2):
                    self.aux[i][j] = 0
                histogram_aux.append(self.aux[i][j])

        f, axarr = plt.subplots(2)
        axarr[0].imshow(self.img,cmap = "gray")
        axarr[1].imshow(self.aux,cmap = "gray")
        plt.show()
        

    def showOriginal(self):
        cv2.imshow("Original",self.img)

    def showAux(self):
        cv2.imshow("Aux",self.aux)

    def minimum(self):
        self.setDimensions()
        mylist = []
        for i in range(0,self.ResultSizeM):
            for j in range(0,self.ResultSizeN):

                mylist.clear()
                for z in range(0,self.m):
                        for k in range(0,self.n):
                            mylist.append(self.aux[i+z][j+k])

                self.aux[i][j] = min(mylist)
        self.showAux()
        
    def maximum(self):
        self.setDimensions()
        mylist = []
        for i in range(0,self.ResultSizeM):
            for j in range(0,self.ResultSizeN):

                mylist.clear()
                for z in range(0,self.m):
                    for k in range(0,self.n):
                        mylist.append(self.aux[i+z][j+k])

                self.aux[i][j] = max(mylist)
        self.showAux()
    
    def median(self):
        self.setDimensions()
        mylist = []
        for i in range(0,self.ResultSizeM):
            for j in range(0,self.ResultSizeN):

                mylist.clear()
                for z in range(0,self.m):
                    for k in range(0,self.n):
                        mylist.append(self.aux[i+z][j+k])
                mylist.sort()
                self.aux[i][j] = mylist[ceil((self.m*self.n)/2)]
        self.showAux()


def menu():
    ims = Image()

    print("Menu")
    print("1.Show Original")
    print("2.Minimum")
    print("3.Maximum")
    print("4.Median")
    print("5.Fourier")
    print("6.average")
    print("7.Show histogram")
    print("8.Otsu's method")
    
    aux = int(input("Input: "))

    if(aux == 1):
        ims.showOriginal()
    if(aux == 2):
        ims.minimum()
    if(aux == 3):
        ims.maximum()
    if(aux == 4):
        ims.median()
    if(aux == 5):
        print("1.lowPass")
        print("2.highPass")
        print("3.bandPass")
        print("4.bandReject")
        print("5.lowpass guass")
        aux = int(input("Input: "))
        ims.fourier(aux)
    if(aux == 6):
        ims.average()
    if(aux == 7):
        ims.showHistogram()
    if(aux == 8):
        ims.otsu()


menu()
cv2.waitKey(0)