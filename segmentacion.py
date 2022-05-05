from cv2 import THRESH_OTSU
import numpy as np
import cv2
from detecta import detect_peaks
from detecta import detect_seq
import matplotlib.pyplot as plt



class imagen:
    def __init__(self):
        
        self.a = cv2.imread("D:\\imagenes\\tomates.jpeg", 0)
        self.img = cv2.imread("D:\\imagenes\\tomates.jpeg", 3)
        b,g,r = cv2.split(self.img)
        self.aux = g.copy()
        self.aux_ = g.copy()
        self.dimensions = [self.a.shape[0],self.a.shape[1]]
        self.mask = self.img.copy()
        self.histogram = []
        self.aux_hist = np.zeros(256)
        self.kernel  = [[1,1,1],[1,1,1],[1,1,1]]
        #print(self.dimensions)
        

    def show(self):
        #print(self.mask)
        plt.imshow(self.a)
        plt.show()

    
    def showH(self):
        for i in range(0,self.dimensions[0]):
            for j in range(0,self.dimensions[1]):
                self.histogram.append(self.aux[i][j])
                self.aux_hist[self.aux[i][j]] += 1
                self.mask[i][j] = 0

        x = np.arange(0,256)
        plt.hist(self.histogram,x,edgecolor ="black")
        plt.xlim(xmin=205, xmax = 256)
        plt.ylim(ymin=0, ymax = 500)
        plt.show()

    def otsu(self):
        for i in range(0,self.dimensions[0]):
            for j in range(0,self.dimensions[1]):
                self.histogram.append(self.aux[i][j])
                self.aux_hist[self.aux[i][j]] += 1
                self.mask[i][j] = 0
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

        plt.show()


    def dimensionesR(self,a,b):
        self.resultadox = self.img.shape[0] - a
        self.resultadoy = self.img.shape[1] - b
        self.kernel = np.zeros((a,b))
        for i in range(0,a):
            for j in range(0,b):
                if(pow(i-np.floor(a/2),2)+pow(j-np.floor(b/2),2) <= pow(np.floor(a/2),2)):
                    self.kernel[i][j] = 1

    def dilatacion(self):
        self.dimensionesR(11,11)
        
        print(self.kernel)
        pixeles = []
        for i in range(0,self.img.shape[0]):
            for j in range(0,self.img.shape[1]):
                pixeles.clear()
                for z in range(0,11):
                    for k in range(0,11):
                        try:
                            pixeles.append(self.kernel[z][k] * self.aux_[i+z][j+k])
                        except:
                            pass
                self.aux[i][j] = max(pixeles)

        self.aux_ = self.aux
        plt.show()
        return self.aux

    def erosion(self):
        self.dimensionesR(11,11)
        
        print(self.kernel)
        pixeles = []
        for i in range(0,self.img.shape[0]):
            for j in range(0,self.img.shape[1]):
                pixeles.clear()
                for z in range(0,11):
                    for k in range(0,11):
                        try:
                            pixeles.append(self.kernel[z][k] * self.aux_[i+z][j+k])
                        except:
                            pass
                pixeles.sort()
                
                try:
                    self.aux[i][j] = pixeles[40]
                except:
                    pass
                
        self.aux_ = self.aux
        f, axarr = plt.subplots(2)
        axarr[0].imshow(self.img,cmap = "gray")
        axarr[1].imshow(self.aux,cmap = "gray")
        plt.show()
        return self.aux
        
                
    def  apertura(self):
        self.erosion()
        self.dilatacion()

        return self.aux
        
    def  cierre(self):
        self.dilatacion()
        self.erosion()
        return self.aux
                
    def gradiente(self):
        self.erosion()
        self.aux[:][:] = self.img[0][:][:] - self.aux[:][:]
        return self.aux

    def mouse(self,event,x,y,flags,parms):
        if event == cv2.EVENT_LBUTTONDOWN:
            print(x,y)
            self.neigborhood([y,x],0)
     
myImagen = imagen()


    
plt.imshow(myImagen.aux)
plt.show()
myImagen.erosion()
for i in range(0,myImagen.dimensions[0]):
    for j in range(0,myImagen.dimensions[1]):
        if(myImagen.aux[i][j] <= 210):
            myImagen.aux[i][j] = 0
        if( j >= 1000):
            myImagen.aux[i][j] = 0
        if( i >= 600):
            myImagen.aux[i][j] = 0

myImagen.erosion()
myImagen.showH()
x = len(detect_peaks(myImagen.aux_hist,show=True))
print(f"tomates: {x}")
plt.imshow(myImagen.aux,cmap="gray")
plt.show()
    
cv2.waitKey(0)






