import cv2 as cv
import numpy as np
import scipy
from PIL import Image
from helper import load_image 
from helper import concatenate_images
from scipy import signal
num_images = 51
   
#Blurring filters to smoothen binary images
def processing0(): 
  
    for k in range(1, num_images):
        image = cv.imread("../data/smoothing/sub_" + str(k) + ".png",0)
        ret, blurred_image = cv.threshold(image, 125, 255, cv.THRESH_BINARY)  
        blurred_image = cv.pyrUp(cv.pyrUp(image))
        blurred_image = cv.medianBlur(blurred_image, 27)
        blurred_image = cv.pyrDown(cv.pyrDown(blurred_image))

        ret, smoothed_image = cv.threshold(blurred_image, 200, 255, cv.THRESH_BINARY)
        cv.imwrite("../data/smoothing_results/sub_" +str(k) + ".png", smoothed_image)


def processing1(): 
    windowSize = 20
    for k in range(1, num_images):
        binaryImage = load_image("../data/smoothing/sub_" + str(k) + ".png")
        kernel = np.ones(windowSize) / (windowSize^2) 
        shape = binaryImage.shape
        
        blurryImage = np.convolve(np.ravel(binaryImage), kernel, 'same')
        binaryImage = blurryImage > 0.5; #Rethreshold
        binaryImage = np.reshape(binaryImage, shape)
        
        new_p = Image.fromarray(np.uint8(np.array(binaryImage) * 255))
        new_p.save("../data/smoothing_results/sub_" +str(k) + ".png")
        
if __name__ == "__main__":
    im1 = processing0()
   #im2 = processing1()

    
