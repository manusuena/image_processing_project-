import cv2
from skimage import io
from skimage.color import rgb2gray
from skimage.filters import threshold_otsu
import matplotlib.pyplot as plt
import numpy as np
import skimage
from skimage.transform import probabilistic_hough_line
from skimage.feature import canny
from matplotlib import cm

avengers = cv2.imread('avengers_imdb.jpg')
height = avengers.shape[0]  # gets height of the image
width = avengers.shape[1]   # gets the width of the image
print(width, height)

gray_avengers = cv2.cvtColor(avengers, cv2.COLOR_BGR2GRAY)   # converst the image to grayscale
(thresh, im_bw) = cv2.threshold(gray_avengers, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)   # covert the image into binary using threshholding

cv2.imshow('Original image', avengers)
cv2.imshow('binary image', im_bw)
cv2.imshow('Gray image', gray_avengers)
cv2.waitKey(0)

#cv2.imwrite("im_bw.jpg", im_bw)
#cv2.imwrite("gray_avengers.jpg", gray_avengers)


bush = cv2.imread('bush_house_wikipedia.jpg')

noise = skimage.util.random_noise(bush, mode='gaussian', seed=None, clip=True, var=0.1) # adds gaussian noise to the image
gauss = skimage.filters.gaussian(noise, sigma=1)  # filters the noise using a gaussian mask
uniform = cv2.blur(noise, (9, 9))  # filters the noise using a uniform  smoothing mask

plt.subplot(131), plt.imshow(uniform), plt.title('noise')
plt.subplot(132), plt.imshow(uniform), plt.title('gauss')
plt.subplot(133), plt.imshow(uniform), plt.title('uniform')
plt.show()

forest = cv2.imread('forestry_commission_gov_uk.jpg')

image = cv2.cvtColor(forest, cv2.COLOR_BGR2RGB)
pixel_values = image.reshape((-1, 3))  # reshape the image to a 2D array of pixels and 3 color values RGB
pixel_values = np.float32(pixel_values)  # convert to float
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)  # stop criteria
k = 5   # number of clusters
_, labels, (centers) = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
centers = np.uint8(centers)  # convert back to 8 bit values
labels = labels.flatten()  # flatten the labels array
segmented_image = centers[labels.flatten()]  # convert all pixels to the color of the centroids
segmented_image = segmented_image.reshape(image.shape)  # reshape back to the original image dimension
plt.imshow(segmented_image)
plt.show()


tennis = cv2.imread('rolland_garros_tv5monde.jpg')

gray = cv2.cvtColor(tennis,cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray,50,150,apertureSize = 3)  # performs canny edge detection on the image
lines = cv2.HoughLinesP(edges,1,np.pi/180,100,minLineLength=100, maxLineGap=6)  # performs hough lines detection
for line in lines:
    x1,y1,x2,y2 = line[0]
    cv2.line(tennis,(x1,y1),(x2,y2),(0,255,0),2)

cv2.imshow('canny',edges)
cv2.imshow('houghlines',tennis)
cv2.waitKey(0)
#cv2.imwrite('canny.jpg',edges)
#cv2.imwrite('houghlines.jpg',tennis)



