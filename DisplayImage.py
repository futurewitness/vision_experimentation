import numpy as np
import cv2


image1 = cv2.imread("C:\Memes+more\More\shaq.jpg")
hsv = cv2.cvtColor(image1,cv2.COLOR_BGR2HSV)
## set threshold
lower_red = np.array([0,100,50])
upper_red = np.array([9000,1000,1000])
## Use range to differentiate objects from the background
mask = cv2.inRange(hsv, lower_red, upper_red)
## Bitwise-and mask and original image
res = cv2.bitwise_and(image1,image1,mask = mask)

cv2.imshow('frame',hsv)
cv2.waitKey(0)
cv2.imshow('mask',mask)
cv2.waitKey(0)
cv2.imshow('res',res)
cv2.imshow('a',image1)
cv2.waitKey(0)
cv2.destroyAllWindows()

