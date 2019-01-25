import cv2
from matplotlib import pyplot as plt
import numpy as np

img = cv2.imread("C:\ok\Big_rat.jpg")
#capture video
cap = cv2.VideoCapture(1)

#create display windows
cv2.namedWindow('adjust')
cv2.namedWindow('new_window')

#empty function

def nothing(x):
    pass

empty_image = np.zeros((100,600,1),np.uint8)

#create trackbars
cv2.createTrackbar('max_val','adjust',0,255,nothing)
cv2.createTrackbar('min_val','adjust',0,255,nothing)

while(True):
    #escape condition
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break;
    #get image from camera
    ret,frame = cap.read()

    #make frame black and white
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    g = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    tmp = frame
    min = cv2.getTrackbarPos('min_val','adjust')
    max = cv2.getTrackbarPos('max_val','adjust')

    #apply  canny edge detection
    can = cv2.Canny(gray,min,max)
    #find contours
    image, contours, hierarchy = cv2.findContours(can, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    new_image = cv2.drawContours(tmp, contours,-1,(0,0,255),1)

    cv2.imshow('new_window',new_image)
    #cv2.imshow('adjust',empty_image)




#Kill windows before they eat up memory
cv2.destroyAllWindows
