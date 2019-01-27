import cv2
from matplotlib import pyplot as plt
import numpy as np

img = cv2.imread('shapes.png')
image2 = cv2.imread('shapes.png')
#capture video
#cap = cv2.VideoCapture(1)

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
        break
    #get image from camera
   # ret,frame = cap.read()

    #make frame black and white
    #gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    g = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    min = cv2.getTrackbarPos('min_val','adjust')
    max = cv2.getTrackbarPos('max_val','adjust')

    #apply  canny edge detection
    x = cv2.GaussianBlur(g,(5,5),0)
    can = cv2.Canny(x,min,max,apertureSize = 3)
    #find contours
    image, contours, hierarchy = cv2.findContours(can, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    cont = contours[1]
    peri = cv2.arcLength(cont, True)
    approx = cv2.approxPolyDP(cont, 0.04 * peri, True)

    new_image = cv2.drawContours(image2,[cont],0,(0,255,0),1)
    cv2.putText(image2,str(len(approx)), (0, 100), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (255, 255, 255), 2)
    cv2.imshow('new_window',image2)
    cv2.imshow('adjust',can)




#Kill windows before they eat up memory
cv2.destroyAllWindows
