import numpy as np
import cv2
from matplotlib import pyplot as plt

cap = cv2.VideoCapture(1)
empty_image = np.zeros((300,255,1),np.uint8)

ones_image = np.zeros((3,3),np.uint8)

ellipse_kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))


def nothing(x):
    pass

cv2.namedWindow('adjust')

cv2.createTrackbar('H_lower','adjust',0,255,nothing)
cv2.createTrackbar('H_upper','adjust',0,255,nothing)

cv2.createTrackbar('S_lower','adjust',0,255,nothing)
cv2.createTrackbar('S_upper','adjust',0,255,nothing)

cv2.createTrackbar('V_lower','adjust',0,255,nothing)
cv2.createTrackbar('V_upper','adjust',0,255,nothing)

cv2.createTrackbar('thresh_lower','adjust',0,255,nothing)
cv2.createTrackbar('thresh_upper','adjust',0,255,nothing)

while(True):
    ret,frame = cap.read()


    h_lower = cv2.getTrackbarPos('H_lower','adjust')
    h_upper = cv2.getTrackbarPos('H_upper','adjust')

    s_lower = cv2.getTrackbarPos('S_lower','adjust')
    s_upper = cv2.getTrackbarPos('S_upper','adjust')

    v_lower = cv2.getTrackbarPos('V_lower','adjust')
    v_upper = cv2.getTrackbarPos('V_upper','adjust')

    lower_limit = np.array([h_lower,s_lower,v_lower])
    upper_limit = np.array([h_upper,s_upper,v_upper])

    ##frame =cv2.medianBlur(frame,5)
    ##
    hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv,lower_limit,upper_limit)
    ##apply mask
    ret = cv2.bitwise_and(frame, frame, mask=mask)
    gray = cv2.cvtColor(ret, cv2.COLOR_BGR2GRAY)
    ## thresholding
    blur = cv2.GaussianBlur(gray,(5,5),0)
    #blur = cv2.bilateralFilter(gray, 9, 75, 75)
    ret1,th_adaptive = cv2.threshold(blur,0,255,cv2.THRESH_BINARY)
    laplace_transformation = cv2.Laplacian(th_adaptive,cv2.CV_64F)

    ## blurred lines


    eroded_image = cv2.erode(ret,ellipse_kernel, iterations=2)
    #dilate_image = cv2.dilate(eroded_image,ones_image,iterations = 1)
    opened_image = cv2.morphologyEx(ret,cv2.MORPH_OPEN,ellipse_kernel)
    #plt.subplot(2,2,1),plt.imshow(ret,cmap = gray)
    #lt.title('Original'),plt.xticks([]),plt.yticks([])
    #plt.show()
    #cv2.imshow('camera_feed',ret)
    #cv2.imshow('adjust',empty_image)
    cv2.imshow('frame',laplace_transformation)
    #cv2.imshow('erosion_test',eroded_image)
    #cv2.imshow('dilate_test',opened_image)

    #cv2.imshow('frame2',only_some)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()