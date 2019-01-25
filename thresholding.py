import cv2
import numpy as np

while(True):
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break;
    img = cv2.imread("C:\Memes+more\more\skatha.png")
    lower_limit = ([0,0,0])
    upper_limit = ([100,100,100])
    frame = img[1000:2500,2000:3000,]
    gray = cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY)
    cv2.imshow('frame',frame)
    cv2.imshow('gray',gray)

