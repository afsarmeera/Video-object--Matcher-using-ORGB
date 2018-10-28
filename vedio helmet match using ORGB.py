# -*- coding: utf-8 -*-
"""
Created on Mon Oct 15 16:34:27 2018

@author: afsar
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Oct 12 19:54:44 2018



@author: afsar
"""
import cv2

import numpy as np
import numpy as np
import cv2
from matplotlib import pyplot as plt
img2 = cv2.imread('C:/Users/afsar/Desktop/VEGA-HELMET-Offroad-helmets-DAnthrasite1-1.jpg',0) # trainImage

cap = cv2.VideoCapture("C:/Users/afsar/Downloads/videoplayback.mp4")
 
subtractor = cv2.createBackgroundSubtractorMOG2(history=20, varThreshold=25, detectShadows=True)
 
while True:
    _, frame = cap.read()
    orb = cv2.ORB_create()

      # find the keypoints and descriptors with ORB
    kp1, des1 = orb.detectAndCompute(frame,None)
    kp2, des2 = orb.detectAndCompute(img2,None)
    
      # create BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    
     # Match descriptors.
    matches = bf.match(des1,des2)
    matches = sorted(matches, key = lambda x:x.distance)
    img3 = cv2.drawMatches(frame,kp1,img2,kp2,matches ,None, flags=2)
    cv2.imshow("Frame", img3)
    #cv2.imshow("mask", mask)
    key = cv2.waitKey(30)
    if key == 27:
        
        break
        
    
    
    #break
 
cap.release()
cv2.destroyAllWindows()