#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 24 2020
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt

path =[]

cap = cv2.VideoCapture('MVI_1452.mp4')

# params for ShiTomasi corner detection
feature_params = dict( maxCorners = 100,
                       qualityLevel = 0.5, # Updated the quality level 
                       minDistance = 7,
                       blockSize = 7 )

# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15,15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Create some random colors
color = np.random.randint(0,255,(100,3))

# Take first frame and find corners in it
ret, old_frame = cap.read()
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)

# Create a mask image for drawing purposes
mask = np.zeros_like(old_frame)
black_background = np.zeros(old_frame.shape, dtype=np.uint8)

while(cap.isOpened()):
    ret,frame = cap.read()
    if not ret:
        break
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # create a window of size (3,3) ---> Structuring element is a square
    kernel = np.ones((3,3), np.uint8) 

    # Apply dilation operation with 2 iterations
    frame_gray = cv2.dilate(frame_gray, kernel, iterations=2)

    # calculate optical flow
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

    # Select good points
    good_new = p1[st==1]
    good_old = p0[st==1]

    # draw the tracks
    for i,(new,old) in enumerate(zip(good_new,good_old)):
        if i==11: # feature 11 is a hardcoded feature which almost resembles the ant with the stone in the scene, hence we can eliminate this problem with a morphological operation of dilation
            a,b = new.ravel()
            path.append([a,b])  
            c,d = old.ravel()
            mask = cv2.line(mask, (a,b),(c,d), color[i].tolist(), 2)
            cv2.line(black_background, (a,b),(c,d),[0,0,255],2 )
            frame = cv2.circle(frame,(a,b),5,color[i].tolist(),-1)
    img = cv2.add(frame,mask)

    cv2.imshow('frame',img)

    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

    # Now update the previous frame and previous points
    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1,1,2)

# print(path)
xcoords = []
ycoords = []
for [x,y] in path:
    xcoords.append(x)
    ycoords.append(y)

plt.plot(xcoords,ycoords)
plt.show()
cap.release()
cv2.destroyAllWindows()
