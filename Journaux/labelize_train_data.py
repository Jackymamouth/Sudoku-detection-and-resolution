# -*- coding: utf-8 -*-
"""
Created on Sun Feb 23 14:43:10 2020

@author: Ouistiti
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt


# Load in training data
im = cv2.imread('train/train_data_number.png')

# image processing
gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray,(5,5),0)
thresh = cv2.adaptiveThreshold(blur,255,1,1,11,2)


# Finding Contours
contours,hierarchy = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)



samples =  np.empty((0,100))
responses=[]
i=1
for cnt in contours:
#    if cv2.contourArea(cnt)>500:
    [x,y,w,h] = cv2.boundingRect(cnt)
#        if  h>110:
    print(i)
    i+=1            
    cv2.rectangle(im,(x,y),(x+w,y+h),(0,0,255),2)
    roi = gray[y:y+h,x:x+w]
    roismall = cv2.resize(roi,(10,10))
    cv2.imshow('gray',roismall)
    key=cv2.waitKey()
    if key : cv2.destroyAllWindows()#cv2.waitKey(0)
    responses.append(int(chr(key)))
    
    sample = roismall.reshape((1,100))
    samples = np.append(samples,sample,0)


plt.figure(figsize = (25,14))
plt.imshow(thresh)
#cv2.imwrite("koll2.png",roismall)
responses = np.array(responses,np.float32)
responses = responses.reshape((responses.size,1))
print("training complete")

np.savetxt('train/generalsamples.data',samples)
np.savetxt('train/generalresponses.data',responses)