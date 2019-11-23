# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 19:12:12 2019

@author: Ouistiti
"""

import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt


# Load in training data
im = cv2.imread('train_data_number.png')

# image processing
gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray,(5,5),0)
thresh = cv2.adaptiveThreshold(blur,255,1,1,11,2)


# Finding Contours
contours,hierarchy = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)


#cv2.boundingRect(contours[0])
#cv2.contourArea(contours[0])
#cv2.rectangle(im,(687,234),(726,272),(0,0,255),5)


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
plt.imshow(im)

responses = np.array(responses,np.float32)
responses = responses.reshape((responses.size,1))
print("training complete")

np.savetxt('generalsamples.data',samples)
np.savetxt('generalresponses.data',responses)



# =============================================================================
# TRAING PART
# =============================================================================

import numpy as np


#######   training part    ############### 
samples = np.loadtxt('generalsamples.data',np.float32)
responses = np.loadtxt('generalresponses.data',np.float32)
responses = responses.reshape((responses.size,1))

model = cv2.ml.KNearest_create()
model.train(samples,cv2.ml.ROW_SAMPLE,responses)

############################# testing part  #########################

im = cv2.imread('algerie.png')
out = np.zeros(im.shape,np.uint8)
gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray,(5,5),0)
thresh = cv2.adaptiveThreshold(blur,255,1,1,11,2)

contours,hierarchy = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

for cnt in contours:
#    if cv2.contourArea(cnt)>500:
    [x,y,w,h] = cv2.boundingRect(cnt)
#        if  h>110:
    cv2.rectangle(im,(x,y),(x+w,y+h),(0,0,255),2)
    roi = gray[y:y+h,x:x+w]
    roismall = cv2.resize(roi,(10,10))
#            plt.imshow(roismall)
    roismall = roismall.reshape((1,100))
    roismall = np.float32(roismall)
    retval, results, neigh_resp, dists = model.findNearest(roismall, k = 1)
    string = str(int((results[0][0])))
    cv2.putText(im,string,(x,y+h),0,1,(0,255,0))


plt.figure(figsize = (25,14))
plt.imshow(im)

plt.figure(figsize = (25,14))
plt.imshow(out)


predict_number(im)

def predict_number(im):
    out = np.zeros(im.shape,np.uint8)
    gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray,(5,5),0)
    thresh = cv2.adaptiveThreshold(blur,255,1,1,11,2)
    
    contours,hierarchy = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
    #    if cv2.contourArea(cnt)>500:
        [x,y,w,h] = cv2.boundingRect(cnt)
    #        if  h>110:
#        cv2.rectangle(im,(x,y),(x+w,y+h),(0,0,255),2)
        roi = gray[y:y+h,x:x+w]
        roismall = cv2.resize(roi,(10,10))
    #            plt.imshow(roismall)
        roismall = roismall.reshape((1,100))
        roismall = np.float32(roismall)
        retval, results, neigh_resp, dists = model.findNearest(roismall, k = 1)
#        string = str(int((results[0][0])))
#        cv2.putText(out,string,(x,y+h),0,1,(0,255,0))

    return int((results[0][0]))
    
