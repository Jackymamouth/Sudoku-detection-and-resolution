# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 19:12:12 2019

@author: Ouistiti
"""

# =============================================================================
#
#   Training model for number recognition. Trains with OpenCv Knn. This script
#   needs labelized sample data.
#
# =============================================================================



# =============================================================================
# TRAING PART
# =============================================================================

import numpy as np
import numpy as np
import cv2
import matplotlib.pyplot as plt

#######   training part    ############### 
samples = np.loadtxt('train/generalsamples.data',np.float32)
responses = np.loadtxt('train/generalresponses.data',np.float32)
responses = responses.reshape((responses.size,1))

model = cv2.ml.KNearest_create()
model.train(samples,cv2.ml.ROW_SAMPLE,responses)

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
    
