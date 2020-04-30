# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 13:58:49 2019

@author: Ouistiti
"""

# =============================================================================
#
#   The main Script (needs a trained model before hand for number recognition)
#
# =============================================================================

# =============================================================================
# Packages
# =============================================================================
#%%
import pandas as pd
import numpy as np
import PyPDF2
import tabula
from pdf2image import convert_from_path
import numpy as np
import cv2
from os import listdir
from PIL import Image
import matplotlib.pyplot as plt
import os
import sys

os.getcwd()
os.chdir('C:\\Users\\Ouistiti\\Documents\\CHALLENGE DATA\\data_challenge')
#dir_path = os.path.dirname(os.path.realpath(__file__))
#%%
# =============================================================================
# Find pages with the sudoku
# =============================================================================


for file_name in listdir("journaux"):
    pdfFileObj = open("journaux/"+file_name, 'rb')
    pdfReader = PyPDF2.PdfFileReader(pdfFileObj)
    sudoku_page=[]
    for i in range(pdfReader.getNumPages()):
        pageObj = pdfReader.getPage(i)
        az=str(pageObj.extractText())
        if(any([j=="sudoku" for j in [x.lower() for x in az.split()]])):
            sudoku_page.append(i)
    if(len(sudoku_page)==1):
        print("OK")
        pages = convert_from_path("journaux/"+file_name, 400) #400 is the Image quality in DPI (default 200)
        pages[sudoku_page[0]].save('sudoku_page/sudoku_page_'+file_name[:8] +'.png')


#%%
# =============================================================================
# Automaticaly find the sudokus position with OpenCv contours
# =============================================================================
        
for page_name in listdir("sudoku_page"):
    # Load in training data
    im = cv2.imread("sudoku_page/"+page_name)
    
    # image processing
    gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray,(5,5),0)
    thresh = cv2.adaptiveThreshold(blur,255,1,1,11,2)
    # Finding Contours
    contours,hierarchy = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    keep=[]
    for cnt in contours:
        if(cv2.contourArea(cnt)>50000): # big enough
            [x,y,w,h] = cv2.boundingRect(cnt)
            ratio=h/w
            print(ratio)
            if(abs(1-ratio)<=0.02): # see if it is of a square form
                i+=1
                keep.append((cv2.contourArea(cnt),cnt))
                
                
    [x,y,w,h] = cv2.boundingRect([j[1] for j in keep if j[0] == max([j[0] for j in keep])][0])
    crop_img = im[y:y+h, x:x+w]
    cv2.imwrite("sudoku/sudoku_"+page_name,crop_img)

#%%
# =============================================================================
# Diviser le sudoku et recuperer info de chaque case
# =============================================================================

# Import function that can predict the numbers 
exec(open('predict_number.py').read()) # CHECK LATER

# Function to check if the cell is blank or not
def is_blank_image(img1,black_thresh):
    pixels = img1.getdata()
    nblack = 0
    for pixel in pixels:
        if ((pixel[0] < black_thresh) | (pixel[1] < black_thresh) | (pixel[2] < black_thresh)):
            nblack += 1
    n = len(pixels)
    
    if (nblack / float(n)) > 0.05:
        decision = False
    else:
        decision = True
    return decision



for sudoku in listdir("sudoku"):
    cropped_img=cv2.imread("sudoku/"+sudoku) 
    height, width, channels = cropped_img.shape
    border=10
    width_len=round(width/9)
    height_len=round(height/9)
    
    original_sudoku=cv2.imread("sudoku/"+sudoku) 
    input_sudoku= [[] for i in range(9)]
    already_filed= [[] for i in range(9)]
    # 81 cells to determine 
#    sudoku = []
    for i in range(9):
        sudoku_temp = []
        for j in range(9):
            print(str(i)+","+str(j))
            # Setting the points for cropped image                 
            left = width_len*i+border
            top = height_len*j+border
            right = width_len*(i+1)-border
            bottom = height_len*(j+1)-border
            crop_img = cropped_img[top:bottom,left:right]
            cv2.imwrite("temp.png",crop_img)
            img1 = Image.open("temp.png")
            # Cropped image of above dimension     
            if(is_blank_image(img1,50)): # fixe number to 0 for empty case
                number=0
                print('0')
            else: # Recognize the number
                pred_num=predict_number(crop_img)
                cv2.putText(cropped_img,str(pred_num),(left,bottom),0,1.2,(0,0,255),2)
                number=pred_num
                print(pred_num)
            if number==0:
                already_filed[j].append(False)
            else:
                already_filed[j].append(True)
            input_sudoku[j].append(number)
    
    
    plt.figure(figsize = (25,14))
    plt.imshow(cropped_img)
    
   #%% 
    
    # =============================================================================
    # Solving algorithm
    # =============================================================================
    
    SudokuSolver(input_sudoku)
    
    # =============================================================================
    # Display results
    # =============================================================================
  #%%  
    heigh, width, channels = cropped_img.shape
    # 81 cases a déterminer 
#    sudoku = []
    for i in range(9):
        sudoku_temp = []
        for j in range(9):
            border=10
            width_len=round(width/9)
            height_len=round(height/9)
            print(str(i)+","+str(j))
            # Setting the points for cropped image 
            left = width_len*i+border
            top = height_len*j+border
            right = width_len*(i+1)-border
            bottom = height_len*(j+1)-border
            # Cropped image of above dimension 
            if(already_filed[j][i]): # fixe number to 0 for empty case
                cv2.putText(original_sudoku,str(input_sudoku[j][i]),(left,bottom),0,1.2,(0,0,255),2)
            else: # Recognize the number
                cv2.putText(original_sudoku,str(input_sudoku[j][i]),(left+10,bottom-5),0,1.5,(255,0,0),3)
                
    plt.figure(figsize = (25,14))
    plt.imshow(original_sudoku)
    
    cv2.imwrite("solved_sudokus/"+sudoku,original_sudoku)