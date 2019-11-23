# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 13:58:49 2019

@author: Ouistiti
"""

# =============================================================================
# Packages
# =============================================================================
import pandas as pd
import numpy as np
import PyPDF2
import tabula
from pdf2image import convert_from_path
import numpy as np
import cv2


# =============================================================================
# Extraction
# =============================================================================
pages = convert_from_path('20190830_PAR.pdf', 400) #400 is the Image quality in DPI (default 200)
pages[16].save("sample.png")
# =============================================================================
# Sudoku Solving process
# =============================================================================

# The preprocessing part starts with converting the given image to grayscale in
# order to simplify processing.

import fitz
pdffile = "20190830_PAR.pdf"
doc = fitz.open(pdffile)
page = doc.loadPage(16) #number of page

page.getFontList()
#pix = page.getPixmap()

#pix.size
#pix.height
#pix.width

#pix.writeImage("bob.png")





from PIL import Image
img = Image.open("sample.png")
area = (1350, 3370, 2135, 4155) # Zone a CROP
cropped_img2 = img.crop(area)
cropped_img2.show()

cropped_img2.save('crop.png')



# =============================================================================
# Number recognition part 
# =============================================================================

# =============================================================================
# Dividing into segments part
# =============================================================================

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

# Size of the image in pixels (size of orginal image) 
# (This is not mandatory)

cropped_img = Image.open("crop.png")
width, height = cropped_img.size 

original_sudoku=cv2.imread("crop.png") 
input_sudoku= [[] for i in range(9)]
already_filed= [[] for i in range(9)]
# 81 cases a déterminer 
sudoku = []
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
        # (It will not change orginal image) 
        img1 = cropped_img.crop((left, top, right, bottom))
#        img2=imageprepare(img1)
        # SEE if image is blank or not
        if(is_blank_image(img1,50)): # fixe number to 0 for empty case
            number=0
            print('0')
        else: # Recognize the number
            img1.save("two.png")
            paths=r"C:\Users\Ouistiti\Documents\CHALLENGE DATA\data_challenge\Journaux\two.png"
            mg  = cv2.imread(paths) 
            print(predict_number(mg))
            cv2.putText(original_sudoku,str(predict_number(mg)),(left,bottom),0,1.2,(0,0,255),2)
            number=predict_number(mg)
        if number==0:
            already_filed[j].append(False)
        else:
            already_filed[j].append(True)
        input_sudoku[j].append(number)


plt.figure(figsize = (25,14))
plt.imshow(original_sudoku)


##### SOLVE SODUKU ALGO PART
input_sudoku2=input_sudoku.copy()
solveSudoku(input_sudoku)
input_sudoku2


### Pretty print the solution



filled_sudoku=cv2.imread("crop.png")
width, height = cropped_img.size 
# 81 cases a déterminer 
sudoku = []
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
            cv2.putText(filled_sudoku,str(input_sudoku[j][i]),(left,bottom),0,1.2,(0,0,255),2)
        else: # Recognize the number
            cv2.putText(filled_sudoku,str(input_sudoku[j][i]),(left+10,bottom-5),0,1.5,(255,0,0),3)
            
plt.figure(figsize = (25,14))
plt.imshow(filled_sudoku)