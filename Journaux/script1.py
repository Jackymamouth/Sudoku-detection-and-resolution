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


# save a image using extension 
#cropped_img.save("geeks.png") 


#algerie = Image.open(r"C:\Users\Ouistiti\Desktop\algerie.png")  
#algerie.show()

boby(algerie)
paths=r"C:\Users\Ouistiti\Desktop\algerie.png"
paths=r"C:\Users\Ouistiti\Documents\CHALLENGE DATA\data_challenge\Journaux\two.png"
mg  = cv2.imread(paths)       
plt.imshow(thresh,'gray')
gray = cv2.cvtColor(mg,cv2.COLOR_BGR2GRAY)

# preprocess image
thresh = cv2.adaptiveThreshold(gray,255,1,1,11,2)

boby(thresh)



width = 28
height = 28
dim = (width, height)
# resize image
resized = cv2.resize(thresh, dim, interpolation = cv2.INTER_AREA)


plt.imshow(resized,'gray')

jack= boby(resized/255)
jack2=jack/255
def boby(re):
    img = re
    img = np.resize(img, (28,28,1))
    im2arr = np.array(img)
    im2arr = im2arr.reshape(1,28,28,1)
    return im2arr

# =============================================================================
# Number recognition part 
# =============================================================================


# Plot ad hoc mnist instances
import numpy
from keras.datasets import mnist
import matplotlib.pyplot as plt
# load (downloaded if needed) the MNIST dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()
# plot 4 images as gray scale
plt.subplot(221)
plt.imshow(X_train[0], cmap=plt.get_cmap('gray'))
plt.subplot(222)
plt.imshow(X_train[1], cmap=plt.get_cmap('gray'))
plt.subplot(223)
plt.imshow(X_train[2], cmap=plt.get_cmap('gray'))
plt.subplot(224)
plt.imshow(X_train[3], cmap=plt.get_cmap('gray'))
# show the plot
plt.show()


# Simple CNN for the MNIST Dataset
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
# load data
(X_train, y_train), (X_test, y_test) = mnist.load_data()
# reshape to be [samples][width][height][channels]
X_train = X_train.reshape((X_train.shape[0], 28, 28, 1)).astype('float32')
X_test = X_test.reshape((X_test.shape[0], 28, 28, 1)).astype('float32')
# normalize inputs from 0-255 to 0-1
X_train = X_train / 255
X_test = X_test / 255
# one hot encode outputs
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]
# define a simple CNN model
def baseline_model():
	# create model
	model = Sequential()
	model.add(Conv2D(32, (5, 5), input_shape=(28, 28, 1), activation='relu'))
	model.add(MaxPooling2D())
	model.add(Dropout(0.2))
	model.add(Flatten())
	model.add(Dense(128, activation='relu'))
	model.add(Dense(num_classes, activation='softmax'))
	# Compile model
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model
# build the model
model = baseline_model()
# Fit the model
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=200)
# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("CNN Error: %.2f%%" % (100-scores[1]*100))


model.predict_classes(jack)[0]

from sklearn.neighbors import KNeighborsClassifier
with open("../data/trainingdata.txt") as textFile:
    features = [line.split() for line in textFile]
with open("../data/traininglabel.txt") as textFile:
    tagg = [line.split() for line in textFile]
tagi=np.array(tagg)
tag=np.ravel(tagi)
with open("testingdata.txt") as textFile:
    test = [line.split() for line in textFile]
clf = KNeighborsClassifier(n_neighbors=2,weights='distance')
clf.fit(features, tag)
preds = clf.predict(test)

# =============================================================================
# Dividing into segments part
# =============================================================================

def is_blank_image(img1):
    pixels = img1.getdata()          # get the pixels as a flattened sequence
    black_thresh = 50
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

is_blank_image(img1)
# Size of the image in pixels (size of orginal image) 
# (This is not mandatory)
algerie=Image.open("geeks.png")
cropped_img = Image.open("crop.png")
width, height = cropped_img.size 

original_sudoku=cv2.imread("crop.png") 

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
        if(is_blank_image(img1)): # fixe number to 0 for empty case
            print('0')
        else: # Recognize the number
            img1.save("two.png")
            paths=r"C:\Users\Ouistiti\Documents\CHALLENGE DATA\data_challenge\Journaux\two.png"
            mg  = cv2.imread(paths) 
            print(predict_number(mg))
            
            cv2.putText(original_sudoku,str(predict_number(mg)),(left,bottom),0,1.2,(0,0,255),2)


plt.figure(figsize = (25,14))
plt.imshow(original_sudoku)
    # Shows the image in image viewer 
    img1.show() 
    

    img1
print(model.predict_classes(jack)[0])




# =============================================================================
# #### CHANGE PNG TO NN 
# =============================================================================


from PIL import Image, ImageFilter


def imageprepare(im):
    """
    This function returns the pixel values.
    The imput is a png file location.
    """
    width = float(im.size[0])
    height = float(im.size[1])
    newImage = Image.new('L', (28, 28), (255))  # creates white canvas of 28x28 pixels

    if width > height:  # check which dimension is bigger
        # Width is bigger. Width becomes 20 pixels.
        nheight = int(round((20.0 / width * height), 0))  # resize height according to ratio width
        if (nheight == 0):  # rare case but minimum is 1 pixel
            nheight = 1
            # resize and sharpen
        img = im.resize((20, nheight), Image.ANTIALIAS)#.filter(ImageFilter.SHARPEN)
        wtop = int(round(((28 - nheight) / 2), 0))  # calculate horizontal position
        newImage.paste(img, (4, wtop))  # paste resized image on white canvas
    else:
        # Height is bigger. Heigth becomes 20 pixels.
        nwidth = int(round((20.0 / height * width), 0))  # resize width according to ratio height
        if (nwidth == 0):  # rare case but minimum is 1 pixel
            nwidth = 1
            # resize and sharpen
        img = im.resize((nwidth, 20), Image.ANTIALIAS)#.filter(ImageFilter.SHARPEN)
        wleft = int(round(((28 - nwidth) / 2), 0))  # caculate vertical pozition
        newImage.paste(img, (wleft, 4))  # paste resized image on white canvas

    newImage.save("samply.png")
    return newImage

x=imageprepare('geeks.png')#file path here
print(len(x))# mnist IMAGES are 28x28=784 pixels


# Reshaping to format which CNN expects (batch, height, width, channels)


