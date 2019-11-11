# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 13:58:49 2019

@author: Ouistiti
"""

# =============================================================================
# Packages
# =============================================================================
import PyPDF2
import tabula
from pdf2image import convert_from_path
# =============================================================================
# Extraction
# =============================================================================
pdfFileObj = open('20190830_PAR.pdf', 'rb')
pdfReader = PyPDF2.PdfFileReader(pdfFileObj)

# Number of Pages of that Document
pdfReader.numPages


pdfReader.getOutlines()

getOutlines(pdfReader)



pages = convert_from_path('20190830_PAR.pdf')
pageObj = pdfReader.getPage(16)

pageObj.extractText()


pageObj.getContents()
pageObj.mediaBox
pageObj.artBox
pageObj.compressContentStreams()
pageObj.trimBox((0,0),(0,0),(40,40),(40,40))

for i in range(numPages):
        page = input1.getPage(i)
        print page.mediaBox.getUpperRight_x(), page.mediaBox.getUpperRight_y()
        page.trimBox.lowerLeft = (25, 25)
        page.trimBox.upperRight = (225, 225)
        page.cropBox.lowerLeft = (50, 50)
        page.cropBox.upperRight = (200, 225) 
# =============================================================================
# Sudoku Solving process
# =============================================================================

# The preprocessing part starts with converting the given image to grayscale in
# order to simplify processing.

# readinf the PDF file that contain Table Data
# you can find find the pdf file with complete code in below
# read_pdf will save the pdf table into Pandas Dataframe
df = tabula.read_pdf("offense.pdf")
# in order to print first 5 lines of Table
df.head()

df = tabula.read_pdf('20190830_PAR.pdf',multiple_tables=True)
df2 = tabula.read_pdf("20190830_PAR.pdf", pages=16)




import fitz
pdffile = "20190830_PAR.pdf"
doc = fitz.open(pdffile)
page = doc.loadPage(16) #number of page

toc = doc.getToC()

# get all links on a page
links = page.getLinks()

annots = page.annots()

pix = page.getPixmap()

pix.size
pix.height
pix.width

pix.writeImage("bob.png")

output = "outfile.png"
pix.writePNG(output)


text = page.getText("xml")

Html_file= open("filename","w")
Html_file.write(text)
Html_file.close()


from PIL import Image
img = Image.open("bob.png")
area = (242, 606, 385, 749)
cropped_img = img.crop(area)
cropped_img.show()

output = "bob.png"
cropped_img.writePNG(output)

# creating a image object (main image)  
im1 = Image.open(r"C:\Users\Ouistiti\Documents\CHALLENGE DATA\data_challenge\Journaux\alice.png")  
  
# save a image using extension 
cropped_img.save("geeks.png") 

# =============================================================================
# Dividing into segments part
# =============================================================================

# 81 cases a déterminer 
sudoku = []
for i in range(10):
    sudoku_temp = []
    for j in range(10):
        border=2
        width_len=round(width/9)
        height_len=round(width/9)
        print(str(i)+","+str(j))
        # Setting the points for cropped image 
        left = width_len*i+border
        top = height_len*j+border
        right = width_len*(i+1)-border
        bottom = height_len*(j+1)-border
          
        # Cropped image of above dimension 
        # (It will not change orginal image) 
        im1 = cropped_img.crop((left, top, right, bottom))
        
        # Get the corresponding number and save it
        


    # Size of the image in pixels (size of orginal image) 
    # (This is not mandatory) 
    width, height = cropped_img.size 
    
    
    
    # Setting the points for cropped image 
    left = width_len*i+border
    top = height_len*j+border
    right = width_len*(i+1)-border
    bottom = height_len*(j+1)-border
      
    # Cropped image of above dimension 
    # (It will not change orginal image) 
    im1 = cropped_img.crop((left, top, right, bottom)) 
      
    # Shows the image in image viewer 
    im1.show() 




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


model.fit()



# =============================================================================
# #### CHANGE PNG TO NN 
# =============================================================================


from PIL import Image, ImageFilter


def imageprepare(argv):
    """
    This function returns the pixel values.
    The imput is a png file location.
    """
    im = Image.open(argv).convert('L')
    width = float(im.size[0])
    height = float(im.size[1])
    newImage = Image.new('L', (28, 28), (255))  # creates white canvas of 28x28 pixels

    if width > height:  # check which dimension is bigger
        # Width is bigger. Width becomes 20 pixels.
        nheight = int(round((20.0 / width * height), 0))  # resize height according to ratio width
        if (nheight == 0):  # rare case but minimum is 1 pixel
            nheight = 1
            # resize and sharpen
        img = im.resize((20, nheight), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
        wtop = int(round(((28 - nheight) / 2), 0))  # calculate horizontal position
        newImage.paste(img, (4, wtop))  # paste resized image on white canvas
    else:
        # Height is bigger. Heigth becomes 20 pixels.
        nwidth = int(round((20.0 / height * width), 0))  # resize width according to ratio height
        if (nwidth == 0):  # rare case but minimum is 1 pixel
            nwidth = 1
            # resize and sharpen
        img = im.resize((nwidth, 20), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
        wleft = int(round(((28 - nwidth) / 2), 0))  # caculate vertical pozition
        newImage.paste(img, (wleft, 4))  # paste resized image on white canvas

    # newImage.save("sample.png

    tv = list(newImage.getdata())  # get pixel values

    # normalize pixels to 0 and 1. 0 is pure white, 1 is pure black.
    tva = [(255 - x) * 1.0 / 255.0 for x in tv]
    print(tva)
    return tva

x=imageprepare('geeks.png')#file path here
print(len(x))# mnist IMAGES are 28x28=784 pixels