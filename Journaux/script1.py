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


pages = convert_from_path('20190830_PAR.pdf')
pageObj = pdfReader.getPage(16)

pageObj.extractText()


for i in range(numPages):
        page = input1.getPage(i)
        print page.mediaBox.getUpperRight_x(), page.mediaBox.getUpperRight_y()
        page.trimBox.lowerLeft = (25, 25)
        page.trimBox.upperRight = (225, 225)
        page.cropBox.lowerLeft = (50, 50)
        page.cropBox.upperRight = (200, 
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