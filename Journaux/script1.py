# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 13:58:49 2019

@author: Ouistiti
"""

# =============================================================================
# Packages
# =============================================================================
import PyPDF2

# =============================================================================
# Extraction
# =============================================================================
pdfFileObj = open('20190830_PAR.pdf', 'rb')
pdfReader = PyPDF2.PdfFileReader(pdfFileObj)

# Number of Pages of that Document
pdfReader.numPages


pageObj = pdfReader.getPage(0)

pageObj.extractText()



# =============================================================================
# Sudoku Solving process
# =============================================================================
