# -*- coding: utf-8 -*-
"""
Created on Sat Nov 23 13:51:19 2019

@author: Ouistiti
"""

# =============================================================================
#
#   THe actual recursive backtracking algo that solves the sudokus
#
# =============================================================================


def SudokuSolver(grid, i=0, j=0):
        i,j = findNextCellToFill(grid, i, j)
        if i == -1: # if the sudoku is filled
                return True
        for e in range(1,10): # I check for each possible number
                if isValid(grid,i,j,e):
                        grid[i][j] = e
                        if SudokuSolver(grid, i, j):
                                return True
                        grid[i][j] = 0
        return False


def findNextCellToFill(grid, i, j):
        for x in range(i,9):
                for y in range(j,9):
                        if grid[x][y] == 0:
                                return x,y
        for x in range(0,9):
                for y in range(0,9):
                        if grid[x][y] == 0:
                                return x,y
        return -1,-1

def isValid(grid, i, j, e):
        rowOk = all([e != grid[i][x] for x in range(9)]) # make sure the number which we are 
                                                         # testing is not in the column
        if rowOk:
                columnOk = all([e != grid[x][j] for x in range(9)]) # same for the row 
                if columnOk:
                        secTopX, secTopY = 3 *(i//3), 3 *(j//3) # I get the corresponding
                                                                # sub square
                        for x in range(secTopX, secTopX+3):
                                for y in range(secTopY, secTopY+3):
                                        if grid[x][y] == e:
                                                return False # I check if the number we are testing
                                                             # isn't already in that sub square
                        return True
        return False
