# -*- coding: utf-8 -*-
"""
Created on Sat Nov 23 13:51:19 2019

@author: Ouistiti
"""


def SudokuSolver(grid, i=0, j=0):
        i,j = findNextCellToFill(grid, i, j)
        if i == -1: # si le sudoku est rempli !
                return True
        for e in range(1,10): # je vérifie pour chaque chiffre possible
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
        rowOk = all([e != grid[i][x] for x in range(9)]) # le chiffre qu'on teste n'est 
                                                         # bien pas présent dans la colonne
        if rowOk:
                columnOk = all([e != grid[x][j] for x in range(9)]) # pareil mais pour ligne
                if columnOk:
                        secTopX, secTopY = 3 *(i//3), 3 *(j//3) #j'étable le carré où 
                                                                # il correspond
                        for x in range(secTopX, secTopX+3):
                                for y in range(secTopY, secTopY+3):
                                        if grid[x][y] == e:
                                                return False # je vérfie si notre chiffre
                                                             # n'est pas déjà dans le carré
                        return True
        return False
