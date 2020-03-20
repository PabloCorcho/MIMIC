#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 16:45:44 2020

@author: pablo
"""

import numpy as np
from numba import jit
from scipy.interpolate import interp2d


#@jit(nopython=True)
#def resize_matrix(matrix, new_dim):
#    
#    new_matrix = np.zeros(new_dim, dtype=np.float64)
#    
#    mask = np.ones(matrix.shape, dtype=np.float64)
#    
#    prior_shape = matrix.shape
#    
#    int_0 = int(prior_shape[0]/new_dim[0])
#    dec_0 = prior_shape[0]/new_dim[0]-int_0
#    
#    int_1 = int(prior_shape[1]/new_dim[1])
#    dec_1 = prior_shape[1]/new_dim[1]-int_1
#    
#    if dec_0>0:
#        mask[0+int_0:-1:int_0+1, :] = dec_0*mask[0+int_0:-1:int_0+1, :]
#    if dec_1>0:
#        mask[:, 0+int_1:-1:int_1+1] = dec_1*mask[:, 0+int_1:-1:int_1+1]
#        
#    
#    matrix = matrix*mask        
#    for ith in range(new_dim[0]):        
#        for jth in range(new_dim[1]):   
##            print('entry: ', ith, jth)                           
#            entry = np.nansum(matrix[ith*int_0:(ith+1)*int_0, 
#                                     jth*int_1:(jth+1)*int_1]
#                             )
#            if dec_0>0: 
#                lower_row = np.nansum(matrix[ith*int_0+1, jth*int_1:(jth+1)*int_1]
#                            )     
#                entry += lower_row
#            if dec_1>0:
#                right_col = np.nansum(matrix[ith*int_0:(ith+1)*int_0, jth*int_1+1])                        
#                entry += right_col
#                if dec_0>0:
#                    low_right_entry = matrix[ith*int_0+1, jth*int_1+1]
#                    entry += low_right_entry
#
#            new_matrix[ith, jth] = entry
#            
#    return new_matrix            

def resize_matrix(matrix, new_dim):
    cummatrix = np.nancumsum(np.nancumsum(matrix, axis=0), axis=1)
    print('MS', cummatrix.shape)
    rows = np.linspace(0, 1, matrix.shape[0])
    cols = np.linspace(0, 1, matrix.shape[1])
    print(rows.shape, cols.shape)
    f = interp2d(cols, rows , cummatrix)
    
    newrows = np.linspace(0, 1, new_dim[0])
    newcols = np.linspace(0, 1, new_dim[1])
    new_cummatrix = f(newcols, newrows)
    
    new_matrix = new_cummatrix      
    new_matrix[:, 1:]  =   new_matrix[:, 1:] - new_matrix[:, :-1]  
    new_matrix[1:, :]  =   new_matrix[1:, :] - new_matrix[:-1, :]  
        
    return new_matrix
if __name__ == '__main__':
    from matplotlib import pyplot as plt
    x = np.linspace(0,10, 80)
    y = np.linspace(0, 10, 72)
    xx, yy = np.meshgrid(x,y)
    z = np.sin(xx+yy)    
    plt.figure()
    plt.imshow(z)
    plt.colorbar()
#    C = np.ones((10,10))    
    
#    C = np.random.rand(451,430)
    z_prime = resize_matrix(z, (100,100))
    plt.figure()
    plt.imshow(z_prime)
    plt.colorbar()
    
#    x = np.linspace(0,10, 30)
#    y = np.linspace(0, 10, 30)
#    xx, yy = np.meshgrid(x,y)
#    z = np.sin(xx**2+yy**2)    
#    plt.figure()
#    plt.imshow(z)
#    plt.colorbar()
#    print('time:', end - start)
    print(np.nansum(z), np.nansum(z_prime), np.nansum(z_prime)/np.nansum(z))
