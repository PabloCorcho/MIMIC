#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 15 11:08:27 2019

@author: pablo
"""

import numpy as np

from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs import utils
from scipy.interpolate import interp1d, RegularGridInterpolator

class image(object):
    
    def __init__(self, hdul, IFU=False):
        self.IFU = IFU
        if self.IFU:
            self.image = hdul[0].data[1400,:,:]
        else:            
            self.image = hdul[0].data
            
        self.wcs = WCS(hdul[0].header)
                
    def crop(self, row_bounds, col_bounds):
        return self.image[col_bounds[0]:col_bounds[1], row_bounds[0]:row_bounds[1]]    
    
    def get_pix_scale(self):
        self.x_scale, self.y_scale = utils.proj_plane_pixel_scales(self.wcs)
        return self.x_scale, self.y_scale
    
    def get_pix_area(self):
        return utils.proj_plane_pixel_area(self.wcs)
    
    def get_im_area(self):        
        return self.image.shape[0]*self.image.shape[1]
    
    def get_coords(self):
        
        if self.IFU:                        
            ar_dec_wl = np.array([[0,0, 1400],
                              [self.image.shape[1]-1,self.image.shape[0]-1, 1400]])
    
            self.world = self.wcs.wcs_pix2world(ar_dec_wl, 0).T            
        
        else:
            ar_dec = np.array([[0,0],
                              [self.image.shape[1]-1,self.image.shape[0]-1]])
            self.world = self.wcs.wcs_pix2world(ar_dec, 0).T            
        self.world = np.array([ np.linspace(self.world[0,0], self.world[0,-1], self.image.shape[1]),
                               np.linspace(self.world[1,0], self.world[1,-1], self.image.shape[0]),
                               ])    
        return self.world # (ra, dec)
        
    def get_FoV(self):
        return self.get_pix_area()*self.get_im_area()
        
        
        
        
class multi_image_match(object):

    def __init__(self, hdul1, hdul2, IFU_2=False):
        self.image1 = image(hdul1)
        if IFU_2:
            self.image2 = image(hdul2, IFU=True)
        else:
            self.image2 = image(hdul2)
            
        self.coords1 = self.image1.get_coords()
        self.coords2 = self.image2.get_coords()
        
        self.check_sizes()
        self.reduce_image()
        
    def check_sizes(self):
        if self.image1.get_FoV()>self.image2.get_FoV():
            self.big_1 = True
            if self.image1.get_pix_area()>self.image2.get_pix_area():
                self.pix_1_smaller = False
            else:                
                self.pix_1_smaller = True
        else:            
            self.big_1 = False
            if self.image1.get_pix_area()>self.image2.get_pix_area():
                self.pix_1_smaller = False
            else:                
                self.pix_1_smaller = True
        
        
    def reduce_image(self):
        if self.big_1:            
            bounds = self.image1.wcs.all_world2pix(
                    np.array([[self.coords2[0][0],self.coords2[1][0]],
                              [self.coords2[0][-1],self.coords2[1][-1]]]), 
                                                   0).T    
            bounds = np.array(bounds, dtype=int)                                     
            self.image1.image = self.image1.crop(bounds[0], bounds[1])
            print(bounds)                                   
        else:
            bounds = self.image2.wcs.all_world2pix(
                    np.array([[self.coords1[0][0],self.coords1[1][0]],
                              [self.coords1[0][-1],self.coords1[1][-1]]]), 
                                                   0).T    
            bounds = np.array(bounds, dtype=int)              
            self.image2.image = self.image2.crop(bounds[0], bounds[1])
            
        
        
        
if __name__ == '__main__':
    hdul= fits.open('../data/GALEX/IMAGES/Galex/AIS_83_50083_0001_sv83/AIS_83_sg83-nd-int.fits')        
    cal_hdul= fits.open('../data/CALIFA/UGC03944.V500.rscube.fits')        
    
    im1 = image(hdul)
    im2 = image(cal_hdul, IFU=True)
    
    multi_match = multi_image_match(hdul, cal_hdul, IFU_2=True)