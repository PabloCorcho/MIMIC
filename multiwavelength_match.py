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

from scipy.ndimage import gaussian_filter

from scipy.interpolate import griddata
from matplotlib import pyplot as plt



class image(object):
    
    def __init__(self, hdul, IFU=False):
        self.IFU = IFU
        if self.IFU:
            self.image = hdul.data[1400,:,:]
        else:            
            self.image = hdul.data
            
        self.wcs = WCS(hdul.header)
        
        ra, dec = self.get_coords()                
        
            
    def crop(self, row_bounds, col_bounds):
        return self.image[col_bounds[0]:col_bounds[1], row_bounds[0]:row_bounds[1]]    
    
    def get_pix_scale(self):
        self.x_scale, self.y_scale = utils.proj_plane_pixel_scales(self.wcs)
        return self.x_scale, self.y_scale
    
    def get_pix_area(self):
        return utils.proj_plane_pixel_area(self.wcs)
    
    def get_im_area(self):        
        return self.image.shape[0]*self.image.shape[1]
    
    def get_centre(self):
        ar_cent, dec_cent = self.wcs.wcs_pix2world(self.image.shape[0]/2,
                                                    self.image.shape[1]/2, 0)
        return ar_cent, dec_cent
    
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
        
    def gauss_smooth(self, sigma):
        return gaussian_filter(self.image)

    def plotframe(self, color='k'):
        from matplotlib.patches import Rectangle
        coords = self.get_coords()
        frame = Rectangle(xy=(coords[0][0], coords[1][0]),
                  width=coords[0][-1]-coords[0][0],
                  height=coords[1][-1]-coords[1][0],
                  fill=False,
                  color=color)     
        
        return frame
        
class multi_image_match(object):

    def __init__(self, hdul1, hdul2, IFU_2=False):
        print('--> Initializing IMAGE MATCHING')
        self.image1 = image(hdul1)
        
        if IFU_2:
            print('Image 2 --> IFU mode')
            self.image2 = image(hdul2, IFU=True)
        else:
            self.image2 = image(hdul2)
        
        print('Image 1 with size {}'.format(self.image1.image.shape))
        print('Image 2 with size {}'.format(self.image2.image.shape))
        #---------------------------------------------------------------------    
        self.coords1 = self.image1.get_coords()                
        self.coords2 = self.image2.get_coords()
        #---------------------------------------------------------------------
        self.check_sizes()
        #---------------------------------------------------------------------
        self.reduce_image()        
        #---------------------------------------------------------------------        
        plt.figure(figsize=(7,7))
        plt.contour(np.log10(self.image1.image), origin='lower', colors='k',
                    alpha=1, levels=20,
                   aspect='auto', vmin=np.percentile(np.log10(self.image1.image),0.5),
                   extent=(self.new_coords1[0][0], self.new_coords1[1][0], 
                           self.new_coords1[0][1], self.new_coords1[1][1]))        
        
        plt.imshow(np.log10(self.image2.image), origin='lower', cmap='gist_ncar',
                    aspect='auto', alpha=0.6,
                   extent=(self.new_coords2[0][0], self.new_coords2[1][0], 
                           self.new_coords2[0][1], self.new_coords2[1][1]))
#        plt.xlim(203, 206)
#        plt.ylim(-32, -31)
        #---------------------------------------------------------------------
#        self.image2.image = self.bin_image(self.image2.image)
        #---------------------------------------------------------------------
#        plt.figure(figsize=(7,7))
#        plt.contour(self.image1.image, origin='lower', colors='k', levels=10,
#                   extent=(self.new_coords[0][0], self.new_coords[1][0], 
#                           self.new_coords[0][1], self.new_coords[1][-1]))
#        plt.imshow(self.image2.image, origin='lower', cmap='gist_ncar', alpha=0.4,
#                   extent=(self.new_coords[0][0], self.new_coords[1][0], 
#                           self.new_coords[0][1], self.new_coords[1][-1]))
        
    def check_sizes(self, show=True):
        """
        This method selects the common area between both images. 
        The reference image will be 1.
        By default, It creates a figure showing the common frame 
        in RA (x) and DEC (y).        
        """
        # find pixel with common RA       
        comRApix = np.where((self.coords1[0]<=np.max(self.coords2[0]))&
                         (self.coords1[0]>=np.min(self.coords2[0]))
                         )[0]
        
        # find pixels with common DEC        
        comDECpix = np.where((self.coords1[1]<=np.max(self.coords2[1]))&
                         (self.coords1[1]>=np.min(self.coords2[1]))
                         )[0]
                            
        print('Image 1 common pixels size: ({:}, {:})'.format(comRApix.size,
                                                              comDECpix.size))
        
        # Corner coordinates
        
        minRA = np.min(self.coords1[0][comRApix])
        maxRA = np.max(self.coords1[0][comRApix])
        minDEC = np.min(self.coords1[1][comDECpix])
        maxDEC = np.max(self.coords1[1][comDECpix])
        if show:
            comFrame = plt.Rectangle(xy=(minRA, minDEC), width=maxRA-minRA,
                                     height=maxDEC-minDEC, hatch='\\', fill=True,
                                     color='g', alpha=.3)
            fig = plt.figure(figsize=(10,10))
            ax = fig.add_subplot(111)
            ax.add_patch(comFrame)
            ax.add_patch(self.image1.plotframe(color='r'))
            ax.add_patch(self.image2.plotframe(color='b'))
            ax.annotate('Image 1', xy=(minRA,maxDEC), color='r')
            ax.plot()        
            plt.show()
    
    
        self.boundRA = np.array([minRA, maxRA])
        self.boundDEC = np.array([minDEC, maxDEC])        
        self.bounds1 = np.array([[comRApix[0], comRApix[-1]], 
                                 [comDECpix[0], comDECpix[-1]]])
        
    def reduce_image(self):
            
        # Image 1                
#        bounds = self.image1.wcs.all_world2pix(
#                self.boundRA, self.boundDEC, 0)            
        bounds =self.bounds1
        bounds = np.sort(np.array(bounds, dtype=int))              
                
        self.image1.image = self.image1.crop(bounds[0], bounds[1])
        self.new_coords1 = self.image1.wcs.all_pix2world(bounds.T, 0)
        
        # Image 2        
        bounds = self.image2.wcs.all_world2pix(
                    self.boundRA, self.boundDEC, 0)            
        
        print(bounds)
        bounds = np.sort(np.array(bounds, dtype=int))             
        print(bounds)
        self.image2.image = self.image2.crop(bounds[0], bounds[1])        
        self.new_coords2 = self.image2.wcs.all_pix2world(bounds.T, 0)
        
        print('Image 2 cropped with errors at edges:\n  RA_err={}, DEC_err={} [arcsec]'.format(3600*
              (self.new_coords1[0,0]-self.new_coords2[0,0]),
              3600*(self.new_coords1[1,1]-self.new_coords2[1,1]))
              )
        
        print(self.new_coords2)
        print(self.new_coords1)
            
    def bin_image(self, image):          
        if not self.pix_1_smaller:
            AR, DEC = np.meshgrid(self.coords2[0], self.coords2[1])
            
            new_ar = np.linspace(self.new_coords[0][0], self.new_coords[1][0], 
                                     self.image1.image.shape[1])
            print(new_ar.shape)
            new_dec = np.linspace(self.new_coords[0][-1], self.new_coords[1][-1],
                                     self.image1.image.shape[0])
            
            new_AR, new_DEC = np.meshgrid(new_ar, new_dec)
            
            new_image = griddata((AR.flatten(),DEC.flatten()), 
                                 image.flatten(), 
                                 (new_AR,new_DEC))
            return new_image
       
        
        
if __name__ == '__main__':
#    hdul= fits.open('/home/pablo/obs_data/test_data/Galex/MISDR2_04288_0826_15295/MISDR2_04288_0826-nd-int.fits')        
#    cal_hdul= fits.open('/home/pablo/obs_data/test_data/NGC2543.V500.rscube.fits')        
#    
#    im1 = image(hdul[0])
#    im2 = image(cal_hdul[0], IFU=True)
#    
#    fig = plt.figure(figsize=(10,10))
#    ax = fig.add_subplot(111)
#    ax.add_patch(im1.plotframe(color='r'))
#    ax.add_patch(im2.plotframe(color='b'))
#    ax.plot()
#    plt.show()
#    multi_match = multi_image_match(hdul[0], cal_hdul[0], IFU_2=True)
    
#    hdul2= fits.open('/home/pablo/obs_data/HiKids/NGC5253/eHst1968221/HST/hst_13364_59_wfc3_uvis_f275w/hst_13364_59_wfc3_uvis_f275w_drz.fits')        
    hdul2= fits.open('/home/pablo/obs_data/HiKids/NGC5253/eHst1968191/HST/hst_12206_03_wfc3_ir_total/hst_12206_03_wfc3_ir_total_drz.fits')        
    hdul1= fits.open('/home/pablo/obs_data/HiKids/NGC5253/Galex/GI4_095049_NGC5253_24189/GI4_095049_NGC5253-nd-int.fits')        
#    hdul1= fits.open('/home/pablo/obs_data/HiKids/NGC5253/VISTA/938_968_48_3461_1.fits')        
    
    im1 = image(hdul1[0])
    im2 = image(hdul2[1])
    
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111)
    ax.add_patch(im1.plotframe(color='r'))
    ax.add_patch(im2.plotframe(color='b'))
    ax.plot()
    plt.show()
    
    multi_match = multi_image_match(hdul1[0], hdul2[1])
    
    