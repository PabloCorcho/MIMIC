#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 20:36:30 2019

@author: pablo
"""

from scipy.ndimage import gaussian_filter
import numpy as np

from astropy.io import fits
from matplotlib import pyplot as plt 

#x = np.linspace(-100, 100, 300)
#y = np.linspace(-100, 100, 300)
#
#X, Y = np.meshgrid(x,y)
#
#z = (X*Y)*np.cos(X**2+Y**2)**2
#
#plt.figure(figsize=(10,10))
#plt.imshow(z)
#
#smooth_z = gaussian_filter(z, 3)
#
#plt.figure(figsize=(10,10))
#plt.imshow(smooth_z)
#
#print(np.sum(smooth_z), np.sum(z))

hdul = fits.open('test_data/Galex/MISDR2_04288_0826_15295/MISDR2_04288_0826-nd-int.fits')

cal_hdul = fits.open('test_data/NGC2543.V500.rscube.fits')
cal_im = cal_hdul[0].data

image = hdul[0].data
plt.figure(figsize=(10,10))
plt.imshow(image, vmax=0.008, vmin=0.0005, origin='lower')
plt.xlim(2800, 2900)
plt.ylim(1400, 1520)

smooth_image = gaussian_filter(image, 1)
smooth_cal_image = gaussian_filter(cal_im, 1)

plt.figure(figsize=(10,10))
plt.imshow(smooth_image, vmax=0.008, vmin=0.0005, origin='lower')
plt.xlim(2800, 2900)
plt.ylim(1400, 1520)

print(np.sum(smooth_image[1300:1520,2800:2900]), np.sum(image[1300:1520,2800:2900]))
