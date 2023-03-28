import pydicom as dicom
import matplotlib.pylab as plt
import cv2   
import numpy as np
from skimage import exposure

image_path = './dataset/AllDICOMs/20586908_6c613a14b80a8591_MG_R_CC_ANON.dcm'

ds = dicom.dcmread(image_path)
dcm_sample = ds.pixel_array
dcm_sample = exposure.equalize_adapthist(dcm_sample)

cv2.namedWindow('Display', cv2.WINDOW_NORMAL)
cv2.imshow('Display', dcm_sample)
cv2.waitKey(0)
  
cv2.destroyAllWindows()