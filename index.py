import pydicom as dicom
import matplotlib.pylab as plt
import cv2   
import numpy as np
from skimage import exposure
import glob
import os
import plistlib
from skimage.draw import polygon

DCM_PATH = 'INbreast Release 1.0/AllDICOMs/'
XML_PATH = 'INbreast Release 1.0/AllXML/'

def load_inbreast_mask(mask_path, imshape=(4084, 3328)):
    """
    This function loads a osirix xml region as a binary numpy array for INBREAST
    dataset
    @mask_path : Path to the xml file
    @imshape : The shape of the image as an array e.g. [4084, 3328]
    return: numpy array where each mass has a different number id.
    """

    def load_point(point_string):
        x, y = tuple([float(num) for num in point_string.strip('()').split(',')])
        return y, x
    i =  0
    mask = np.zeros(imshape)
    with open(mask_path, 'rb') as mask_file:
        plist_dict = plistlib.load(mask_file, fmt=plistlib.FMT_XML)['Images'][0]
        numRois = plist_dict['NumberOfROIs']
        rois = plist_dict['ROIs']
        assert len(rois) == numRois
        for roi in rois:
            numPoints = roi['NumberOfPoints']
            if roi['Name'] == 'Mass':
                i+=1
                points = roi['Point_px']
                assert numPoints == len(points)
                points = [load_point(point) for point in points]
                if len(points) <= 2:
                    for point in points:
                        mask[int(point[0]), int(point[1])] = i
                else:
                    x, y = zip(*points)
                    x, y = np.array(x), np.array(y)
                    poly_x, poly_y = polygon(x, y, shape=imshape)
                    mask[poly_x, poly_y] = i
    return mask

def crop(img, mask):
    """
    Crop breast ROI from image.
    @img : numpy array image
    @mask : numpy array mask of the lesions
    return: numpy array of the ROI extracted for the image, 
            numpy array of the ROI extracted for the breast mask,
            numpy array of the ROI extracted for the masses mask
    """
    # Otsu's thresholding after Gaussian filtering
    blur = cv2.GaussianBlur(img, (5,5), 0)
    _, breast_mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    cnts, _ = cv2.findContours(breast_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt = max(cnts, key = cv2.contourArea)
    x, y, w, h = cv2.boundingRect(cnt)

    return img[y:y+h, x:x+w], breast_mask[y:y+h, x:x+w], mask[y:y+h, x:x+w]

def truncation_normalization(img, mask):
    """
    Pixel clipped and normalized in breast ROI
    """
    Pmin = np.percentile(img[mask!=0], 5)
    Pmax = np.percentile(img[mask!=0], 99)
    truncated = np.clip(img,Pmin, Pmax)  
    normalized = (truncated - Pmin) / (Pmax - Pmin)
    normalized[mask==0]=0
    return normalized

def clahe(img, clip):
    #contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=clip)
    cl = clahe.apply(np.array(img*255, dtype=np.uint8))
    return cl

def synthetized_images(patient_id, suffix_path):
    """
    Create a 3-channel image composed of the truncated and normalized image,
    the contrast enhanced image with clip limit 1, 
    and the contrast enhanced image with clip limit 2 
    @patient_id : patient id to recover image and mask in the dataset
    return: numpy array of the breast region, numpy array of the synthetized images, numpy array of the masses mask
    """
    image_path = glob.glob(os.path.join(DCM_PATH, str(patient_id) + suffix_path))[0]
    mass_mask = load_inbreast_mask(os.path.join(XML_PATH, str(patient_id) + '.xml'))
    ds = dicom.dcmread(image_path)
    pixel_array_numpy = ds.pixel_array
    breast, mask, mass_mask = crop(pixel_array_numpy, mass_mask)
    normalized = truncation_normalization(breast, mask)

    cl1 = clahe(normalized, 1.0)
    cl2 = clahe(normalized, 2.0)
    #synthetized = normalized
    synthetized = cv2.merge((np.array(normalized*255, dtype=np.uint8),cl1,cl2))
    return breast, synthetized, mass_mask

if __name__ == "__main__":
    
    #patient_id = '20586934' #'20586908'
    #suffix = '_6c613a14b80a8591_MG_L_CC_ANON.dcm' #'_6c613a14b80a8591_MG_R_CC_ANON.dcm'

    # Scan the file directory for all image files and get the patient_id and suffix_path from them:
    imgFileName_list = os.scandir(DCM_PATH)

    for imgFile in imgFileName_list:
        if(imgFile.path.find('.dcm') != -1):
            imgFileName = imgFile.path.split(DCM_PATH)[1]
            paths = imgFileName.split('_')
            patient_id = paths[0]
            suffix = '_' + '_'.join(paths[1:])
            print('Processing patient file: ' + patient_id + suffix)

            original, synthetized, mass_mask = synthetized_images(patient_id, suffix)

            synthetized = cv2.cvtColor(synthetized, cv2.COLOR_BGR2RGB)
            cv2.imwrite(patient_id + '_synthetized.jpeg', synthetized.astype(np.uint8))
            cv2.imwrite(patient_id + '_mask.jpeg', (mass_mask*255).astype(np.uint8))
    
    # synthetized = cv2.cvtColor(synthetized, cv2.COLOR_BGR2RGB)

    # cv2.namedWindow('Display', cv2.WINDOW_NORMAL)
    # cv2.namedWindow('Output', cv2.WINDOW_NORMAL)
    # cv2.namedWindow('Mask', cv2.WINDOW_NORMAL)
    # cv2.imshow('Display', original.astype(np.uint8))
    # cv2.imshow('Output', synthetized.astype(np.uint8))
    # cv2.imshow('Mask', (mass_mask*255).astype(np.uint8))
    # cv2.waitKey(0)

    # cv2.imwrite(patient_id + '_mask.jpeg', (mass_mask*255).astype(np.uint8))
    # cv2.imwrite(patient_id + '_synthetized.jpeg', synthetized.astype(np.uint8))
    
    cv2.destroyAllWindows()