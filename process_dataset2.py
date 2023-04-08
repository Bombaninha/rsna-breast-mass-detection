import pydicom as dicom
import cv2   
import numpy as np
import pandas as pd
import glob
import os

#DCM_PATH = 'INbreast Release 1.0/AllDICOMs/'
DCM_PATH = 'data/AllDICOMs/'

def crop(img):
    # Otsu's thresholding after Gaussian filtering
    blur = cv2.GaussianBlur(img, (5,5), 0)
    _, breast_mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    cnts, _ = cv2.findContours(breast_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt = max(cnts, key = cv2.contourArea)
    x, y, w, h = cv2.boundingRect(cnt) # get bounding rect that covers the breast area

    return img[y:y+h, x:x+w], breast_mask[y:y+h, x:x+w], (x, y, w, h)

def truncation_normalization(img, mask):
    # Get min and max values for normalization
    Pmin = np.percentile(img[mask!=0], 5)
    Pmax = np.percentile(img[mask!=0], 99)
    # Truncate image
    truncated = np.clip(img, Pmin, Pmax) 
    normalized = (truncated - Pmin) / (Pmax - Pmin)
    normalized[mask == 0] = 0
    return normalized

def contrast_enhancement(img, clip):
    # Contrast Limited Adaptative Histogram Equalization
    clahe = cv2.createCLAHE(clipLimit=clip)
    cl = clahe.apply(np.array(img * 255, dtype=np.uint8))
    return cl

def process_images(patient_id, suffix_path, orientation='R'):
    image_path = glob.glob(os.path.join(DCM_PATH, str(patient_id) + suffix_path))[0]
    ds = dicom.dcmread(image_path)
    pixel_array_numpy = ds.pixel_array
    breast, mask, dims = crop(pixel_array_numpy)
    normalized = truncation_normalization(breast, mask)

    cl1 = contrast_enhancement(normalized, 1.0)
    cl2 = contrast_enhancement(normalized, 2.0)
    #processed = np.array(normalized*255, dtype=np.uint8)
    processed = cv2.merge((np.array(normalized*255, dtype=np.uint8),cl1,cl2))
    return breast, processed, mask, dims, pixel_array_numpy

if __name__ == "__main__":

    if not os.path.exists('results'):
        os.mkdir('results')

    max_w, max_h, max_w_img_id, max_h_img_id = 0, 0, 0, 0

    df = pd.read_csv('data/INbreast.csv', delimiter=';')

    # Scan the file directory for all image files and get the patient_id and suffix_path from them:
    imgFileName_list = os.scandir(DCM_PATH)
    orientation = []
    for imgFile in imgFileName_list:
        if(imgFile.path.find('.dcm') != -1):
            imgFileName = imgFile.path.split(DCM_PATH)[1]
            paths = imgFileName.split('_')
            patient_id = paths[0]

            img_class = df.loc[df['File Name'] == int(patient_id)]['Bi-Rads']
            img_class = img_class.to_string(index = False)
            
            # if img_class == '1':
            #     img_class = '0'
            # elif img_class == '2':
            #     img_class = '1'
            # else:
            #     img_class = '2'
            #print(img_class)

            orientation.append(patient_id + '_' + paths[3] + '_' + img_class)
            suffix = '_' + '_'.join(paths[1:])

            original, processed, mask, dims, img = process_images(patient_id, suffix, orientation)
            max_w = max(dims[2], max_w)
            max_h = max(dims[3], max_h)
            if dims[2] == max_w : 
                max_w_img_id = patient_id
            if dims[3] == max_h : 
                max_h_img_id = patient_id

            if not os.path.exists('results/' + img_class):
            # if the directory is not present then create it.
                os.mkdir('results/' + img_class)

            processed = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
            processed = cv2.resize(processed, (512,512), interpolation=cv2.INTER_CUBIC)
            
            cv2.imwrite('results/' + img_class + '/' + patient_id + '_processed.png', processed.astype(np.uint8))
            
            if img_class == '1' or img_class == '5':
                flippedH = cv2.flip(processed, 1)
                flippedV = cv2.flip(processed, 0)
                
                cv2.imwrite('results/' + img_class + '/' + patient_id + '_processedH.png', flippedH.astype(np.uint8))
                cv2.imwrite('results/' + img_class + '/' + patient_id + '_processedV.png', flippedV.astype(np.uint8))
                
            elif img_class != '2':
                flippedH = cv2.flip(processed, 1)
                flippedV = cv2.flip(processed, 0)
                flippedD = cv2.flip(processed, -1)
                
                cv2.imwrite('results/' + img_class + '/' + patient_id + '_processedH.png', flippedH.astype(np.uint8))
                cv2.imwrite('results/' + img_class + '/' + patient_id + '_processedV.png', flippedV.astype(np.uint8))
                cv2.imwrite('results/' + img_class + '/' + patient_id + '_processedD.png', flippedD.astype(np.uint8))

            # print images used in our presentation:
            # if patient_id == '20587080':
            #     cv2.imwrite('data/' + patient_id + '_original.png', img.astype(np.uint8))
            #     cv2.imwrite('data/' + patient_id + '_original_cropped.png', original.astype(np.uint8))
            # if patient_id == '22614097':
            #     cv2.imwrite('data/' + patient_id + '_original.png', original.astype(np.uint8))
            # if patient_id == '53586869':
            #     cv2.imwrite('data/' + patient_id + '_original.png', original.astype(np.uint8))

    # get ids of the images used in our presentation
    # print(max_w_img_id)
    # print(max_h_img_id)

    cv2.destroyAllWindows()