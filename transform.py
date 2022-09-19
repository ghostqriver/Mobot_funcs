'''
This file contain the functions which can be used in the image transformations and iteratively changed whole dataset using a transformation function

c(img): Transform a img from BGR to RGB
    img: a read in image in the BGR format
    
automatic_brightness_and_contrast(image, clip_hist_percent): The function automatically changed brightness and contrast of a given image
    img              : a readin image
    clip_hist_percent: the parameter which control how much will be clip in the hist of original image's grayscale histogram, 10 by default   

brighter_CLAHE(img,clipLimit,tileGridSize): The function apply the CLAHE on a given image
    clipLimit,tileGridSize: the main parameters which should be given when apply the CLAHE, 
        clipLimit         : float, 3.0 by
        tileGridSize      : 1*2 tuple

brightening_dataset(brightening_func,image_root,tar_folder,para = None): Do the transformation using the brightening_func on a given dataset.
    brightening_func: the brighten function's name
    image_root      : the images path which store all images of the dataset
    tar_folder      : the path which the transformed images should be stored in
    para            : the first parameter for brightening_func, because here we only define two functions brighter_CLAHE and automatic_brightness_and_contrast both with the default parameter, for further using, we can modify this parameter for more complex transformation
          
'''

import cv2
import numpy as np
import random
import os
import matplotlib.pyplot as plt
import glob

def c(img): 
    '''
    Transform a img from BGR to RGB
    img: a readin image in the BGR format
    '''
    return img[:,:,[2,1,0]]
    
def automatic_brightness_and_contrast(image, clip_hist_percent=10):
    '''
    The function automatically changed brightness and contrast of a given image
    img: a readin image
    clip_hist_percent: the parameter which control how much will be clip in the hist of original image's grayscale histogram, 10 by default
    
    '''
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Calculate grayscale histogram
    hist = cv2.calcHist([gray],[0],None,[256],[0,256])
    hist_size = len(hist)

    # Calculate cumulative distribution from the histogram
    accumulator = []
    accumulator.append(float(hist[0]))
    for index in range(1, hist_size):
        accumulator.append(accumulator[index -1] + float(hist[index]))

    # Locate points to clip
    maximum = accumulator[-1]
    clip_hist_percent *= (maximum/100.0)
    clip_hist_percent /= 2.0

    # Locate left cut
    minimum_gray = 0
    while accumulator[minimum_gray] < clip_hist_percent:
        minimum_gray += 1

    # Locate right cut
    maximum_gray = hist_size -1
    while accumulator[maximum_gray] >= (maximum - clip_hist_percent):
        maximum_gray -= 1

    # Calculate alpha and beta values
    alpha = 255 / (maximum_gray - minimum_gray)
    beta = -minimum_gray * alpha

    '''
    # Calculate new histogram with desired range and show histogram 
    new_hist = cv2.calcHist([gray],[0],None,[256],[minimum_gray,maximum_gray])
    plt.plot(hist)
    plt.plot(new_hist)
    plt.xlim([0,256])
    plt.show()
    '''

    auto_result = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return auto_result

def brighter_CLAHE(img,clipLimit=3.0,tileGridSize=(8,8)):
    '''
    The function apply the CLAHE on a given image
    clipLimit,tileGridSize: the main parameters which should be given when apply the CLAHE, 
        clipLimit: float, 3.0 by
        tileGridSize: 1*2 tuple
        
    '''
    lab= cv2.cvtColor(img, cv2.COLOR_BGR2LAB,)
    l_channel, a, b = cv2.split(lab)

    # Applying CLAHE to L-channel
    # feel free to try different values for the limit and grid size:
    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
    cl = clahe.apply(l_channel)

    # merge the CLAHE enhanced L-channel with the a and b channel
    limg = cv2.merge((cl,a,b))

    # Converting image from LAB Color model to BGR color spcae
    enhanced_img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    
    return enhanced_img

def brightening_dataset(brightening_func,image_root,tar_folder,para = None):
    '''
    Do the transformation using the brightening_func on a given dataset.
    brightening_func: the brighten function's name
    image_root      : the images path which store all images of the dataset
    tar_folder      : the path which the transformed images should be stored in
    para            : the first parameter for brightening_func, because here we only define two functions brighter_CLAHE and automatic_brightness_and_contrast both with the default parameter, for further using, we can modify this parameter for more complex transformation
   
   '''
    if not os.path.exists(tar_folder):
        os.mkdir(tar_folder) 
    
    file_names = glob.glob(image_root+'*.png')
    
    for i in file_names:

        img = cv2.imread(i)

            
        if para == None:

            img_brighter = brightening_func(img)

        else:

            img_brighter = brightening_func(img,para)

        cv2.imwrite(tar_folder+i.split('/')[2], img_brighter, (int(cv2.IMWRITE_JPEG_QUALITY),100,cv2.IMWRITE_PNG_COMPRESSION,0))

        
        if random.randint(0,len(file_names)) > len(file_names)/100*99 :

            plt.figure(figsize=(10,5))
            plt.subplot(1,2,1)
            plt.imshow(c(img))
            plt.subplot(1,2,2)
            plt.imshow(c(img_brighter))
            plt.suptitle(i)
            plt.show()
             