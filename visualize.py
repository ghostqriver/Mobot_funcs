import skimage.io as io
import matplotlib.pyplot as plt
import cv2
from pycocotools.coco import COCO
import os
import random

def coco_json_show(json_file,image_path,image_name=None):
    
    '''
    Given the path of json_file and images' path, random show 5 images with its annotations in the coco json file.
    If given a image_name, then only show that image.
    '''

    coco = COCO(json_file)

    if image_name == None:
        image_ids = random.sample(coco.getImgIds(),5)
    else:
        image_ids = None
        for i in coco.getImgIds():
            if coco.loadImgs(i)[0]['file_name'] == image_name:
                image_ids = [i]
        if image_ids == None:
            raise ValueError('There is no '+image_name+' in dataset')
            
    for img in image_ids:

        imginfo = coco.loadImgs(img)[0]
        im = cv2.imread(os.path.join(image_path,coco.loadImgs(img)[0]['file_name']))
        annIds = coco.getAnnIds(img,catIds=[0,1])
        annsInfo = coco.loadAnns(annIds)
        img_name = coco.loadImgs(img)[0]['file_name']
    

        # Show the image
        plt.figure(figsize=(15,15))
        plt.imshow(im[:,:,[2,1,0]])
        plt.axis('off')
        coco.showAnns(annsInfo, True)

        # Show the text for each bbox
        coordinates=[]
        for j in range(len(annsInfo)):
            left = annsInfo[j]['bbox'][0]
            top = annsInfo[j]['bbox'][1]
            plt.text(left,top+15,coco.loadCats(annsInfo[j]['category_id'])[0]['supercategory'],fontsize=10)


        plt.title(img_name)
        plt.show()
