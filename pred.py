'''
This file contains some functions for predict use the model.

ARGS: Arguments for setting up the model:
    MODEL_DIR        : the folder under the current path which store the best model, I will always upload the best model in this folder
    MODEL_CKPT       : the model's name which was named by its checkpoints, in the best_model folder, the model is eg. model_0012499.pth
    BACKBONE         : We are using the Mask_RCNN 101 by default
    SCORE_THRESH_TEST: The model will keep the prediction result with the score >= SCORE_THRESH_TEST, always 0.5 - 0.7

setup(args): It will config the model using the given parameters in the ARGS object. If there the MODEL_DIR exists or was given but MODEL_CKPT doesn't exists or given, then it will use the final model in the MODEL_DIR.If neither the MODEL_DIR nor MODEL_CKPT exists or was given then it will use the pretrained model with the given backbone

Image_Prediction(args,file_name): To do the prediction on a given file, the output image will be stored in the current path with the original filename + _pred and _pred_mask suffix, the return will be the prediction results.
    args     : the ARGS object for config the model
    file_name：the image path and name which you want do the prediction on

Video_Prediction(args,file_name): Do the prediction on the video
    args     : the ARGS object for config the model
    file_name：the video path and name which you want do the prediction on
'''

import detectron2
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer,ColorMode
from detectron2.utils.video_visualizer import VideoVisualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

import matplotlib.pyplot as plt
import numpy as np
import time
import tqdm
import torch
import glob
import os, json, cv2, random


class ARGS:
    '''
    Arguments for setting up the model:
    MODEL_DIR: the folder under the current path which store the best model, I will always upload the best model in this folder
    MODEL_CKPT: the model's name which was named by its checkpoints, in the best_model folder, the model is eg. model_0012499.pth
    BACKBONE: We are using the Mask_RCNN 101 by default
    SCORE_THRESH_TEST: The model will keep the prediction result with the score >= SCORE_THRESH_TEST, always 0.5 - 0.7
    '''
    def __init__(
            self, MODEL_DIR, 
            MODEL_CKPT = None, 
            BACKBONE = "COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml",
            SCORE_THRESH_TEST = 0.7
           ):
        self.MODEL_DIR = MODEL_DIR
        self.MODEL_CKPT = MODEL_CKPT
        self.BACKBONE = BACKBONE
        self.SCORE_THRESH_TEST = SCORE_THRESH_TEST
  
  
def setup(args):
    '''
    It will config the model using the given parameters in the ARGS object.
    If there the MODEL_DIR exists or was given but MODEL_CKPT doesn't exists or given, then it will use the final model in the MODEL_DIR
    If neither the MODEL_DIR nor MODEL_CKPT exists or was given then it will use the pretrained model with the given backbone
    
    '''
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(args.BACKBONE)) # try some bigger 

    if args.MODEL_DIR is not None and args.MODEL_CKPT is not None and os.path.isfile(os.path.join(args.MODEL_DIR, "{}.pth".format(args.MODEL_CKPT))):
        cfg.MODEL.WEIGHTS = os.path.join(args.MODEL_DIR, "{}.pth".format(args.MODEL_CKPT)) 
        print('Use model weights from:', cfg.MODEL.WEIGHTS)
    # evaluate finetuned with ungiven checkpoint -> model_final
    elif args.MODEL_DIR is not None and args.MODEL_CKPT is None and os.path.isfile(os.path.join(args.MODEL_DIR, "{}.pth".format("model_final"))):
        cfg.MODEL.WEIGHTS = os.path.join(args.MODEL_DIR, "{}.pth".format("model_final")) 
        print('Use model weights from:', cfg.MODEL.WEIGHTS)
    # if not model given -> evaluete with pretrained
    else:
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(args.BACKBONE)
        print("Apply pre-trained "+args.BACKBONE+" model")

    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2  
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.SCORE_THRESH_TEST  # set the testing threshold for this model

    return cfg
    
    
def save_prediction(output,file_name):
    
    '''
    To save the mask with the color end 0 :(128, 0, 0) side 1: (0, 128, 0), used in the Image_Prediction() function
    
    output: the output prediction of the model
    
    file_name: the file name which the model did the prediction on, for read the original image file and draw the prediction on it
    
    '''
    
    image = cv2.imread(file_name)
    mask = np.zeros((image.shape))
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    instances = output["instances"].to("cpu")


    v = Visualizer(image,MetadataCatalog.get("MOBOT_SampleDataset"), instance_mode = ColorMode.SEGMENTATION,) 

    # Draw the side firstly, because if there are overlap areas on end and side, then end's color can cover the side's color, 
    # to make red always in front of green and there will not be red + green ≈ yellow on the masks

    for ind,(box,mask_,class_) in enumerate(list(zip(instances.pred_boxes,instances.pred_masks,instances.pred_classes))):

            if class_ == 1: # side
                mask[:,:,1][mask_] = 128
                visualization = v.draw_binary_mask(np.array(mask_),alpha = 0.5, color='green')
                visualization = v.draw_box(box,alpha = 1,edge_color = 'green')
                visualization = v.draw_text(str(ind+1)+'.side',box[:2],color = 'green', font_size = 28)

    for ind,(box,mask_,class_) in enumerate(list(zip(instances.pred_boxes,instances.pred_masks,instances.pred_classes))):


            if class_ == 0: # end
                mask[:,:,0][mask_] = 128
                mask[:,:,1][mask_] = 0
                visualization = v.draw_binary_mask(np.array(mask_),alpha = 0.5, color='red')
                visualization = v.draw_box(box,alpha = 1,edge_color = 'red')
                visualization = v.draw_text(str(ind+1)+'.end',box[:2],color = 'red', font_size = 28)


    cv2.imwrite(file_name.split('.')[0]+'_pred' +'.png', visualization.get_image()[:,:,::-1]) 
    cv2.imwrite(file_name.split('.')[0]+'_pred_mask' +'.png', mask[:,:,::-1])       


    print('Saved prediction as',file_name.split('.')[0]+'_pred' +'.png')
    print('Saved mask as',file_name.split('.')[0]+'_pred_mask' +'.png')
    
    
def Image_Prediction(args,file_name):
    
    '''
    To do the prediction on a given file_name, the output image will be stored in the current path with the original filename + _pred and _pred_mask suffix, the return will be the prediction results.
    args: the ARGS object for config the model
    file_name：the image path and name which you want do the prediction on
    '''
    cfg = setup(args)
    image = cv2.imread(file_name)
    predictor = DefaultPredictor(cfg)
    output = predictor(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    save_prediction(output,file_name)

    return output
 
 
def runOnVideo(predictor, video, maxFrames, file_name):
    """ 
    Runs the predictor on every frame in the video (unless maxFrames is given),
    and returns the frame with the predictions drawn.
    """
    if not os.path.exists(file_name.split('.')[0]+'_pred'):
        os.mkdir(file_name.split('.')[0]+'_pred')
    
    readFrames = 0
    while True:
        hasFrame, frame = video.read()
        if not hasFrame:
            break

        # Get prediction results for this frame
        output = predictor(frame)
        instances = output["instances"].to("cpu")
        
        # Make sure the frame is colored
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        v = Visualizer(frame,MetadataCatalog.get("MOBOT_SampleDataset"), instance_mode = ColorMode.SEGMENTATION,) 

        for box,mask_,class_ in list(zip(instances.pred_boxes,instances.pred_masks,instances.pred_classes)):

            if class_ == 1: # side
                visualization = v.draw_binary_mask(np.array(mask_),alpha = 0.5, color='green')
                visualization = v.draw_box(box,alpha = 1,edge_color = 'green')
                visualization = v.draw_text('side',box[:2],color = 'green', font_size = 28)

            elif class_ == 0: # end
                visualization = v.draw_binary_mask(np.array(mask_),alpha = 0.5, color='red')
                visualization = v.draw_box(box,alpha = 1,edge_color = 'red')
                visualization = v.draw_text('end',box[:2],color = 'red', font_size = 28)

        if len(instances) > 0:
            # Convert Matplotlib RGB format to OpenCV BGR format
            visualization = cv2.cvtColor(visualization.get_image(), cv2.COLOR_RGB2BGR)

            # Write test image
            cv2.imwrite('./'+file_name.split('.')[0]+'_pred/'+ str(readFrames) +'.png', visualization)
        else:
            cv2.imwrite('./'+file_name.split('.')[0]+'_pred/'+ str(readFrames) +'.png', frame)
        
        yield visualization

        readFrames += 1
        if readFrames > maxFrames:
            break


def Video_Prediction(args,file_name):
    '''
    Do the prediction on the video
    args: the ARGS object for config the model
    file_name：the video path and name which you want do the prediction on
    '''
    cfg = setup(args)
    video = cv2.VideoCapture(file_name)
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames_per_second = video.get(cv2.CAP_PROP_FPS)
    num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    
    video_writer = cv2.VideoWriter(file_name.split('.')[0]+'_pred.mp4', fourcc=cv2.VideoWriter_fourcc(*"mp4v"), fps=float(frames_per_second), frameSize=(width, height), isColor=True)
    
    predictor = DefaultPredictor(cfg)
    
    for i,visualization in tqdm.tqdm(enumerate(runOnVideo(predictor, video, num_frames, file_name)), total=num_frames):

        # Write to video file
        video_writer.write(visualization)
    
    print('Saved prediction as',file_name.split('.')[0]+'_pred' +'.mp4')
    print('Saved frames in /'+file_name.split('.')[0]+'_pred')

    # Release resources
    video.release()
    video_writer.release()
