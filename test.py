'''
This file contains some functions for test the model, set the metadata which is the dataset will be used for test your self before do the test. 

ARGS: The model parameters for setup should give to the test functions.
    MODEL_DIR         : the directory path which stores the model
    OUTPUT_DIR        : the directory path which you wan to store the test result 
    MODEL_CKPT        : the model checkpoint's file name (eg. model_0010499.pth --> use model_0010499)
    BACKBONE          : the BACKBONE we are using, default value is "COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml" 
    SCORE_THRESH_TEST : the prediction with a confidence score above the threshold value is kept, and the remaining are discarded, 0.7 by default

setup(args): It will config the model using the given parameters in the ARGS object. If there the MODEL_DIR exists or was given but MODEL_CKPT doesn't exists or given, then it will use the final model in the MODEL_DIR.If neither the MODEL_DIR nor MODEL_CKPT exists or was given then it will use the pretrained model with the given backbone

get_best_model(dir): Return the best model's name which saved in the ../Mobot/best_model or a given directory, only works when there is one .pth file in a certain length path, for some complex path please set the model name yourself
    dir : the directory path which only store one model.pth file which should be the best model

test(args,metadata): Do the test use the given model (args) on the given dataset (metadata), will return a score_list contain the scores results.
    args    : the ARGS object
    metadata: should be defined with detectron2.data.MetadataCatalog.get() function which contain the at least json_file and image_root informations

test_candidate(metadata,candidate_dir): The function will read the .pth file in the candidate_dir (default:best_model/candidate_model) folder automatically, and use the given dataset in metadata to get the result score, it will return a score list, also show the score table, when the model path is complex then this function can not work maybe, then define your owns model pred iteration please.
    metadata     : should be defined with detectron2.data.MetadataCatalog.get() function which contain the at least json_file and image_root informations
    candidate_dir: the path which all models in it are the candidate models you want to test
    
    



mask2polygon(mask): Transform a mask(n*m) array into the polygon(2*d) array

test_to_coco(args, metadata): Give the model parameters and the metadata(read from the MetadataCatalog.get()) as the test set for prediction, 
                            then save all the prediction result into cocojson file format, in the same sequence to the original dataset's json file.
                            
 
 
'''

import os
from collections import OrderedDict
import torch
import logging

import detectron2
from detectron2 import model_zoo
import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer, PeriodicCheckpointer
from detectron2.config import get_cfg
from detectron2.data import (
    MetadataCatalog,
    DatasetCatalog,
    build_detection_test_loader,
    build_detection_train_loader,
)
from detectron2.engine import default_argument_parser, default_setup, default_writers, launch
from detectron2.evaluation import COCOEvaluator,DatasetEvaluators,inference_on_dataset,print_csv_format
from detectron2.modeling import build_model
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
import matplotlib.pyplot as plt
import numpy as np
import glob
import tqdm
import random
import cv2
from shapely.geometry import Polygon

logger = logging.getLogger("detectron2")

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import pylab,json


class ARGS:
    '''
    The model parameters for setup should give to the test functions.
    MODEL_DIR : the directory path which stores the model
    OUTPUT_DIR: the directory path which you wan to store the test result 
    MODEL_CKPT: the model checkpoint's file name (eg. model_0010499.pth --> use model_0010499)
    BACKBONE  : the BACKBONE we are using, default value is "COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml" 
    SCORE_THRESH_TEST : the prediction with a confidence score above the threshold value is kept, and the remaining are discarded, 0.7 by default
    
    '''
    def __init__(
            self, MODEL_DIR, OUTPUT_DIR, 
            MODEL_CKPT = None, 
            BACKBONE = "COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml",
            SCORE_THRESH_TEST = 0.7
                ):
        self.MODEL_DIR,self.OUTPUT_DIR = MODEL_DIR,OUTPUT_DIR
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
    cfg.merge_from_file(model_zoo.get_config_file(args.BACKBONE)) 

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
    cfg.OUTPUT_DIR = args.OUTPUT_DIR
    
    return cfg


def get_evaluator(cfg, dataset_name, output_folder=None):
    '''
    Get the COCOEvaluator
    
    '''
    if output_folder is None:
        output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        
    evaluator_list = []

    evaluator_list.append(COCOEvaluator(dataset_name, output_dir=output_folder,))

    if len(evaluator_list) == 1:
        return evaluator_list[0]
    
    return DatasetEvaluators(evaluator_list)


def do_test(cfg, model):
    '''
    Do the test and return the test score on the cfg.DATASETS.TEST dataset, which will be set using the given metadata in test function, or you can modify the setup() function to setup a default one
    using cfg.DATASETS.TEST = ('MOBOT_Test',)
    
    '''
    results = OrderedDict() 
    for dataset_name in cfg.DATASETS.TEST: 
        data_loader = build_detection_test_loader(cfg, dataset_name)
        evaluator = get_evaluator( # Get the evaluator according to the given evaluator_type
            cfg, dataset_name, os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name)
        )
        results_i = inference_on_dataset(model, data_loader, evaluator) # Predict
        results[dataset_name] = results_i
        if comm.is_main_process(): 
            logger.info("Evaluation results for {} in csv format:".format(dataset_name))
            print_csv_format(results_i)
    if len(results) == 1: # Only one given test set
        results = list(results.values())[0]
    return results

def get_best_model(dir = '../Mobot/best_model'):
    '''
    Return the best model's name which saved in the ../Mobot/best_model or a given directory, only works when there is one .pth file in a certain length path, for some complex path please set the model name yourself
    
    '''
    return glob.glob(dir+'*/.pth')[0].split('.')[-2].split('/')[-1]
    
    
def print_score(model_list,score_list):
    '''
    For show the scores in the sequence segm AP-end, segm AP-side, segm AP, bbox AP-end, bbox AP-side, bbox AP in a table
    
    '''
    print('\t model \t',end = ' |\t')
    for model in model_list:
        print(model,end = '\t|\t')
    print()
    for i in ['segm','bbox']:
        for j in ['AP-end','AP-side','AP']:
            print(i+'\t'+j+'\t',end = ' |\t')
            
            for score in score_list:
                    print(round(score[i][j],5),end= '\t|\t')
                    
            print()
            
            
def test(args,metadata):
    '''
    Do the test use the given model (args) on the given dataset (metadata), will return a score_list contain the scores results.
    args: the ARGS object
    metadata: should be defined with detectron2.data.MetadataCatalog.get() function which contain the at least json_file and image_root informations
    
    '''

    cfg = setup(args)

    model = build_model(cfg)

    logger.info("Model:\n{}".format(model))

    
    cfg.DATASETS.TEST = [metadata.name,]

    DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(  #DetectionCheckpointer: correctly load checkpoints
        cfg.MODEL.WEIGHTS, resume=False # resume_or_load: show the model final training information last time
    )
    
    score_list = do_test(cfg, model)
    
    model_name = args.MODEL_CKPT
    
    print_score([model_name],[score_list])
    return score_list


def test_candidate(metadata,candidate_dir = 'best_model/candidate_model'):
    '''
    The function will read the .pth file in the candidate_dir (default:best_model/candidate_model) folder automatically, and use the given dataset in metadata to get the result score, it will return a score list, also show the score table, when the model path is complex then this function can not work maybe, then define your owns model pred iteration please.
    metadata: should be defined with detectron2.data.MetadataCatalog.get() function which contain the at least json_file and image_root informations
    candidate_dir: the path which all models in it are the candidate models you want to test

    '''
    model_list = np.sort(list(map(lambda x:x.split('.')[-2].split('/')[-1],glob.glob(candidate_dir+'/*.pth'))))
    score_list = []
    
    for model in model_list:
        
        args = ARGS(
                    MODEL_DIR = candidate_dir,
                    OUTPUT_DIR = 'test' ,
                    MODEL_CKPT =  model ,
                    BACKBONE = "COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml",
                    )

    
        cfg = setup(args)

        model = build_model(cfg)

        logger.info("Model:\n{}".format(model))

        cfg.DATASETS.TEST = [metadata.name,]

        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(  
            cfg.MODEL.WEIGHTS, resume=False 
        )
        score_list.append(do_test(cfg, model))
    print_score(model_list,score_list)
    return score_list

            
def mask2polygon(mask):
    '''
    Transform the mask into polygon
    mask: 
    '''
    contours, hierarchy = cv2.findContours((mask).astype(np.uint8), cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    # mask_new, contours, hierarchy = cv2.findContours((mask).astype(np.uint8), cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    segmentation = []
    for contour in contours:
        contour_list = contour.flatten().tolist()
        if len(contour_list) > 4:# and cv2.contourArea(contour)>10000
            segmentation.append(contour_list)
    return segmentation
    
def test_to_coco(args, metadata):
   
    '''
    Save the prediction result into coco json format
    
    val_set = False, set to True use the validation set to do the test
    '''
    
    cfg = setup(args)
    predictor = DefaultPredictor(cfg)
    # metadata = MetadataCatalog.get("MOBOT_Test")
    # if val_set:
    #     cfg.DATASETS.TEST = ("MOBOT_Val",)
    #     metadata = MetadataCatalog.get("MOBOT_Val")
    # predictor = DefaultPredictor(cfg)
    
    json_ = detectron2.data.datasets.load_coco_json(metadata.json_file, metadata.image_root, metadata.name, extra_annotation_keys=None)
    annos = []
    id_ = 0
    for d in tqdm.tqdm(json_, total = len(json_)):   
#     for d in random.sample(json_,5):
        input_ = cv2.imread(d['file_name'])
        print(d['file_name'])
        image_id = d['image_id'] 
        
        output = predictor(input_)
        
        instances = output["instances"].to("cpu")
        
        for ind,(box,mask_,class_,score) in enumerate(list(zip(instances.pred_boxes,instances.pred_masks,instances.pred_classes,instances.scores))):
            
            polygon = mask2polygon(np.array(mask_))

            area = Polygon(list(zip(polygon[0][0::2],polygon[0][1::2]))).area             
            
            bbox = list(np.array(box,dtype=float))
            bbox[2],bbox[3] = bbox[2]-bbox[0],bbox[3]-bbox[1]
            
            anno = {'iscrowd' : 0, 'bbox' : bbox,  'category_id' : int(class_), 
                    'image_id' : image_id, 'id' : id_ ,
                    'segmentation': polygon , 'area': area,
                    'score': int(score)}
            
            id_ += 1
#             print(anno)
            annos.append(anno)
            
    # Save the output into cocojson
    with open(metadata.json_file,'r+') as file:
        content = file.read()
    
    json_origin = json.loads(content)
    
    json_origin['annotations'] = annos

    new_json = json.dumps(json_origin)#转化为json格式文件

    with open(metadata.name+'_pred.json','w+') as file:
        file.write(new_json)

    print('Saved in',metadata.name+'_pred.json')
    
def list2poly(list_):
    poly = list(zip(list_[0][0::2],list_[0][1::2]))
    return poly

def poly_iou(poly1, poly2):
    poly1 = Polygon(poly1)
    poly2 = Polygon(poly2)
    intersection_area = poly1.intersection(poly2).area
    union_area = poly1.union(poly2).area
    return intersection_area / union_area
    

def total_IoU(gt_path,dt_path,image_root='../Datasets/Test',plot=False):

    '''
    Calculate the IoU in the saving prediction json file to the original file
    '''
    
    thres = 1000


    cocoGt = COCO(gt_path)
    cocoDt = COCO(dt_path)

    IoU_ls = []
    for img_id in cocoGt.getImgIds():
    
        # Read annos in Gt
        annIds_gt = cocoGt.getAnnIds(imgIds=[img_id],catIds=[0])
        annsInfo_gt = cocoGt.loadAnns(annIds_gt)

        # Read annos in Dt
        annIds_dt = cocoDt.getAnnIds(imgIds=[img_id],catIds=[0])
        annsInfo_dt = cocoDt.loadAnns(annIds_dt)

        img_name = cocoGt.loadImgs(img_id)[0]['file_name'] # Same to Dt, read one is enough
        img_size = (cocoGt.loadImgs(img_id)[0]['height'],cocoGt.loadImgs(img_id)[0]['width'])  # Same to Dt, read one is enough
    
        # Set different threshold for different img_size
        if img_size[1] == 1920:
            area_thres = thres * 1.5
        else:
            area_thres = thres
        
    
        # Plot the segementations on Gt and Dt     
        if plot:
            img = cv2.imread(os.path.join(image_root,img_name))
    
        if not (len(annIds_gt) == 0): # Only show images and calculate IoU which have the end segementation on ground truth

            print(img_id,':',img_name)
            print(img_size)

            show_im = 0
            
            if plot:
                visualizer = Visualizer(img[:, :, ::-1],  scale=1)

            for gt_segms in annsInfo_gt:

                ious = []

                if gt_segms['area'] > area_thres:
                    polygon_gt = list(zip(gt_segms['segmentation'][0][0::2],gt_segms['segmentation'][0][1::2]))
                    if plot:
                        left = gt_segms['bbox'][0]
                        top = gt_segms['bbox'][1]
                        plt.text(left+30,top+50,'area:'+str(gt_segms['area'] ),fontsize=8,color='white')
                        out = visualizer.draw_polygon(polygon_gt,color = 'red')
                    show_im += 1

                    for dt_segms in annsInfo_dt: 

                        polygon_dt = list(zip(dt_segms['segmentation'][0][0::2],dt_segms['segmentation'][0][1::2]))
                        iou = poly_iou(polygon_gt,polygon_dt)

                        if iou > 0: # Only keep the segmentation which IoU>0 to the ground truth       
                            if plot:
                                left = dt_segms['bbox'][0]
                                top = dt_segms['bbox'][1]
                                plt.text(left+30,top+15,'iou:'+str(round(iou,2)),fontsize=10,color='white')
                                out = visualizer.draw_polygon(polygon_dt,color = 'orange')
                            ious.append(iou)


                    if len(ious) == 0: 

                        IoU_ls.append(0) # Set this ground truth's IoU to 0
                        if plot:
                            left = gt_segms['bbox'][0]
                            top = gt_segms['bbox'][1]
                            plt.text(left+30,top+15,'iou:'+str(0),fontsize=10,color='white')

                    else:
                        for i in ious:
                            IoU_ls.append(i) # Append all ious of this ground truth into the IoU list

            if show_im > 0:
                if plot:
                    plt.imshow(out.get_image())
                    plt.show()
            
    print('Total Average IoU:',np.mean(IoU_ls))
    return np.mean(IoU_ls),IoU_ls