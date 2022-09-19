import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

import logging 
import os
from collections import OrderedDict
import torch
from torch.nn.parallel import DistributedDataParallel
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
from detectron2.evaluation import (
    CityscapesInstanceEvaluator,
    CityscapesSemSegEvaluator,
    COCOEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
    LVISEvaluator,
    PascalVOCDetectionEvaluator,
    SemSegEvaluator,
    inference_on_dataset,
    print_csv_format,
)
from detectron2.modeling import build_model
from detectron2.solver import build_lr_scheduler, build_optimizer
from detectron2.utils.events import EventStorage
from detectron2.engine import DefaultPredictor
from detectron2.utils.video_visualizer import VideoVisualizer
from detectron2.utils.visualizer import Visualizer,ColorMode

from detectron2.data import transforms as T
import albumentations as A
from detectron2.data import detection_utils as utils
from detectron2.data import DatasetMapper


import time
import matplotlib.pyplot as plt
import numpy as np
import json 
import glob
import cv2
import tqdm
import copy
import random
import seaborn as sns 
import math

logger = logging.getLogger("detectron2")




class ARGS:
    def __init__(
            self, MODEL_DIR, OUTPUT_DIR, BASE_LR, MAX_ITER, STEPS,
            CHECKPOINT_PERIOD, EVAL_PERIOD,
            MODEL_CKPT=None, BACKBONE="COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml",
            resume=False, ):
        self.MODEL_DIR,self.OUTPUT_DIR = MODEL_DIR,OUTPUT_DIR
        self.BASE_LR,self.MAX_ITER = BASE_LR,MAX_ITER
        self.STEPS = STEPS
        self.CHECKPOINT_PERIOD = CHECKPOINT_PERIOD
        self.EVAL_PERIOD = EVAL_PERIOD
        self.MODEL_CKPT = MODEL_CKPT
        self.BACKBONE = BACKBONE
        self.resume=resume

def setup(args):
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

    # cfg.DATASETS.TRAIN = ("MOBOT_SampleDataset",)    
    cfg.DATASETS.TRAIN = ("MOBOT_Train",)
    cfg.DATASETS.TEST = ("MOBOT_Val",)
    
    
    cfg.SOLVER.IMS_PER_BATCH = 3  # This is the real "batch size" commonly known to deep learning people 每个batch有几张图像
    
    
    # Checkpoint 记录时间应该和eval_period 保持一致
    
    
    cfg.DATALOADER.NUM_WORKERS = 2 
    cfg.DATALOADER.SHUFFLE = True
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 256 
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2  # only has one class (ballon). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
    # NOTE: this config means the number of classes, but a few popular unofficial tutorials incorrect uses num_classes+1 here.
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set the testing threshold for this model


    
    cfg.OUTPUT_DIR=args.OUTPUT_DIR
    cfg.SOLVER.BASE_LR = args.BASE_LR # pick a good LR 0.00025
    cfg.SOLVER.MAX_ITER = args.MAX_ITER
    cfg.SOLVER.STEPS = args.STEPS 
    
    cfg.SOLVER.CHECKPOINT_PERIOD = args.CHECKPOINT_PERIOD # Save the checkpoint each 2000 iterations 
    cfg.TEST.EVAL_PERIOD = args.EVAL_PERIOD # the period to run eval_function. Set to 0 to not evaluate periodically (but still evaluate after the last iteration if eval_after_train is True).                                   

    
    return cfg
    

def get_evaluator(cfg, dataset_name, output_folder=None):
    """
    Our dataset's evaluator_type is coco by default, so it will only call COCOEvaluator here
    """
    if output_folder is None:
        output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        
    evaluator_list = []
    evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type

    if evaluator_type in ["coco", "coco_panoptic_seg"]:
        evaluator_list.append(COCOEvaluator(dataset_name, output_dir=output_folder))

    if len(evaluator_list) == 1:
        return evaluator_list[0]
    return DatasetEvaluators(evaluator_list)


def do_test(cfg, model):
    print('New test function:')
    results = OrderedDict() #有序字典
    for dataset_name in cfg.DATASETS.TEST: # Read the tuple in cfg.DATASETS.TEST use all Test sets
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
    

def gen_lr_list_choosing_lr(cfg,start,start_lr=1e-7,end_lr=1e-2 ):
    
    start_lr = start_lr
    end_lr = end_lr
    iters = cfg.SOLVER.MAX_ITER
    lr_list = list(np.zeros(start))

    lr = start_lr
    while lr < end_lr:
        lr_list_ = list(np.linspace(lr,lr*10,int((iters-start)/5)))#logspace
        lr_list = lr_list + lr_list_
        lr = lr*10
    lr_list = np.array(lr_list).flatten()
    p1 = plt.plot(lr_list)
    plt.show()
    return lr_list
    
    
def gen_lr_list_1(cfg,start,min_lr,max_lr):
    
    each = int(len(glob.glob(MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).image_root+'/*.png'))/cfg.SOLVER.IMS_PER_BATCH) #*5 # about one epoch
    lr_list = list(np.zeros(start))   
    i = 0 # counter to decide increase or decrease
    
    while len(lr_list) < cfg.SOLVER.MAX_ITER:
        if i%2 == 0: 
            lr_list = lr_list + list(np.sort(np.random.uniform(min_lr,max_lr,each)))
        else: 
            lr_list = lr_list + list(np.sort(np.random.uniform(max_lr,min_lr,each))[::-1])
        i += 1
    
        lr_list = list(np.array(lr_list).flatten())
    plt.plot(lr_list)
    plt.title('Learning Rate')
    plt.show()
    return lr_list

def gen_lr_list_2(cfg,start,min_lr = 5e-06,max_lr = 0.000298):#lr2
    
    each = math.ceil((cfg.SOLVER.MAX_ITER-start)*(1/5))*2
    lr_list = list(np.zeros(start))    
    
    
    lr_list = lr_list + list(np.sort(np.random.uniform(min_lr,max_lr,each)))
 
    lr_list = lr_list + list(np.sort(np.random.uniform(max_lr,min_lr,each))[::-1])
    
    lr_list = lr_list + list(np.sort(np.random.uniform(min_lr,min_lr/100,math.ceil(each/2)))[::-1])
    
    lr_list = list(np.array(lr_list).flatten())
    plt.plot(lr_list)
    plt.title('Learning Rate interval')
    plt.show()
    return lr_list
    

def color_augmentation(img):
    myimg = img.copy()
    #myimg = c(myimg)
    transform = A.Compose([
                A.RandomBrightnessContrast(brightness_limit=[-0.1,0.2],contrast_limit=0.3,),
                A.transforms.HueSaturationValue(p=0.5),
#                 A.RandomGamma(p=1),
                A.Blur()
                ])
    transformed = transform(image = myimg)
    return transformed["image"]
    

class PixelAugmentation(T.Augmentation):
    def __init__(self):
        super(PixelAugmentation, self).__init__()
        
    def get_transform(self, image):
        return T.ColorTransform(lambda img : color_augmentation(img))


def do_train(cfg, model, file_name, resume = False, lr_strategy = None, min_lr = None, max_lr = None, Aug = False,):
    model.train() 
    optimizer = build_optimizer(cfg, model) 
    scheduler = build_lr_scheduler(cfg, optimizer) # Build but won't be used, we use the  lr_list

    # Load the iteration times from last training, then start with this number+1, end with max_iter in config
    checkpointer = DetectionCheckpointer( # Create a checkerpoint object which will save the checker point into cfg.OUTPUT_DIR
        model, cfg.OUTPUT_DIR, optimizer=optimizer, scheduler=scheduler
    )
    start_iter = ( # Load the current model's checkpoint information to do iteration
        checkpointer.resume_or_load(cfg.MODEL.WEIGHTS, resume=resume).get("iteration", -1) + 1 
    )
    max_iter = cfg.SOLVER.MAX_ITER

    
    # Generate lr_list
    use_scheduler = False
    if lr_strategy == 1:
        lr_list = gen_lr_list_1(cfg, start_iter, min_lr, max_lr)
    elif lr_strategy == 2:
        lr_list = gen_lr_list_2(cfg, start_iter, min_lr, max_lr)
    elif lr_strategy == 3:
        lr_list = gen_lr_list_choosing_lr(cfg, start_iter, min_lr, max_lr)
    else:
        use_scheduler = True

    periodic_checkpointer = PeriodicCheckpointer(
        checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD, max_iter=max_iter
    )

    writers = default_writers(cfg.OUTPUT_DIR, max_iter) if comm.is_main_process() else []

    
    if Aug == True:
        data_loader = build_detection_train_loader(cfg, mapper =  DatasetMapper
                                                   (cfg, is_train=True, augmentations=[
                                                                                          T.Resize((720,1280)),
                                                                                          T.RandomApply(PixelAugmentation(),prob=1),
                                                                                          T.RandomApply(T.RandomCrop("relative_range",[0.8,0.8]),prob=1),
                                                                                          T.RandomApply(T.RandomRotation(angle=[-15, 15],),prob=0.8),
                                                                                          T.RandomFlip(prob=0.5, horizontal=True, vertical=False)
                                                                                       ]
                                                   )
                                                  )
    else:
        data_loader = build_detection_train_loader(cfg)
    
    
    logger.info("Starting training from iteration {}".format(start_iter))
    
    # Record the losses and accuracy scores
    train_loss = np.array([0.,0.,0.,0.,0.]) # 'loss_cls''loss_box_reg''loss_mask''loss_rpn_cls''loss_rpn_loc'
    train_losses = 0
    test_acc = OrderedDict()
    train_loss_list = OrderedDict()
    
    with EventStorage(start_iter) as storage:
        for data, iteration in zip(data_loader, range(start_iter, max_iter)):
           
            storage.iter = iteration
            
            if use_scheduler == False:
                optimizer.param_groups[0]["lr"] = lr_list[iteration] 
            
            
           # Show the images in the dataloader, aug or nonaug
#             print(str(len(data))+" images each batch")
#             for i in data:
#                 plt.imshow(c(np.transpose(i['image'],(1,2,0))))
#                 plt.title(i['file_name'])
#                 plt.show()
            
            loss_dict = model(data)
            losses = sum(loss_dict.values())
            
            assert True, loss_dict
#             assert torch.isfinite(losses).all(), loss_dict
            
            loss_dict_reduced = {k: v.item() for k, v in comm.reduce_dict(loss_dict).items()}
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())
            if comm.is_main_process():
                storage.put_scalars(total_loss=losses_reduced, **loss_dict_reduced)

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            storage.put_scalar("lr", optimizer.param_groups[0]["lr"], smoothing_hint=False)
            
            lr = optimizer.param_groups[0]["lr"]
            
            if use_scheduler == True:
                scheduler.step()

            if (
                cfg.TEST.EVAL_PERIOD > 0
                and (iteration + 1) % cfg.TEST.EVAL_PERIOD == 0
                and iteration != max_iter - 1
            ):
                test_acc[iteration+1] = do_test(cfg, model)
                test_acc[iteration+1]['lr'] = lr

                # Compared to "train_net.py", the test results are not dumped to EventStorage
                comm.synchronize()
            
            train_loss += np.array(list(loss_dict_reduced.values()))
            train_losses += losses_reduced
            
            if iteration - start_iter > 5 and (
                (iteration + 1) % 20 == 0 or iteration == max_iter - 1
            ):
                
                train_loss_list[iteration+1] = {key:value for key,value in  zip(loss_dict_reduced.keys(),train_loss / 20)}
                train_loss_list[iteration+1]['total_loss'] = train_losses / 20
                train_loss_list[iteration+1]['lr'] = lr

                train_loss = np.array([0.,0.,0.,0.,0.])
                train_losses = 0
                
                for writer in writers:
                    writer.write()
                    
            # Save the score list
            if (
                cfg.TEST.EVAL_PERIOD > 0
                and (iteration + 1) % cfg.TEST.EVAL_PERIOD == 0
                and iteration != max_iter - 1
            ):
                np.save(cfg.OUTPUT_DIR+'/'+file_name+'.npy', {'train_loss':train_loss_list,'test_acc':test_acc}) 
                    
            periodic_checkpointer.step(iteration)
            
    return {'train_loss':train_loss_list,'test_acc':test_acc}


def train(args,score_file_name = '{}',lr_strategy=None,min_lr=None, max_lr=None, Aug=False):
    cfg = setup(args)

    model = build_model(cfg)
    
    logger.info("Model:\n{}".format(model))
    

    scores = do_train(cfg, model, resume=args.resume, file_name=score_file_name,lr_strategy=lr_strategy,min_lr=min_lr,max_lr=max_lr,Aug=Aug)
    scores['test_acc'][cfg.SOLVER.MAX_ITER] = do_test(cfg, model)
    np.save(cfg.OUTPUT_DIR+'/'+score_file_name+'_final.npy', scores)
    return scores
        
def showscore(scores,AP_bbox=True,AP_segm=True,AP_segm_end=True,AP_segm_side=True):
    
    if AP_bbox == True:
        AP_bbox = list(map(lambda a:a['AP'],list(map(lambda a:a['bbox'],list(scores['test_acc'].values())))))
        plt.plot(list(scores['test_acc'].keys())
             ,AP_bbox,label='bbox AP')
        
    if AP_segm == True:
        AP_segm = list(map(lambda a:a['AP'],list(map(lambda a:a['segm'],list(scores['test_acc'].values())))))
        plt.plot(list(scores['test_acc'].keys())
             ,AP_segm,label='segm AP')
        
    if AP_segm_end == True:
        AP_segm_end = list(map(lambda a:a['AP-end'],list(map(lambda a:a['segm'],list(scores['test_acc'].values())))))
        plt.plot(list(scores['test_acc'].keys())
             ,AP_segm_end,label='segm AP end')

        
    if AP_segm_side == True:    
        AP_segm_side = list(map(lambda a:a['AP-side'],list(map(lambda a:a['segm'],list(scores['test_acc'].values())))))
        plt.plot(list(scores['test_acc'].keys())
             ,AP_segm_side,label='segm AP side')
        
    AP_segm_end_max = max(AP_segm_end)

    plt.vlines(list(scores['test_acc'].keys())[np.argmax(np.array(AP_segm_end))],AP_segm_end_max-8,AP_segm_end_max+3,'red','dashed')
    for i in AP_segm_end:
        plt.text(list(scores['test_acc'].keys())[np.where(np.array(AP_segm_end) == i)[0][0]],i,round(i,2),fontsize='x-small')
    plt.xticks(list(scores['test_acc'].keys()))
    plt.xlabel('iteration/checkpoints')
    plt.legend()
    
#     print(AP_segm_end)
    print("The highest score on segm Ap end here is",AP_segm_end_max,', in checkpoint',list(scores['test_acc'].keys())[np.argmax(np.array(AP_segm_end))]-1)