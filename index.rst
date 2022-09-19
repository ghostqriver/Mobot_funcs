Tutorials
======================================
This is the documentation file for the Mobot project, it contains 5 modules in total. As the project is updated, the content of this document will be continuously supplemented. For the examples on how to use the Mobot_funcs functions you can read the `example notebook <https://github.com/ghostqriver/Mobot_funcs/blob/main/Mobot_example.ipynb/>`_.

train
----------------------------------
``train.py``

test
----------------------------------
``test.py``
This file contains some functions for test the model, set the metadata which is the dataset will be used for test your self before do the test. 

ARGS: 
    The model parameters for setup should give to the test functions.
    
    parameters:
        MODEL_DIR         : the directory path which stores the model
        
        OUTPUT_DIR        : the directory path which you wan to store the test result 
        
        MODEL_CKPT        : the model checkpoint's file name (eg. model_0010499.pth --> use model_0010499)
        
        BACKBONE          : the BACKBONE we are using, default value is "COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml" 
        
        SCORE_THRESH_TEST : the prediction with a confidence score above the threshold value is kept, and the remaining are discarded, 0.7 by default

setup(args): 
    It will config the model using the given parameters in the ARGS object. If there the MODEL_DIR exists or was given but MODEL_CKPT doesn't exists or given, then it     will use the final model in the MODEL_DIR.If neither the MODEL_DIR nor MODEL_CKPT exists or was given then it will use the pretrained model with the given backbone

get_best_model(dir): 
    Return the best model's name which saved in the ../Mobot/best_model or a given directory, only works when there is one .pth file in a certain length path, for some     complex path please set the model name yourself.
    
    parameters:
        dir : the directory path which only store one model.pth file which should be the best model

test(args,metadata): 
    Do the test use the given model (args) on the given dataset (metadata), will return a score_list contain the scores results.
    
    parameters:
        args    : the ARGS object
        
        metadata: should be defined with detectron2.data.MetadataCatalog.get() function which contain the at least json_file and image_root informations

test_candidate(metadata,candidate_dir): 
    The function will read the .pth file in the candidate_dir (default:best_model/candidate_model) folder automatically, and use the given dataset in metadata to get the result score, it will return a score list, also show the score table, when the model path is complex then this function can not work maybe, then define your owns model pred iteration please.
    
    parameters:
        metadata     : should be defined with detectron2.data.MetadataCatalog.get() function which contain the at least json_file and image_root informations
        
        candidate_dir: the path which all models in it are the candidate models you want to test

prediction
----------------------------------
``prediction.py``
This file contains some functions for predict use the model.

ARGS: 
    Arguments for setting up the model.
     
    parameters:
        MODEL_DIR        : the folder under the current path which store the best model, I will always upload the best model in this folder
        
        MODEL_CKPT       : the model's name which was named by its checkpoints, in the best_model folder, the model is eg. model_0012499.pth
        
        BACKBONE         : We are using the Mask_RCNN 101 by default
        
        SCORE_THRESH_TEST: The model will keep the prediction result with the score >= SCORE_THRESH_TEST, always 0.5 - 0.7

setup(args): 
    It will config the model using the given parameters in the ARGS object. If there the MODEL_DIR exists or was given but MODEL_CKPT doesn't exists or given, then it will use the final model in the MODEL_DIR.If neither the MODEL_DIR nor MODEL_CKPT exists or was given then it will use the pretrained model with the given backbone

Image_Prediction(args,file_name): 
    To do the prediction on a given file, the output image will be stored in the current path with the original filename + _pred and _pred_mask suffix, the return will be the prediction results. 
    
    parameters:
    args     : the ARGS object for config the model
    
    file_name：the image path and name which you want do the prediction on

Video_Prediction(args,file_name): 
    Do the prediction on the video.
     
    parameters:
    args     : the ARGS object for config the model
    
    file_name：the video path and name which you want do the prediction on
    
transform
----------------------------------
``transform.py``

c(img): 
    Transform a img from BGR to RGB.
    
    parameters:
        img: a read in image in the BGR format
    
automatic_brightness_and_contrast(image, clip_hist_percent): 
    The function automatically changed brightness and contrast of a given image.
    
    parameters:
        img              : a readin image
        
        clip_hist_percent: the parameter which control how much will be clip in the hist of original image's grayscale histogram, 10 by default   

brighter_CLAHE(img,clipLimit,tileGridSize): 
    The function apply the CLAHE on a given image.
    
    parameters:
        clipLimit,tileGridSize: the main parameters which should be given when apply the CLAHE
        
        clipLimit         : float, 3.0 by default
        
        tileGridSize      : 1*2 tuple

brightening_dataset(brightening_func,image_root,tar_folder,para = None):
    Do the transformation using the brightening_func on a given dataset.
    
    parameters:
        brightening_func: the brighten function's name
        
        image_root      : the images path which store all images of the dataset
        
        tar_folder      : the path which the transformed images should be stored in
        
        para            : the first parameter for brightening_func, because here we only define two functions brighter_CLAHE and automatic_brightness_and_contrast both with the default parameter, for further using, we can modify this parameter for more complex transformation
          
visualize
----------------------------------
``visualize.py`` This file is used for visualizzation for checking whether the current augmentation or coco json file works or not.

coco_json_show(json_file,image_path,image_name=None): 
    Given the path of json_file and images' path, random show 5 images with its annotations in the coco json file. If given a image_name, then only show that image.
        
    parameters:
        json_file: the path of the json
        
        image_path: the path contain the images in the json
        
        image_name: a certain file name, if given will only show this image
