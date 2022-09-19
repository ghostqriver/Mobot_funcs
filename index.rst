Tutorials
======================================
This is the documentation file for the Mobot project, it contains 5 modules in total. As the project is updated, the content of this document will be continuously supplemented. For the examples on how to use the Mobot_funcs functions you can read the <https://github.com/ghostqriver/Mobot_funcs/blob/main/Mobot_example.ipynb>

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
    Return the best model's name which saved in the ../Mobot/best_model or a given directory, only works when there is one .pth file in a certain length path, for some     complex path please set the model name yourself
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

transform
----------------------------------
``transform.py``

c(img): 
    Transform a img from BGR to RGB
    parameters:
        img: a read in image in the BGR format

visualize
----------------------------------
``visualize.py``
