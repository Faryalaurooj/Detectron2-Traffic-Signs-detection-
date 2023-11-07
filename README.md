# train-object-detector-detectron2
In this repo, we will train detectron2 model on a custom dataset and evaluate the model performance. Detectron2 is a platform for object detection, segmentation and other visual recognition tasks. We will evaluate its perfromance as a detection model. 

![inbox_3400968_20ca377934c5ed5f8c1e4272c838b01a_ts_detections](https://github.com/Faryalaurooj/Detectron2-Traffic-Signs-detection-/assets/138756263/c669d87f-499d-4aab-b10d-5b6b33afdd49)

![Figure_1](https://github.com/Faryalaurooj/Detectron2-Traffic-Signs-detection-/assets/138756263/cff3283b-2af0-428e-8932-ab0e2f1142d2)


# Quick Start

## install requirements
Download the repo 

```
git clone repo https://github.com/Faryalaurooj/Detectron2-Traffic-Signs-detection
```

Install following requirements 
```
!pip install torch==2.0.0
!pip install torchvision==0.15.1
!pip install opencv-python==4.6.0.66
!pip install matplotlib==3.5.3
!pip install git+https://github.com/facebookresearch/detectron2.git
```
# Download dataset 
You need to download a dataset in yolo format , i used this link to download the dataset 
```
https://www.kaggle.com/datasets/valentynsichkar/traffic-signs-dataset-in-yolo-format
```

The data is in this format 
    """
    annotations should be provided in yolo format, this is: 
            class, xc, yc, w, h
    data needs to follow this structure:
    
    data-dir
    ----- train
    --------- imgs
    ------------ filename0001.jpg
    ------------ filename0002.jpg
    ------------ ....
    --------- anns
    ------------ filename0001.txt
    ------------ filename0002.txt
    ------------ ....
    ----- val
    --------- imgs
    ------------ filename0001.jpg
    ------------ filename0002.jpg
    ------------ ....
    --------- anns
    ------------ filename0001.txt
    ------------ filename0002.txt
    ------------ ....
    
    """

Make a class names file if you are using any other data with any other classes. In this repo the classes are 

[10/27 01:25:19 d2.data.build]: Distribution of instances among all 4 categories:
|  category   | #instances   |  category  | #instances   |  category  | #instances   |
|:-----------:|:-------------|:----------:|:-------------|:----------:|:-------------|
| prohibitory | 557          |   danger   | 219          | mandatory  | 163          |
|    othe     | 274          |            |              |            |              |
|    total    | 1213         |            |              |            |              |


# Run training 
Use following command if you are training on your own machine gpu (in train.py and util.py select device as cuda on all places)

```
!python train.py --device gpu --learning-rate 0.00001 --iterations 6000   
```



ELse you can use google collab for training 
1. open google collab  https://colab.research.google.com/
2.  upload this file there 'Train_Detectron2_Object_Detector_Custom_Data.ipynb' , its here in this repo.
3.  Then step by step , mount your google drive , upload your dataset and your code folder in your google drive
4.  install requirements , select from run time options , select T4 GPU
5.  change directory ``` %cd /content/gdrive/MyDrive/train-object-detector-detectron2```
6.  then run command ```!python train.py --learning-rate 0.00001 --iterations 6000 ``` ,


# Predict / Test
Run inference on all images in val folder using this command
```
python predict.py
```

we will get detections with following details for all the images: 

00899  = name of image
['1', '0.6551470588235294', '0.63375', '0.04191176470588235', '0.06375']
{'instances': Instances(num_instances=17, image_height=800, image_width=1360, fields=[pred_boxes: Boxes(tensor([[ 978.5349,  333.1594, 1010.5735,  365.5128],
        [ 992.7749,  382.6991, 1007.3533,  397.7653],
        [ 347.4969,  515.8345,  362.4913,  529.5972],
        [ 659.0547,  178.6254,  682.6287,  203.6596],
        [ 992.9014,  381.3641, 1008.4732,  396.9509],
        [ 356.6705,  473.9312,  367.5020,  488.5890],
        [ 297.5882,  462.0896,  310.7014,  481.6461],
        [ 662.8211,  498.4236,  678.5941,  514.8988],
        [ 338.2036,  516.5497,  354.2000,  530.7401],
        [ 388.0743,  480.5992,  398.6339,  492.6743],
        [1051.5106,  422.0099, 1066.0228,  440.2997],
        [1051.7937,  421.9811, 1066.2158,  440.1296],
        [ 381.9724,  480.4517,  392.2268,  492.6051],
        [ 589.7984,  501.7503,  614.8694,  517.4730],
        [ 292.0667,  485.1740,  306.1211,  499.2074],
        [ 355.3669,  510.6437,  368.5223,  523.5047],
        [ 295.8283,  472.7765,  309.0530,  488.2390]])), scores: tensor([0.9021, 0.1172, 0.0994, 0.0956, 0.0882, 0.0789, 0.0740, 0.0698, 0.0612,
        0.0592, 0.0583, 0.0547, 0.0540, 0.0532, 0.0524, 0.0504, 0.0501]), pred_classes: tensor([3, 0, 2, 3, 3, 0, 1, 2, 2, 0, 2, 3, 0, 2, 0, 2, 1])])}

so we are getting the number of instances (detections of traffic signs) , height , width , bounding box coordinates , confidence scores of each instance in the form of a .txt file.

# Evaluation 
Now we need to find Precision (P) Recall (R) and F1 scores for our evaluation results. 
Run following command to perfrom evaluation on all images in val folder
```
python eval.py
```

It will evaluate each image class-wise using the predicted results .txt file achived above which contains details like ( class , bounding box x , y coordinates , width , height).

we will get following results in evaluation for all the images :

Evaluation/labels/00825.txt
Class 0:
True Positives: 1
False Negatives: 0
False Positives: 0
Class 1:
True Positives: 0
False Negatives: 0
False Positives: 0
Class 2:
True Positives: 0
False Negatives: 0
False Positives: 0
Class 3:
True Positives: 0
False Negatives: 0
False Positives: 0
Evaluation/labels/00838.txt
Evaluation/labels/00838.txt
Class 0:
True Positives: 1
False Negatives: 1
False Positives: 0
Class 1:
True Positives: 0
False Negatives: 0
False Positives: 0
Class 2:
True Positives: 0
False Negatives: 0
False Positives: 0
Class 3:
True Positives: 0
False Negatives: 0
False Positives: 0
Evaluation/labels/00874.txt
Evaluation/labels/00874.txt
Class 0:
True Positives: 0
False Negatives: 0
False Positives: 0
Class 1:
True Positives: 1
False Negatives: 0
False Positives: 0
Class 2:
True Positives: 0
False Negatives: 0
False Positives: 0
Class 3:
True Positives: 0
False Negatives: 0
False Positives: 0
Evaluation/labels/00852.txt
Evaluation/labels/00852.txt
Class 0:
True Positives: 0
False Negatives: 0
False Positives: 0
Class 1:
True Positives: 1
False Negatives: 0
False Positives: 0
Class 2:
True Positives: 0
False Negatives: 0
False Positives: 0
Class 3:
True Positives: 0
False Negatives: 0
False Positives: 0
Evaluation/labels/00833.txt
Evaluation/labels/00833.txt
Class 0:
True Positives: 1
False Negatives: 0
False Positives: 0
Class 1:
True Positives: 0
False Negatives: 0
False Positives: 0
Class 2:
True Positives: 0
False Negatives: 0
False Positives: 0
Class 3:
True Positives: 0
False Negatives: 0
False Positives: 0
Evaluation/labels/00881.txt
Evaluation/labels/00881.txt
Class 0:
True Positives: 1
False Negatives: 1
False Positives: 0
Class 1:
True Positives: 0
False Negatives: 0
False Positives: 0
Class 2:
True Positives: 0
False Negatives: 0
False Positives: 0
Class 3:
True Positives: 0
False Negatives: 0
False Positives: 0
{'image_name': 'Total', 'Class-id': 'All Images', 'TP': 56, 'FP': 1, 'FN': 40, 'Precision': 0.9824561403508771, 'Recall': 0.5833333333333334, 'F1': 0.7320261437908496}

# My Analysis
I have used COCO-Detection/retinanet_R_101_FPN_3x.yaml model for this repository. A large collection of baseline models trained with detectron2 in Sep-Oct, 2019 are also provided inside detectron2 official directory. All numbers were obtained on Big Basin servers with 8 NVIDIA V100 GPUs & NVLink. The speed numbers are periodically updated with latest PyTorch/CUDA/cuDNN versions. You can access these models from code using detectron2.model_zoo API.

Retinanet model architechture:

Model:
RetinaNet(
  (backbone): FPN(

    )
    (bottom_up): ResNet(
    .....     
  )
  (head): RetinaNetHead(
    (cls_subnet): Sequential(
      (0): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (1): ReLU()
      (2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (3): ReLU()
      (4): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (5): ReLU()
      (6): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (7): ReLU()
    )
    (bbox_subnet): Sequential(
      (0): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (1): ReLU()
      (2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (3): ReLU()
      (4): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (5): ReLU()
      (6): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (7): ReLU()
    )
    (cls_score): Conv2d(256, 720, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (bbox_pred): Conv2d(256, 36, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  )
  (anchor_generator): DefaultAnchorGenerator(
    (cell_anchors): BufferList()
  )
)
