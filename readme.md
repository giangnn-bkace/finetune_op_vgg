# finetune_op_vgg
## Introduction
![What is doing?](https://github.com/giangnn-bkace/finetune_op_vgg/blob/master/images/Finetune.JPG)
We attempt to fine-tune open-pose model (for human) for chimpanzee pose estimation.

### An example of apply the current OpenPose model for human pose, face, hand recognition
![pose recognition](https://github.com/giangnn-bkace/finetune_op_vgg/blob/master/images/pose-example1.gif)

### NOT good for chimpanzee
![chim](https://github.com/giangnn-bkace/finetune_op_vgg/blob/master/images/direct.jpg)

### Objective
![Objective](https://github.com/giangnn-bkace/finetune_op_vgg/blob/master/images/objective.JPG)

## Packages:
- tensorflow
- keras
- numpy
- scipy
- opencv

## OpenPose model architecture
![OpenPose architecture](https://github.com/giangnn-bkace/finetune_op_vgg/blob/master/images/openpose_architecture.png)

## Fintune vvg layer's
![Finetune vgg](https://github.com/giangnn-bkace/finetune_op_vgg/blob/master/images/vgg.png)
- Download original openpose model at https://www.dropbox.com/s/llpxd14is7gyj0z/model.h5
- Put downloaded [model.h5](https://www.dropbox.com/s/llpxd14is7gyj0z/model.h5) in this project folder (`finetune_op_vgg` folder)
### Step 1:
Collect some human images and train a classifier layer for human/Not human image classification 
- run `python Step1TrainClassifier.py`. 
>**Default:** A `step1.h5` file will be created to store weights of fine-tuned model.
### Step 2:
Fine-tune some feature-extractor layers for chimpanzee/Not chimpanzee image classification
- run `python Step2TrainFeatureExtractor.py`. 
>**Default:** A `step2.h5` file will be created to store weights of fine-tuned model.
<br />To lock a layer, add it name to `lock` list in `Step2TrainFeatureExtractor.py`.
![Lock layers](https://github.com/giangnn-bkace/finetune_op_vgg/blob/master/LockLayers.JPG "Lock layers")
### Test fine-tuned model:
- run `test_finetuned_model.py`. 
>**Default:** Input images are from `images` folder. Output images will be saved in `result` folder.

## Results
![result1](https://github.com/giangnn-bkace/finetune_op_vgg/blob/master/images/fine-tune_result1.JPG)

![result2](https://github.com/giangnn-bkace/finetune_op_vgg/blob/master/images/fine-tune_result2.JPG)
