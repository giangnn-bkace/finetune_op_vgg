# finetune_op_vgg

## Prerequired:
- tensorflow
- keras
- numpy
- scipy
- opencv

## Fintune vvg layer's
- Download original openpose model at https://www.dropbox.com/s/llpxd14is7gyj0z/model.h5
- Put downloaded [model.h5](https://www.dropbox.com/s/llpxd14is7gyj0z/model.h5) in this project folder (`finetune_op_vgg` folder)
### Step 1:
- run `python Step1TrainClassifier.py`. 
>**Default:** A `step1.h5` file will be created to store weights of fine-tuned model.
### Step 2:
- run `python Step2TrainFeatureExtractor.py`. 
>**Default:** A `step2.h5` file will be created to store weights of fine-tuned model.<br />
To lock a layer, add it name to `lock` list in `Step2TrainFeatureExtractor.py`.
![Lock layers](https://github.com/giangnn-bkace/finetune_op_vgg/blob/master/LockLayers.JPG "Lock layers")
### Test fine-tuned model:
- run `test_finetuned_model.py`. 
>**Default:** Input images are from `images` folder. Output images will be saved in `result` folder.
