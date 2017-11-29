# finetune_op_vgg

## Prerequired:
- tensorflow
- keras
- numpy
- scipy
- opencv

## Fintune vvg layer's
### Step 1:
- run `python Step1TrainClassifier.py`. Default: A `step1.h5` file will be created to store weights of fine-tuned model.
### Step 2:
- run `python Step2TrainFeatureExtractor.py`. Default: A `step2.h5` file will be created to store weights of fine-tuned model.
### Test fine-tuned model:
- run `test_finetuned_model.py`. Default: Input images are from `images` folder. Output images will be saved in `result` folder.
