![image](https://github.com/Heyjuke58/frcnn-wind-turbine-detection/blob/master/images/example_res.jpg)
Example from test set. Blue: Ground truth boxes, red: predictions
# faster-RCNN implementation for detection of wind turbines from aerial images
Forked from [here](https://github.com/kentaroy47/frcnn-from-scratch-with-keras).

Thanks go out to [kentaroy47](https://github.com/kentaroy47)!

If you understand german, you can consult my bachelor thesis for context and details, why and how this was implemented. I added a pdf to this repo.

## Frameworks
I updated support of tensorflow 2, originally was:

Tested with Tensorflow==1.12.0 and Keras 2.2.4.

## How to run

Prerequisites: Python 3.6, pip > 20, virtualenv

#### Setup virtual environment

Windows
```
python -m virtualenv env

.\env\Scripts\activate

pip install -r requirements.txt
```
Linux
```
virtualenv env

source ./env/bin/activate

pip install -r requirements.txt

```

### Prepare Annotation File
Due to data property reasons only a few example images are annotated in the annotation csv ``set_splits/bboxes_example.csv``.\
One line in the annotation csv holds the information about one turbine in the image.\
An example line: ``22084_7.227_49.476_2011-12-01.jpg,999,64,1089,154,turbine,large,176.0,train``\
The format is: ``image_filename,x1,y1,x2,y2,class,size_category,size,set``\
The coordinates are the top left and the bottom right corner.


#### Train RPN alone and with Classifier
```
python train_rpn.py -p set_splits/bboxes_example.csv
```
```
python train_frcnn.py -p set_splits/bboxes_example.csv -rpn models/rpn/rpn_model.hdf5
```
Hyperparameters can be changed in the config file ``keras_frcnn/config.py``, as well as with other options (s. `train_rpn.py` and `train_frcnn.py` for the documentation)

### Test RPN alone and with Classifier
```
python test_rpn.py -p set_splits/bboxes_example.csv --write --load models/rpn/rpn_model.hdf5
```
```
python test_frcnn.py -p set_splits/bboxes_example.csv --write --load models/frcnn/frcnn_model.hdf5
```
The option --write saves the tested pictures with predictions and ground truth boxes in the folder results.\
The option --load loads the model to test.



