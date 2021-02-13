from __future__ import division
import cv2
import csv
import numpy as np
import sys
import pickle
from optparse import OptionParser
from keras_frcnn import config
from keras import backend as K
from keras.layers import Input
from keras.models import Model
from keras_frcnn import losses as loss_func
from keras_frcnn import data_generators
from keras.optimizers import Adam

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

sys.setrecursionlimit(40000)

parser = OptionParser()

parser.add_option("-p", "--path", dest="test_path", help="Path to test data.")
parser.add_option("-n", "--num_rois", type="int", dest="num_rois", help="Number of ROIs per iteration. Higher means more memory use.", default=30)
parser.add_option("--config_filename", dest="config_filename", help="Location to read the metadata related to the training (generated when training).", default="training_configs/config_train_batched_rpn.pickle")
parser.add_option("--network", dest="network", help="Base network to use. Supports vgg or resnet50.", default='resnet50')
parser.add_option("--load", dest="load", help="specify model path.", default='models/rpn_lr/rpn.resnet50.weights.28-0.62-1.66.hdf5')
(options, args) = parser.parse_args()

if not options.test_path:   # if filename is not given
	parser.error('Error: path to test data must be specified. Pass --path to command line')


config_output_filename = options.config_filename

with open(config_output_filename, 'rb') as f_in:
	C = pickle.load(f_in)

# we will use resnet. may change to vgg
if options.network == 'vgg':
	C.network = 'vgg16'
	from keras_frcnn import vgg as nn
elif options.network == 'resnet50':
	from keras_frcnn import resnet as nn
	C.network = 'resnet50'
elif options.network == 'vgg19':
	from keras_frcnn import vgg19 as nn
	C.network = 'vgg19'
elif options.network == 'mobilenetv1':
	from keras_frcnn import mobilenetv1 as nn
	C.network = 'mobilenetv1'
#	from keras.applications.mobilenet import preprocess_input
elif options.network == 'mobilenetv1_05':
	from keras_frcnn import mobilenetv1_05 as nn
	C.network = 'mobilenetv1_05'
#	from keras.applications.mobilenet import preprocess_input
elif options.network == 'mobilenetv1_25':
	from keras_frcnn import mobilenetv1_25 as nn
	C.network = 'mobilenetv1_25'
#	from keras.applications.mobilenet import preprocess_input
elif options.network == 'mobilenetv2':
	from keras_frcnn import mobilenetv2 as nn
	C.network = 'mobilenetv2'
else:
	print('Not a valid model')
	raise ValueError

# turn off any data augmentation at test time
C.use_horizontal_flips = False
C.use_vertical_flips = False
C.rot_90 = False

def format_img_size(img, C):
	""" formats the image size based on config """
	img_min_side = float(C.im_size)
	(height,width,_) = img.shape
		
	if width <= height:
		ratio = img_min_side/width
		new_height = int(ratio * height)
		new_width = int(img_min_side)
	else:
		ratio = img_min_side/height
		new_width = int(ratio * width)
		new_height = int(img_min_side)
	img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
	return img, ratio	

def format_img_channels(img, C):
	""" formats the image channels based on config """
	img = img[:, :, (2, 1, 0)] # BGR -> RGB
	img = img.astype(np.float32)
	img /= 255.
	# img[:, :, 0] -= C.img_channel_mean[0]
	# img[:, :, 1] -= C.img_channel_mean[1]
	# img[:, :, 2] -= C.img_channel_mean[2]
	#
	# img[:,:,0] /= C.img_channel_std[0]
	# img[:,:,1] /= C.img_channel_std[1]
	# img[:,:,2] /= C.img_channel_std[2]

	img = np.transpose(img, (2, 0, 1))
	img = np.expand_dims(img, axis=0)
	return img

def format_img(img, C):
	""" formats an image for model prediction based on config """
	img, ratio = format_img_size(img, C)
	img = format_img_channels(img, C)
	return img, ratio


# Method to transform the coordinates of the bounding box to its original size
def get_real_coordinates(ratio, x1, y1, x2, y2):

	real_x1 = int(round(x1 // ratio))
	real_y1 = int(round(y1 // ratio))
	real_x2 = int(round(x2 // ratio))
	real_y2 = int(round(y2 // ratio))

	return (real_x1, real_y1, real_x2 ,real_y2)

class_mapping = C.class_mapping

if 'bg' not in class_mapping:
	class_mapping['bg'] = len(class_mapping)

class_mapping = {v: k for k, v in class_mapping.items()}
print(class_mapping)
class_to_color = {class_mapping[v]: np.random.randint(0, 255, 3) for v in class_mapping}
C.num_rois = int(options.num_rois)

if C.network == 'resnet50':
	num_features = 1024
elif C.network =="mobilenetv2":
	num_features = 320
else:
	# may need to fix this up with your backbone..!
	print("backbone is not resnet50. number of features chosen is 512")
	num_features = 512

if K.common.image_dim_ordering() == 'th':
	input_shape_img = (3, None, None)
	input_shape_features = (num_features, None, None)
else:
	input_shape_img = (None, None, 3)
	input_shape_features = (None, None, num_features)


img_input = Input(shape=input_shape_img)
roi_input = Input(shape=(C.num_rois, 4))
feature_map_input = Input(shape=input_shape_features)

# define the base network (resnet here, can be VGG, Inception, etc)
shared_layers = nn.nn_base(img_input)

# define the RPN, built on the base layers
num_anchors = len(C.anchor_box_scales) * len(C.anchor_box_ratios)
rpn_layers = nn.rpn(shared_layers, num_anchors)

#classifier = nn.classifier(feature_map_input, roi_input, C.num_rois, nb_classes=len(class_mapping))

model_rpn = Model(img_input, rpn_layers[:2])
#model_classifier = Model([feature_map_input, roi_input], classifier)

# model loading
C.model_path = options.load
print('Loading weights from {}'.format(C.model_path))
model_rpn.load_weights(C.model_path, by_name=True)

optimizer = Adam(lr=1e-3, clipnorm=0.001)
model_rpn.compile(optimizer=optimizer, loss=[loss_func.rpn_loss_cls(num_anchors), loss_func.rpn_loss_regr(num_anchors)])

#### load images here ####
from keras_frcnn.simple_parser import get_data
all_imgs, classes_count, class_mapping = get_data(options.test_path, test_only=False)

train_imgs = [s for s in all_imgs if s['imageset'] == 'train']
data_gen_val = data_generators.get_anchor_gt(train_imgs, classes_count, C, nn.get_img_output_length,K.common.image_dim_ordering(), mode='val')

with open('log/losses.csv', 'w') as log:
	log_writer = csv.writer(log, delimiter=';')
	log_writer.writerow(['loss', 'loss_rpn_cls', 'loss_rpn_regr', 'img'])
	for img in train_imgs:
		X_gen, Y_gen, imgdata = next(data_gen_val)

		loss = model_rpn.test_on_batch(X_gen, Y_gen)

		log_writer.writerow([loss[0], loss[1], loss[2], imgdata['filepath']])
		print(f'Row written with loss {loss[0]} for image {imgdata["filepath"]}')