from __future__ import division
import random
import pprint
import sys
import time
import numpy as np
from optparse import OptionParser
import pickle
import os
import csv
# gpu setting
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import tensorflow as tf
config2 = tf.compat.v1.ConfigProto()
config2.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config2)

from keras import backend as K
from keras.optimizers import Adam, SGD, RMSprop
from keras.layers import Input
from keras.models import Model, load_model
from keras_frcnn import config, data_generators
from keras_frcnn import losses as loss_func
from keras_frcnn.simple_parser import get_data
from keras.utils import generic_utils, plot_model

sys.setrecursionlimit(40000)

parser = OptionParser()

parser.add_option("-p", "--path", dest="train_path", help="Path to training data.")
parser.add_option("--network", dest="network", help="Base network to use. Supports vgg or resnet50.", default='resnet50')

parser.add_option("--num_epochs", type="int", dest="num_epochs", help="Number of epochs.", default=30)
parser.add_option("--elen", dest="epoch_length", help="set the epoch length. def=17824", default=17824)
parser.add_option("--opt", dest="optimizers", help="set the optimizer to use", default="Adam")
parser.add_option("--lr", dest="lr", help="learn rate", type=float, default=1e-3)
parser.add_option("--val", dest="validation_steps", help="Number of validation steps to perform in each epoch", type="int", default=2971)

parser.add_option("--config_filename", dest="config_filename", help="Location to store all the metadata related to the training (to be used when testing).", default="training_configs/rpn.pickle")
parser.add_option("--load", dest="load", help="What model to load", default=None)

parser.add_option("--translate", dest="translate", help="Augment with random translation by 200px in x and y direction in training. (Default=True).", action="store_true", default=True)
parser.add_option("--hf", dest="horizontal_flips", help="Augment with horizontal flips in training. (Default=true).", action="store_false", default=True)
parser.add_option("--vf", dest="vertical_flips", help="Augment with vertical flips in training. (Default=false).", action="store_true", default=True)
parser.add_option("--rot", "--rot_90", dest="rot_90", help="Augment with 90 degree rotations in training. (Default=false).", action="store_true", default=True)

(options, args) = parser.parse_args()

if not options.train_path:   # if filename is not given
	parser.error('Error: path to training data must be specified. Pass --path to command line')

# pass the settings from the command line, and persist them in the config object
C = config.Config()


C.translate = bool(options.translate)
C.use_horizontal_flips = bool(options.horizontal_flips)
C.use_vertical_flips = bool(options.vertical_flips)
C.rot_90 = bool(options.rot_90)

# we will use resnet. may change to others
if options.network == 'vgg' or options.network == 'vgg16':
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
elif options.network == 'mobilenetv2':
	from keras_frcnn import mobilenetv2 as nn
	C.network = 'mobilenetv2'
elif options.network == 'densenet':
	from keras_frcnn import densenet as nn
	C.network = 'densenet'
else:
	print('Not a valid model')
	raise ValueError

# set the path to weights based on backend and model
C.base_net_weights = nn.get_weight_path()

all_imgs, classes_count, class_mapping = get_data(options.train_path)

if 'bg' not in classes_count:
	classes_count['bg'] = 0
	class_mapping['bg'] = len(class_mapping)

C.class_mapping = class_mapping

inv_map = {v: k for k, v in class_mapping.items()}

print('Training images per class:')
pprint.pprint(classes_count)
print('Num classes (including bg) = {}'.format(len(classes_count)))

config_output_filename = options.config_filename

with open(config_output_filename, 'wb') as config_f:
	pickle.dump(C,config_f)
	print('Config has been written to {}, and can be loaded when testing to ensure correct results'.format(config_output_filename))

random.shuffle(all_imgs)

num_imgs = len(all_imgs)

train_imgs = [s for s in all_imgs if s['imageset'] == 'train']
val_imgs = [s for s in all_imgs if s['imageset'] == 'val']

print('Num train samples {}'.format(len(train_imgs)))
print('Num val samples {}'.format(len(val_imgs)))

data_gen_train = data_generators.get_anchor_gt(train_imgs, classes_count, C, nn.get_img_output_length, K.common.image_dim_ordering(), mode='train', hist_equal=True)
data_gen_val = data_generators.get_anchor_gt(val_imgs, classes_count, C, nn.get_img_output_length,K.common.image_dim_ordering(), mode='val', hist_equal=True)

if K.common.image_dim_ordering() == 'th':
	input_shape_img = (3, 800, 800)
else:
	input_shape_img = (800, 800, 3)

img_input = Input(shape=input_shape_img)

# define the RPN, built on the base layers
num_anchors = len(C.anchor_box_scales) * len(C.anchor_box_ratios)
rpn = nn.rpn(nn.nn_base(img_input, trainable=True), num_anchors)

model_rpn = Model(img_input, rpn[:2])

# load pretrained weights
try:
	if options.load is None:
		print('loading weights from {}'.format(C.base_net_weights))
		model_rpn.load_weights(C.base_net_weights, by_name=True)
except:
	print('Could not load pretrained model weights. Weights can be found in the keras application folder https://github.com/fchollet/keras/tree/master/keras/applications')

# optimizer setup
if options.optimizers == "SGD":
	optimizer = SGD(lr=options.lr/100, decay=0.0005, momentum=0.9)
else:
	optimizer = Adam(lr=options.lr, clipnorm=0.001)

# may use this to resume from rpn models or previous training. specify either rpn or frcnn model to load
if options.load is not None:
	print("loading previous model from ", options.load)
	model_rpn.load_weights(options.load, by_name=True)

# compile the model AFTER loading weights!
model_rpn.compile(optimizer=optimizer, loss=[loss_func.rpn_loss_cls(num_anchors), loss_func.rpn_loss_regr(num_anchors)])

epoch_length = int(options.epoch_length)
num_epochs = int(options.num_epochs)
iter_num = 0

losses = np.zeros((epoch_length, 2))
start_time = time.time()

best_loss = np.Inf

class_mapping_inv = {v: k for k, v in class_mapping.items()}
print('Starting training')

with open('log/rpn/rpn_training.csv', 'w') as log:
	log_writer = csv.writer(log, delimiter=';')
	log_writer.writerow(['curr_loss', 'loss_rpn_cls', 'loss_rpn_regr', 'curr_val_loss', 'val_loss_rpn_cls', 'val_loss_rpn_regr'])
for epoch_num in range(num_epochs):
	progbar = generic_utils.Progbar(epoch_length)
	print('Epoch {}/{}'.format(epoch_num + 1, num_epochs))

	while True:

		X, Y, img_data = next(data_gen_train)
		loss_rpn = model_rpn.train_on_batch(X, Y)

		losses[iter_num, 0] = loss_rpn[1]
		losses[iter_num, 1] = loss_rpn[2]

		iter_num += 1

		progbar.update(iter_num, [('rpn_cls', np.mean(losses[:iter_num, 0])), ('rpn_regr', np.mean(losses[:iter_num, 1]))])

		if iter_num == epoch_length:
			loss_rpn_cls = np.mean(losses[:, 0])
			loss_rpn_regr = np.mean(losses[:, 1])

			print('Starting validation...')
			val_losses = np.zeros((options.validation_steps, 2))

			for val_step in range(options.validation_steps):
				X_val, Y_val, img_data = next(data_gen_val)
				val_loss = model_rpn.test_on_batch(X_val, Y_val)

				val_losses[val_step, 0] = val_loss[1]
				val_losses[val_step, 1] = val_loss[2]

			val_loss_cls = np.mean(val_losses[:, 0])
			val_loss_regr = np.mean(val_losses[:, 1])

			print('... Validation done.')

			if C.verbose:
				print(f'Loss RPN classifier: {loss_rpn_cls}')
				print(f'Loss RPN regression: {loss_rpn_regr}')
				print(f'Validation Loss RPN classifier: {val_loss_cls}')
				print(f'Validation Loss RPN regression: {val_loss_regr}')
				print(f'Elapsed time: {time.time() - start_time}')

			curr_loss = loss_rpn_cls + loss_rpn_regr
			curr_val_loss = val_loss_cls + val_loss_regr
			iter_num = 0
			start_time = time.time()

			with open('log/rpn/rpn_training.csv', 'a') as log:
				log_writer = csv.writer(log, delimiter=';')
				log_writer.writerow([curr_loss, loss_rpn_cls, loss_rpn_regr, curr_val_loss, val_loss_cls, val_loss_regr])

			if curr_loss < best_loss:
				if C.verbose:
					print('Total loss decreased from {} to {}, saving weights'.format(best_loss,curr_loss))
				best_loss = curr_loss
				model_rpn.save_weights(f'./models/rpn/rpn.{options.network}.weights.{epoch_num}-{round(curr_loss, 2)}-{round(curr_val_loss, 2)}.hdf5')

			break

print('Training complete, exiting.')

session.close()