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
import keras_frcnn.roi_helpers as roi_helpers
from keras.utils import generic_utils, plot_model

sys.setrecursionlimit(40000)

parser = OptionParser()

parser.add_option("-p", "--path", dest="train_path", help="Path to training data.")
parser.add_option("-o", "--parser", dest="parser", help="Parser to use. One of simple or pascal_voc", default="simple")
parser.add_option("-n", "--num_rois", type="int", dest="num_rois", help="Number of RoIs to process at once.", default=15)
parser.add_option("--network", dest="network", help="Base network to use. Supports vgg or resnet50.", default='resnet50')
parser.add_option("--translate", dest="translate", help="Augment with random translation by 200px in x and y direction in training. (Default=True).", action="store_true", default=True)
parser.add_option("--hf", dest="horizontal_flips", help="Augment with horizontal flips in training. (Default=true).", action="store_false", default=True)
parser.add_option("--vf", dest="vertical_flips", help="Augment with vertical flips in training. (Default=false).", action="store_true", default=True)
parser.add_option("--rot", "--rot_90", dest="rot_90", help="Augment with 90 degree rotations in training. (Default=false).", action="store_true", default=True)
parser.add_option("--num_epochs", type="int", dest="num_epochs", help="Number of epochs.", default=20)
parser.add_option("--config_filename", dest="config_filename", help="Location to store all the metadata related to the training (to be used when testing).", default="training_configs/frcnn.pickle")
parser.add_option("--rpn", dest="rpn_weight_path", help="Input path for rpn.")
parser.add_option("--opt", dest="optimizers", help="set the optimizer to use", default="Adam")
parser.add_option("--elen", dest="epoch_length", help="set the epoch length. def=1000", default=5150)
parser.add_option("--load", dest="load", help="What model to load", default=None)
parser.add_option("--lr", dest="lr", help="learn rate", type=float, default=1e-3)
parser.add_option("--val", dest="validation_steps", help="Number of validation steps to perform in each epoch", type="int", default=858)

(options, args) = parser.parse_args()

if not options.train_path:   # if filename is not given
    parser.error('Error: path to training data must be specified. Pass --path to command line')
if options.parser == 'pascal_voc':
    from keras_frcnn.pascal_voc_parser import get_data
elif options.parser == 'simple':
    from keras_frcnn.simple_parser import get_data
else:
    raise ValueError("Command line option parser must be one of 'pascal_voc' or 'simple'")

# pass the settings from the command line, and persist them in the config object
C = config.Config()


C.translate = bool(options.translate)
C.use_horizontal_flips = bool(options.horizontal_flips)
C.use_vertical_flips = bool(options.vertical_flips)
C.rot_90 = bool(options.rot_90)

# mkdir to save models.
if not os.path.isdir("models"):
  os.mkdir("models")
if not os.path.isdir("models/frcnn"):
  os.mkdir("models/frcnn")
C.model_path = os.path.join("models/frcnn", 'frcnn.' + options.network)
C.num_rois = int(options.num_rois)

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

data_gen_train = data_generators.get_anchor_gt(train_imgs, classes_count, C, nn.get_img_output_length, K.common.image_dim_ordering(), mode='train')
data_gen_val = data_generators.get_anchor_gt(val_imgs, classes_count, C, nn.get_img_output_length,K.common.image_dim_ordering(), mode='val')

if K.common.image_dim_ordering() == 'th':
    input_shape_img = (3, None, None)
else:
    input_shape_img = (None, None, 3)

img_input = Input(shape=input_shape_img)
roi_input = Input(shape=(None, 4))

# define the base network (resnet here, can be VGG, Inception, etc)
shared_layers = nn.nn_base(img_input, trainable=True)

# define the RPN, built on the base layers
num_anchors = len(C.anchor_box_scales) * len(C.anchor_box_ratios)
rpn = nn.rpn(shared_layers, num_anchors)

classifier = nn.classifier(shared_layers, roi_input, C.num_rois, nb_classes=len(classes_count), trainable=True)

model_rpn = Model(img_input, rpn[:2])
model_classifier = Model([img_input, roi_input], classifier)

# this is a model that holds both the RPN and the classifier, used to load/save weights for the models
model_all = Model([img_input, roi_input], rpn[:2] + classifier)

# load pretrained weights
try:
    print('loading weights from {}'.format(C.base_net_weights))
    model_rpn.load_weights(C.base_net_weights, by_name=True)
    model_classifier.load_weights(C.base_net_weights, by_name=True)
except:
    print('Could not load pretrained model weights. Weights can be found in the keras application folder \
		https://github.com/fchollet/keras/tree/master/keras/applications')

# optimizer setup
if options.optimizers == "SGD":
    if options.rpn_weight_path is not None:
        optimizer = SGD(lr=options.lr/100, decay=0.0005, momentum=0.9)
        optimizer_classifier = SGD(lr=options.lr/5, decay=0.0005, momentum=0.9)
    else:
        optimizer = SGD(lr=options.lr/10, decay=0.0005, momentum=0.9)
        optimizer_classifier = SGD(lr=options.lr/10, decay=0.0005, momentum=0.9)
else:
    optimizer = Adam(lr=options.lr, clipnorm=0.001)
    optimizer_classifier = Adam(lr=options.lr, clipnorm=0.001)

# may use this to resume from rpn models or previous training. specify either rpn or frcnn model to load
if options.load is not None:
    print("loading previous model from ", options.load)
    model_rpn.load_weights(options.load, by_name=True)
    model_classifier.load_weights(options.load, by_name=True)
elif options.rpn_weight_path is not None:
    print("loading RPN weights from ", options.rpn_weight_path)
    model_rpn.load_weights(options.rpn_weight_path, by_name=True)
else:
    print("no previous model was loaded")

# compile the model AFTER loading weights!
model_rpn.compile(optimizer=optimizer, loss=[loss_func.rpn_loss_cls(num_anchors), loss_func.rpn_loss_regr(num_anchors)])
model_classifier.compile(optimizer=optimizer_classifier, loss=[loss_func.class_loss_cls, loss_func.class_loss_regr(len(classes_count)-1)], metrics={'dense_class_{}'.format(len(classes_count)): 'accuracy'})
model_all.compile(optimizer='Adam', loss='mae')

epoch_length = int(options.epoch_length)
num_epochs = int(options.num_epochs)
iter_num = 0

losses = np.zeros((epoch_length, 5))
rpn_accuracy_rpn_monitor = []
rpn_accuracy_for_epoch = []
start_time = time.time()

best_loss = np.Inf

class_mapping_inv = {v: k for k, v in class_mapping.items()}
print('Starting training')

vis = True

with open('log/frcnn/frcnn_training.csv', 'w') as log:
	log_writer = csv.writer(log, delimiter=';')
	log_writer.writerow(['mean_overlapping_bboxes', 'class_acc', 'loss_rpn_cls', 'loss_rpn_regr', 'loss_class_cls', 'loss_class_regr', 'curr_loss', 'val_loss_class_cls', 'val_loss_class_regr', 'val_accuracy', 'curr_val_loss'])
for epoch_num in range(num_epochs):
	progbar = generic_utils.Progbar(epoch_length)
	print('Epoch {}/{}'.format(epoch_num + 1, num_epochs))
	while True:
		if len(rpn_accuracy_rpn_monitor) == epoch_length and C.verbose:
			mean_overlapping_bboxes = float(sum(rpn_accuracy_rpn_monitor))/len(rpn_accuracy_rpn_monitor)
			rpn_accuracy_rpn_monitor = []
			print('Average number of overlapping bounding boxes from RPN = {} for {} previous iterations'.format(mean_overlapping_bboxes, epoch_length))
			if mean_overlapping_bboxes == 0:
				print('RPN is not producing bounding boxes that overlap the ground truth boxes. Check RPN settings or keep training.')
		X, Y, img_data = next(data_gen_train)

		loss_rpn = model_rpn.train_on_batch(X, Y)

		P_rpn = model_rpn.predict_on_batch(X)
		R, prob = roi_helpers.rpn_to_roi(P_rpn[0], P_rpn[1], C, K.common.image_dim_ordering(), use_regr=True, overlap_thresh=1e-12, max_boxes=300)
		# note: calc_iou converts from (x1,y1,x2,y2) to (x,y,w,h) format
		X2, Y1, Y2 = roi_helpers.calc_iou(R, img_data, C, class_mapping)

		if X2 is None:
			rpn_accuracy_rpn_monitor.append(0)
			rpn_accuracy_for_epoch.append(0)
			continue

		neg_samples = np.where(Y1[0, :, -1] == 1)
		pos_samples = np.where(Y1[0, :, -1] == 0)

		if len(neg_samples) > 0:
			neg_samples = neg_samples[0]
		else:
			neg_samples = []

		if len(pos_samples) > 0:
			pos_samples = pos_samples[0]
		else:
			pos_samples = []

		if len(pos_samples) < C.num_rois//2:
			selected_pos_samples = pos_samples.tolist()
		else:
			selected_pos_samples = np.random.choice(pos_samples, C.num_rois//2, replace=False).tolist()

		# select negative regions with high object score for hard negative mining
		selected_neg_samples = neg_samples[:5].tolist()

		selected_neg_samples.extend(np.random.choice(neg_samples[5:], C.num_rois - len(selected_pos_samples) - len(selected_neg_samples), replace=False).tolist())

		sel_samples = selected_pos_samples + selected_neg_samples

		rpn_accuracy_rpn_monitor.append(len(pos_samples))
		rpn_accuracy_for_epoch.append((len(pos_samples)))

		loss_class = model_classifier.train_on_batch([X, X2[:, sel_samples, :]], [Y1[:, sel_samples, :], Y2[:, sel_samples, :]])

		losses[iter_num, 0] = loss_rpn[1]
		losses[iter_num, 1] = loss_rpn[2]

		losses[iter_num, 2] = loss_class[1]
		losses[iter_num, 3] = loss_class[2]
		losses[iter_num, 4] = loss_class[3]

		iter_num += 1

		progbar.update(iter_num, [('rpn_cls', np.mean(losses[:iter_num, 0])), ('rpn_regr', np.mean(losses[:iter_num, 1])),
								  ('detector_cls', np.mean(losses[:iter_num, 2])), ('detector_regr', np.mean(losses[:iter_num, 3])),
								 ("average number of objects", len(selected_pos_samples))])

		if iter_num == epoch_length:
			loss_rpn_cls = np.mean(losses[:, 0])
			loss_rpn_regr = np.mean(losses[:, 1])
			loss_class_cls = np.mean(losses[:, 2])
			loss_class_regr = np.mean(losses[:, 3])
			class_acc = np.mean(losses[:, 4])

			mean_overlapping_bboxes = float(sum(rpn_accuracy_for_epoch)) / len(rpn_accuracy_for_epoch)
			rpn_accuracy_for_epoch = []

			print('Starting validation...')
			val_losses = np.zeros((options.validation_steps, 3))

			for val_step in range(options.validation_steps):
				X_val, Y_val, img_data = next(data_gen_val)
				P_rpn = model_rpn.predict_on_batch(X_val)

				R, prob = roi_helpers.rpn_to_roi(P_rpn[0], P_rpn[1], C, K.common.image_dim_ordering(), use_regr=True, overlap_thresh=1e-12, max_boxes=300)
				# note: calc_iou converts from (x1,y1,x2,y2) to (x,y,w,h) format
				X2, Y1, Y2 = roi_helpers.calc_iou(R, img_data, C, class_mapping)

				if X2 is None:
					continue

				neg_samples = np.where(Y1[0, :, -1] == 1)
				pos_samples = np.where(Y1[0, :, -1] == 0)

				if len(neg_samples) > 0:
					neg_samples = neg_samples[0]
				else:
					neg_samples = []

				if len(pos_samples) > 0:
					pos_samples = pos_samples[0]
				else:
					pos_samples = []

				if len(pos_samples) < C.num_rois//2:
					selected_pos_samples = pos_samples.tolist()
				else:
					selected_pos_samples = np.random.choice(pos_samples, C.num_rois//2, replace=False).tolist()

					# select negative regions with high object score for hard negative mining
				selected_neg_samples = neg_samples[:5].tolist()

				selected_neg_samples.extend(np.random.choice(neg_samples[5:], C.num_rois - len(selected_pos_samples) - len(selected_neg_samples), replace=False).tolist())

				sel_samples = selected_pos_samples + selected_neg_samples

				val_loss = model_classifier.test_on_batch([X_val, X2[:, sel_samples, :]], [Y1[:, sel_samples, :], Y2[:, sel_samples, :]])

				val_losses[val_step, 0] = val_loss[1]
				val_losses[val_step, 1] = val_loss[2]
				val_losses[val_step, 2] = val_loss[3]

			val_loss_cls = np.mean(val_losses[:, 0])
			val_loss_regr = np.mean(val_losses[:, 1])
			val_acc = np.mean(val_losses[:, 2])

			print('... Validation done.')

			if C.verbose:
				print(f'Mean number of bounding boxes from RPN overlapping ground truth boxes: {mean_overlapping_bboxes}')
				print(f'Classifier accuracy for bounding boxes from RPN: {class_acc}')
				print(f'Loss RPN classifier: {loss_rpn_cls}')
				print(f'Loss RPN regression: {loss_rpn_regr}')
				print(f'Loss Detector classifier: {loss_class_cls}')
				print(f'Loss Detector regression: {loss_class_regr}')
				print(f'Val Loss Detector classifier: {val_loss_cls}')
				print(f'Val Loss Detector regression: {val_loss_regr}')
				print(f'Val accuracy: {val_acc}')
				print(f'Elapsed time: {time.time() - start_time}')

			curr_loss = loss_class_cls + loss_class_regr
			curr_val_loss = val_loss_cls + val_loss_regr
			iter_num = 0
			start_time = time.time()

			with open('log/frcnn/frcnn_training.csv', 'a') as log:
				log_writer = csv.writer(log, delimiter=';')
				log_writer.writerow([mean_overlapping_bboxes, class_acc, loss_rpn_cls, loss_rpn_regr, loss_class_cls, loss_class_regr, curr_loss, val_loss_cls, val_loss_regr, val_acc, curr_val_loss])

			if curr_loss < best_loss:
				if C.verbose:
					print('Total loss decreased from {} to {}, saving weights'.format(best_loss,curr_loss))
				best_loss = curr_loss
				model_all.save_weights(f'models/frcnn/frcnn.{options.network}.weights.{epoch_num}-{round(curr_loss, 2)}-{round(curr_val_loss, 2)}.hdf5')

			break

print('Training complete, exiting.')
