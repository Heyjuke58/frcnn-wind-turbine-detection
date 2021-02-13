from __future__ import division
import cv2
import numpy as np
import sys
import csv
import pickle
from optparse import OptionParser

from keras_frcnn import config
from keras import backend as K
from keras.layers import Input
from keras.models import Model
from keras_frcnn import roi_helpers
from keras.applications.mobilenet import preprocess_input
from keras_frcnn.data_generators import iou
from pyproj import Transformer
from geopy.distance import geodesic
from sklearn.metrics import average_precision_score, precision_recall_curve

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

sys.setrecursionlimit(40000)

parser = OptionParser()

parser.add_option("-p", "--path", dest="test_path", help="Path to test data.")
parser.add_option("-n", "--num_rois", type="int", dest="num_rois", help="Number of ROIs per iteration. Higher means more memory use.", default=30)
parser.add_option("--config_filename", dest="config_filename", help="Location to read the metadata related to the training (generated when training).", default="training_configs/rpn.pickle")
parser.add_option("--network", dest="network", help="Base network to use. Supports vgg or resnet50.", default='resnet50')
parser.add_option("--write", dest="write", help="to write out the image with detections or not.", action='store_true')
parser.add_option("--load", dest="load", help="specify model path.")
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

	img[:, :, 0] -= C.img_channel_mean[0]
	img[:, :, 1] -= C.img_channel_mean[1]
	img[:, :, 2] -= C.img_channel_mean[2]

	img[:,:,0] /= C.img_channel_std[0]
	img[:,:,1] /= C.img_channel_std[1]
	img[:,:,2] /= C.img_channel_std[2]

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

# define the base network (resnet here, can be VGG, Inception, etc)
shared_layers = nn.nn_base(img_input)

# define the RPN, built on the base layers
num_anchors = len(C.anchor_box_scales) * len(C.anchor_box_ratios)
rpn_layers = nn.rpn(shared_layers, num_anchors)

model_rpn = Model(img_input, rpn_layers)

# model loading
C.model_path = options.load
print('Loading weights from {}'.format(C.model_path))
model_rpn.load_weights(C.model_path, by_name=True)

#### load images here ####
from keras_frcnn.simple_parser import get_data
all_imgs, classes_count, class_mapping = get_data(options.test_path, test_only=True)

with open('log/rpn/rpn_results.csv', 'w') as result_csv:
	log_writer = csv.writer(result_csv, delimiter=';')
	log_writer.writerow(['confidence_threshold', 'iou_threshold', 'precision', 'recall', 'f1_score', 'avg_precision', 'mean_distance',
						 'precision_b_small', 'precision_b_medium', 'precision_b_large',
						 'recall_b_small', 'recall_b_medium', 'recall_b_large',
						 'mean_distance_b_small', 'mean_distance_b_medium', 'mean_distance_b_high'])

for confidence_threshold in [0.9, 0.999, 0.99999]:
	for iou_threshold_tp in [0.1, 0.2, 0.3, 0.4]:
		# Confusion Matrix initialization
		tp = 0
		fp = 0
		fn = 0

		# for the 3 buckets
		tp_b_small, tp_b_medium, tp_b_large = 0, 0, 0
		fn_b_small, fn_b_medium, fn_b_large = 0, 0, 0
		fp_b_small, fp_b_medium, fp_b_large = 0, 0, 0

		# false negatives with height
		fn_height = []

		# For average precision
		avg_prec = []

		# transformers for distance calculation in epsg 3857
		transformer = Transformer.from_proj('epsg:3857', 'epsg:4326', always_xy=True)
		transformer2 = Transformer.from_proj('epsg:4326', 'epsg:3857', always_xy=True)

		distance_sum = 0
		distance_norm = 0

		distance_sum_b_small, distance_sum_b_medium, distance_sum_b_large = 0, 0, 0
		distance_norm_b_small, distance_norm_b_medium, distance_norm_b_large = 0, 0, 0

		for img_data in all_imgs:
			lon, lat = [float(x) for x in img_data['filepath'].split('_')[1:3]]
			reference_point_4326 = (lon, lat)
			reference_point_3857 = transformer2.transform(reference_point_4326[0], reference_point_4326[1])

			img = cv2.imread(img_data['filepath'])
			# cropping image to remove google label
			img_cropped = img[0:1280-50, 0:1280-50]

			# preprocess image
			X, ratio = format_img(img_cropped, C)
			if K.common.image_dim_ordering() == 'tf':
				X = np.transpose(X, (0, 2, 3, 1))

			# get the feature maps and output from the RPN
			[Y1, Y2, F] = model_rpn.predict(X)

			R, prob = roi_helpers.rpn_to_roi(Y1, Y2, C, K.common.image_dim_ordering(), overlap_thresh=0.3)

			# Pad prob in case it has not 300 probabilities
			if prob.shape[0] != 300:
				prob = np.pad(prob, (0, 300 - prob.shape[0]), 'constant', constant_values=(0, 0))

			# convert from (x1,y1,x2,y2) to (x,y,w,h)
			R[:, 2] -= R[:, 0]
			R[:, 3] -= R[:, 1]

			bboxes = []
			probs = []
			for jk in range(R.shape[0]//C.num_rois + 1):
				ROIs = np.expand_dims(R[C.num_rois*jk:C.num_rois*(jk+1), :], axis=0)
				if ROIs.shape[1] == 0:
					break

				if jk == R.shape[0]//C.num_rois:
					#pad R
					curr_shape = ROIs.shape
					target_shape = (curr_shape[0], C.num_rois, curr_shape[2])
					ROIs_padded = np.zeros(target_shape).astype(ROIs.dtype)
					ROIs_padded[:, :curr_shape[1], :] = ROIs
					ROIs_padded[0, curr_shape[1]:, :] = ROIs[0, 0, :]
					ROIs = ROIs_padded

				for ii in range(ROIs.shape[1]):
					# Remove objects with lower confidence than confidence_threshold
					if prob[jk*ROIs.shape[1] + ii] < confidence_threshold:
						continue

					(x, y, w, h) = ROIs[0, ii, :]

					bboxes.append([16 * x, 16 * y, 16 * (x + w + 1), 16 * (y + h + 1)])
					probs.append(prob[jk*ROIs.shape[1] + ii])

			# Check whether image has predicted boxes
			if len(bboxes) != 0:
				# Apply strict NMS
				new_boxes, new_probs = roi_helpers.non_max_suppression_fast(np.array(bboxes), np.array(probs), overlap_thresh=1e-12)
			else:
				print(f'Image {img_data["filepath"].split("/")[-1]} without predictions!')
				new_boxes, new_probs = (np.array([]), np.array([]))

			gt_boxes = img_data['bboxes'].copy()
			for gt_box in gt_boxes:
				gt_box['best_iou'] = 0					# best iou for gt box
				gt_box['best_iou_index'] = None			# index of prediction with best iou score
				gt_box['overlaps_index'] = []			# other indexes of overlapping predictions

			if options.write:
				for gt_box in gt_boxes:
					cv2.rectangle(img_cropped, (gt_box['x1'], gt_box['y1']), (gt_box['x2'], gt_box['y2']), (215, 35, 35), 2)

			# Iterate over remaining boxes
			for jk in range(new_boxes.shape[0]):
				(x1, y1, x2, y2) = new_boxes[jk,:]
				(real_x1, real_y1, real_x2, real_y2) = get_real_coordinates(ratio, x1, y1, x2, y2)

				overlap = False

				for gt_box in gt_boxes:
					if real_x1 >= gt_box['x2'] or real_y1 >= gt_box['y2'] or real_x2 <= gt_box['x1'] or real_y2 <= gt_box['y1']:
						continue
					else:
						curr_iou = iou((real_x1, real_y1, real_x2, real_y2), (gt_box['x1'], gt_box['y1'], gt_box['x2'], gt_box['y2']))
						overlap = True
						if gt_box['best_iou'] < curr_iou:
							gt_box['best_iou'] = curr_iou
							if gt_box['best_iou_index']:
								gt_box['overlaps_index'].append(gt_box['best_iou_index'])
							if curr_iou > iou_threshold_tp:
								gt_box['best_iou_index'] = jk
							else:
								gt_box['overlaps_index'].append(jk)
						else:
							gt_box['overlaps_index'].append(jk)

						# Calculate distance from center point in meters
						if curr_iou >= iou_threshold_tp:
							center_gt_x, center_gt_y = (gt_box['x1'] + (gt_box['x2'] - gt_box['x1']) / 2, gt_box['y1'] + (gt_box['y2'] - gt_box['y1']) / 2)
							center_bb_x, center_bb_y = (real_x1 + (real_x2 - real_x1) / 2, real_y1 + (real_y2 - real_y1) / 2)
							d_x, d_y = (abs(center_gt_x - center_bb_x), abs(center_gt_y - center_bb_y))
							d_x_3857, d_y_3857 = (d_x * 1528.74 / 1280, d_y * 1528.74 / 1280)
							new_point_3857 = (reference_point_3857[0] + d_x_3857, reference_point_3857[1] + d_y_3857)
							new_point_4326 = transformer.transform(new_point_3857[0], new_point_3857[1])

							# geopy expects (lat, lon)
							distance = geodesic((new_point_4326[1], new_point_4326[0]), (reference_point_4326[1], reference_point_4326[0])).meters
							distance_sum += distance
							globals()['distance_sum_b_' + gt_box['bucket']] += distance
							distance_norm += 1
							globals()['distance_norm_b_' + gt_box['bucket']] += 1

				# We have not found a gt box that overlaps with the predicted box, so we have a false positive here
				if not overlap:
					fp += 1
					avg_prec.append([0, new_probs[jk]])

				# Draw predicted boxes
				if options.write:
					cv2.rectangle(img_cropped, (real_x1, real_y1), (real_x2, real_y2), (35, 35, 215), 2)
					textLabel = f'{round(float(new_probs[jk]), 7)}'
					textOrg = (real_x1, real_y1 - 10)
					cv2.putText(img_cropped, textLabel, textOrg, cv2.FONT_HERSHEY_SIMPLEX, 0.75, (35, 35, 215), 2)

			for gt_box in gt_boxes:
				if gt_box['best_iou'] >= iou_threshold_tp:
					tp += 1
					globals()['tp_b_' + gt_box['bucket']] += 1
					avg_prec.append([1, new_probs[gt_box['best_iou_index']]])

					for index in gt_box['overlaps_index']:
						fp += 1
						globals()['fp_b_' + gt_box['bucket']] += 1
						avg_prec.append([0, new_probs[index]])

				else:
					for index in gt_box['overlaps_index']:
						fp += 1
						globals()['fp_b_' + gt_box['bucket']] += 1
						avg_prec.append([0, new_probs[index]])

					fn += 1
					globals()['fn_b_' + gt_box['bucket']] += 1
					fn_height.append(float(gt_box['height']))

			# save images
			if options.write and iou_threshold_tp == 0.1:
				import os
				if not os.path.isdir("results"):
					os.mkdir("results")
				cv2.imwrite(f'./results/rpn/{confidence_threshold}/{img_data["filepath"].split("/")[-1]}', img_cropped)

		avg_prec_np_arr = np.array(avg_prec)
		avg_prec_np_arr_sorted = avg_prec_np_arr[np.argsort(avg_prec_np_arr[:, 1], axis=0)[::-1]]
		avg_precision = average_precision_score(avg_prec_np_arr_sorted[:, 0], avg_prec_np_arr_sorted[:, 1])

		print(f'average precision: {avg_precision}')

		precision = tp / (tp + fp)
		recall = tp / (tp + fn)
		f1_score = 2 * precision * recall / (precision + recall)
		mean_distance = distance_sum / distance_norm
		print(f'Precision: {precision}\nRecall: {recall}\nF1-Score: {f1_score}\nMean distance in meters: {mean_distance}\nTP: {tp}\nFP: {fp}\nFN: {fn}')

		precision_b_small = tp_b_small / (tp_b_small + fp_b_small) if tp_b_small + fp_b_small != 0 else 0
		precision_b_medium = tp_b_medium / (tp_b_medium + fp_b_medium) if tp_b_medium + fp_b_medium != 0 else 0
		precision_b_large = tp_b_large / (tp_b_large + fp_b_large) if tp_b_large + fp_b_large != 0 else 0

		recall_b_small = tp_b_small / (tp_b_small + fn_b_small) if tp_b_small + fn_b_small != 0 else 0
		recall_b_medium = tp_b_medium / (tp_b_medium + fn_b_medium) if tp_b_medium + fn_b_medium != 0 else 0
		recall_b_large = tp_b_large / (tp_b_large + fn_b_large) if tp_b_large + fn_b_large != 0 else 0

		mean_distance_b_small = distance_sum_b_small / distance_norm_b_small if distance_norm_b_small != 0 else 1
		mean_distance_b_medium = distance_sum_b_medium / distance_norm_b_medium if distance_norm_b_medium != 0 else 1
		mean_distance_b_large = distance_sum_b_large / distance_norm_b_large if distance_norm_b_large != 0 else 1

		with open('log/rpn/rpn_results.csv', 'a') as result_csv:
			log_writer = csv.writer(result_csv, delimiter=';')
			log_writer.writerow([confidence_threshold, iou_threshold_tp, precision, recall, f1_score, avg_precision, mean_distance,
								 precision_b_small, precision_b_medium, precision_b_large,
								 recall_b_small, recall_b_medium, recall_b_large,
								 mean_distance_b_small, mean_distance_b_medium, mean_distance_b_large])

		with open('log/rpn/rpn_height_fn_' + str(round(iou_threshold_tp, 1)) + '.csv', 'w') as height_csv:
			for row in fn_height:
				height_csv.write(str(row) + '\n')