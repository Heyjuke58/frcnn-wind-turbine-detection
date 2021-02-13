import cv2
import numpy as np
import copy


def augment(img_data, config, augment=True):
	assert 'filepath' in img_data
	assert 'bboxes' in img_data
	assert 'width' in img_data
	assert 'height' in img_data

	img_data_aug = copy.deepcopy(img_data)

	img = cv2.imread(img_data_aug['filepath'])
	# cropping image to remove google label
	img = img[0:1280-50, 0:1280-50]

	if augment:

		if config.translate:
			x_translation = np.random.randint(0, 201)
			y_translation = np.random.randint(0, 201)
			img = img[y_translation:1230 - (200 - y_translation), x_translation:1230 - (200 - x_translation)]
			# Only continue with boxes that are still visible in the translated image
			img_data_aug['bboxes'] = [bbox for bbox in img_data_aug['bboxes']
										if bbox['x1'] + (bbox['x2'] - bbox['x1']) / 2 < 1229 - (200 - x_translation)
										and bbox['y1'] + (bbox['y2'] - bbox['y1']) / 2 < 1229 - (200 - y_translation)
										and bbox['x1'] + (bbox['x2'] - bbox['x1']) / 2 >= x_translation
										and bbox['y1'] + (bbox['y2'] - bbox['y1']) / 2 >= y_translation]

			for bbox in img_data_aug['bboxes']:
				x1 = bbox['x1']
				y1 = bbox['y1']
				x2 = bbox['x2']
				y2 = bbox['y2']
				bbox['x1'] = max(x1 - x_translation, 0)
				bbox['y1'] = max(y1 - y_translation, 0)
				bbox['x2'] = min(x2 - x_translation, 1029)
				bbox['y2'] = min(y2 - y_translation, 1029)

		rows, cols = img.shape[:2]

		if config.use_horizontal_flips and np.random.randint(0, 2) == 0:
			img = cv2.flip(img, 1)
			for bbox in img_data_aug['bboxes']:
				x1 = bbox['x1']
				x2 = bbox['x2']
				bbox['x2'] = cols - x1
				bbox['x1'] = cols - x2

		if config.use_vertical_flips and np.random.randint(0, 2) == 0:
			img = cv2.flip(img, 0)
			for bbox in img_data_aug['bboxes']:
				y1 = bbox['y1']
				y2 = bbox['y2']
				bbox['y2'] = rows - y1
				bbox['y1'] = rows - y2

		if config.rot_90:
			angle = np.random.choice([0,90,180,270],1)[0]
			if angle == 270:
				img = np.transpose(img, (1,0,2))
				img = cv2.flip(img, 0)
			elif angle == 180:
				img = cv2.flip(img, -1)
			elif angle == 90:
				img = np.transpose(img, (1,0,2))
				img = cv2.flip(img, 1)
			elif angle == 0:
				pass

			for bbox in img_data_aug['bboxes']:
				x1 = bbox['x1']
				x2 = bbox['x2']
				y1 = bbox['y1']
				y2 = bbox['y2']
				if angle == 270:
					bbox['x1'] = y1
					bbox['x2'] = y2
					bbox['y1'] = cols - x2
					bbox['y2'] = cols - x1
				elif angle == 180:
					bbox['x2'] = cols - x1
					bbox['x1'] = cols - x2
					bbox['y2'] = rows - y1
					bbox['y1'] = rows - y2
				elif angle == 90:
					bbox['x1'] = rows - y2
					bbox['x2'] = rows - y1
					bbox['y1'] = x1
					bbox['y2'] = x2        
				elif angle == 0:
					pass

	img_data_aug['width'] = img.shape[1]
	img_data_aug['height'] = img.shape[0]

	return img_data_aug, img


def augment_modified(img_data, config, augment=True):
	assert 'filepath' in img_data
	assert 'coordinates' in img_data
	assert 'width' in img_data
	assert 'height' in img_data

	img_data_aug = copy.deepcopy(img_data)

	img = cv2.imread(img_data_aug['filepath'])
	# cropping image to remove google label
	img = img[0:1280-50, 0:1280-50]

	if augment:

		if config.translate:
			x_translation = np.random.randint(0, 201)
			y_translation = np.random.randint(0, 201)
			img = img[y_translation:1230 - (200 - y_translation), x_translation:1230 - (200 - x_translation)]
			# Only continue with boxes that are still visible in the translated image
			img_data_aug['coordinates'] = [coord for coord in img_data_aug['coordinates']
									  if coord['x'] < 1229 - (200 - x_translation)
									  and coord['y'] < 1229 - (200 - y_translation)
									  and coord['x'] >= x_translation
									  and coord['y'] >= y_translation]

			for coord in img_data_aug['coordinates']:
				x = coord['x']
				y = coord['y']
				coord['x'] = x - x_translation
				coord['y'] = y - y_translation

		rows, cols = img.shape[:2]

		if config.use_horizontal_flips and np.random.randint(0, 2) == 0:
			img = cv2.flip(img, 1)
			for coord in img_data_aug['coordinates']:
				x = coord['x']
				coord['x'] = cols - x

		if config.use_vertical_flips and np.random.randint(0, 2) == 0:
			img = cv2.flip(img, 0)
			for coord in img_data_aug['coordinates']:
				y = coord['y']
				coord['y'] = rows - y

		if config.rot_90:
			angle = np.random.choice([0,90,180,270],1)[0]
			if angle == 270:
				img = np.transpose(img, (1,0,2))
				img = cv2.flip(img, 0)
			elif angle == 180:
				img = cv2.flip(img, -1)
			elif angle == 90:
				img = np.transpose(img, (1,0,2))
				img = cv2.flip(img, 1)
			elif angle == 0:
				pass

			for coord in img_data_aug['coordinates']:
				x = coord['x']
				y = coord['y']
				if angle == 270:
					coord['x'] = y
					coord['y'] = cols - x
				elif angle == 180:
					coord['x'] = cols - x
					coord['y'] = rows - y
				elif angle == 90:
					coord['x'] = rows - y
					coord['y'] = x
				elif angle == 0:
					pass

	img_data_aug['width'] = img.shape[1]
	img_data_aug['height'] = img.shape[0]

	return img_data_aug, img