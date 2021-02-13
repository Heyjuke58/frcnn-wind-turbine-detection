import cv2
import os

def get_data(input_path: str, test_only: bool = False):
	found_bg = False
	all_imgs = {}
	classes_count = {}
	class_mapping = {}
	
	with open(input_path, 'r') as f:

		print('Parsing annotation files')

		for line in f:
			filename, x1, y1, x2, y2, class_name, bucket, height, set = line.strip().split(',')

			if test_only and set != 'test':
				continue

			if class_name not in classes_count:
				classes_count[class_name] = 1
			else:
				classes_count[class_name] += 1

			if class_name not in class_mapping:
				if class_name == 'bg' and found_bg == False:
					print('Found class name with special name bg. Will be treated as a background region (this is usually for hard negative mining).')
					found_bg = True
				class_mapping[class_name] = len(class_mapping)

			if filename not in all_imgs:
				path = 'images/' + filename
				if not os.path.exists(path):
					continue
				img = cv2.imread(path)
				img = img[0:1280-50, 0:1280-50]
				(rows,cols) = img.shape[:2]
				all_imgs[filename] = {}
				all_imgs[filename]['filepath'] = path
				all_imgs[filename]['width'] = cols
				all_imgs[filename]['height'] = rows
				all_imgs[filename]['bboxes'] = []
				all_imgs[filename]['imageset'] = set


			if int(x1) + (int(x2) - int(x1)) / 2 >= 1230 or int(y1) + (int(y2) - int(y1)) / 2 >= 1230:
				continue
			x2 = 1229 if int(x2) >= 1230 else int(x2)
			y2 = 1229 if int(y2) >= 1230 else int(y2)

			all_imgs[filename]['bboxes'].append({'class': class_name, 'bucket': bucket, 'height': height, 'x1': int(x1), 'y1': int(y1), 'x2': int(x2), 'y2': int(y2)})

		all_data = []
		for key in all_imgs:
			all_data.append(all_imgs[key])

		# make sure the bg class is last in the list
		if found_bg:
			if class_mapping['bg'] != len(class_mapping) - 1:
				key_to_switch = [key for key in class_mapping.keys() if class_mapping[key] == len(class_mapping)-1][0]
				val_to_switch = class_mapping['bg']
				class_mapping['bg'] = len(class_mapping) - 1
				class_mapping[key_to_switch] = val_to_switch
		
		return all_data, classes_count, class_mapping


def get_data_modified(input_path: str, test_only: bool = False):
	found_bg = False
	all_imgs = {}
	classes_count = {}
	class_mapping = {}

	with open(input_path,'r') as f:

		print('Parsing annotation files')

		for line in f:
			filename, x, y, class_name, set = line.strip().split(',')

			if test_only and set != 'test':
				continue

			if class_name not in classes_count:
				classes_count[class_name] = 1
			else:
				classes_count[class_name] += 1

			if class_name not in class_mapping:
				if class_name == 'bg' and found_bg == False:
					print('Found class name with special name bg. Will be treated as a background region (this is usually for hard negative mining).')
					found_bg = True
				class_mapping[class_name] = len(class_mapping)

			if filename not in all_imgs:
				all_imgs[filename] = {}
				path = 'images/' + filename
				img = cv2.imread(path)
				img = img[0:1280-50, 0:1280-50]
				(rows,cols) = img.shape[:2]
				all_imgs[filename]['filepath'] = path
				all_imgs[filename]['width'] = cols
				all_imgs[filename]['height'] = rows
				all_imgs[filename]['coordinates'] = []
				all_imgs[filename]['imageset'] = set

			if int(x) >= 1230 or int(y) >= 1230:
				continue

			all_imgs[filename]['coordinates'].append({'class': class_name, 'x': int(x), 'y': int(y)})

		all_data = []
		for key in all_imgs:
			all_data.append(all_imgs[key])

		# make sure the bg class is last in the list
		if found_bg:
			if class_mapping['bg'] != len(class_mapping) - 1:
				key_to_switch = [key for key in class_mapping.keys() if class_mapping[key] == len(class_mapping)-1][0]
				val_to_switch = class_mapping['bg']
				class_mapping['bg'] = len(class_mapping) - 1
				class_mapping[key_to_switch] = val_to_switch

		return all_data, classes_count, class_mapping
