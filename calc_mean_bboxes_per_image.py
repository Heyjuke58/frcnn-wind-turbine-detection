import numpy as np


id = None
with open('set_splits/bboxes_train_val_test_split_3buckets.csv', 'r') as bboxes:
    count_boxes = []
    boxes = 0
    for line in bboxes:
        filename, x1, y1, x2, y2, class_name, bucket, set = line.strip().split(',')
        new_id = filename.split('_')[0]

        if not id:
            id = new_id

        if id == new_id:
            boxes += 1
        else:
            count_boxes.append(boxes)
            id = new_id
            boxes = 1

print(len(count_boxes))
np_arr = np.array(count_boxes)

mean = np.mean(np_arr)

print(mean)