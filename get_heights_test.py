with open('log/test_heights.csv', 'w') as test_heights:
    with open('set_splits/bboxes_train_val_test_split_3buckets_with_height.csv', 'r') as bboxes:
        for line in bboxes:
            filename, x1, y1, x2, y2, class_name, bucket, height, set = line.strip().split(',')

            if set == 'test':
                test_heights.write(height + '\n')