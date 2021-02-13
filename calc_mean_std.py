import cv2
import os
import numpy as np

# https://stats.stackexchange.com/questions/25848/how-to-sum-a-standard-deviation

dir = r'images'

red_means = np.array([])
green_means = np.array([])
blue_means = np.array([])

red_var = np.array([])
green_var = np.array([])
blue_var = np.array([])

for img_name in os.listdir(dir):
    if int(img_name.split('_')[0]) >= 27001:
        img_path = os.path.join(dir, img_name)
        img = cv2.imread(img_path)
        # cropping image to remove google label
        img = img[0:1280-50, 0:1280-50]

        img = img[:, :, (2, 1, 0)] # BGR -> RGB

        red_means = np.append(red_means, np.mean(img[:, :, 0].flatten()))
        green_means = np.append(green_means, np.mean(img[:, :, 1].flatten()))
        blue_means = np.append(blue_means, np.mean(img[:, :, 2].flatten()))

        red_var = np.append(red_var, np.var(img[:, :, 0].flatten()))
        green_var = np.append(green_var, np.var(img[:, :, 1].flatten()))
        blue_var = np.append(blue_var, np.var(img[:, :, 2].flatten()))

red_mean = np.mean(red_means)
green_mean = np.mean(green_means)
blue_mean = np.mean(blue_means)

red_std = np.sqrt(np.mean(red_var))
green_std = np.sqrt(np.mean(green_var))
blue_std = np.sqrt(np.mean(blue_var))


print(f'RED\n____________\nMean: {red_mean}\nStd: {red_std}\n\nGREEN\n__________\nMean: {green_mean}\nStd: {green_std}\n\nBLUE\n__________\nMean: {blue_mean}\nStd: {blue_std}')
