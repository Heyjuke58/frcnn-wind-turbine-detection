import cv2
import os

def apply_brightness_contrast(input_img, brightness = 0, contrast = 0):

    if brightness != 0:
        if brightness > 0:
            shadow = brightness
            highlight = 255
        else:
            shadow = 0
            highlight = 255 + brightness
        alpha_b = (highlight - shadow)/255
        gamma_b = shadow

        buf = cv2.addWeighted(input_img, alpha_b, input_img, 0, gamma_b)
    else:
        buf = input_img.copy()

    if contrast != 0:
        f = 131*(contrast + 127)/(127*(131-contrast))
        alpha_c = f
        gamma_c = 127*(1-f)

        buf = cv2.addWeighted(buf, alpha_c, buf, 0, gamma_c)

    return buf


def hisEqulColor(img):
    ycrcb=cv2.cvtColor(img,cv2.COLOR_BGR2YCR_CB)
    channels=cv2.split(ycrcb)
    cv2.equalizeHist(channels[0],channels[0])
    cv2.merge(channels,ycrcb)
    cv2.cvtColor(ycrcb,cv2.COLOR_YCR_CB2BGR,img)
    return img



dir = r'D:\_nefino\turbine_detection\turbine_detection\images'
images = ['7802_7.729_49.578_2004-12-01.jpg', '20099_9.509_52.688_2005-07-01.jpg', '18767_8.020_49.746_2014-01-01.jpg', '26505_12.292_52.200_2005-02-01.jpg', '5421_11.481_51.682_2008-12-01.jpg', '5711_6.979_52.055_2009-01-01.jpg', '12867_8.207_49.177_2014-08-01.jpg']
i = 0

for img_name in images:
    img_path = os.path.join(dir, img_name)
    img = cv2.imread(img_path)
    # cropping image to remove google label
    img = img[0:1280-50, 0:1280-50]

    cv2.imshow('before', img)

    bright_img = apply_brightness_contrast(img, 45, 45)

    cv2.imshow('bright', bright_img)


    high_contrast_img = hisEqulColor(img)
    cv2.imshow('hist_equal', high_contrast_img)

    cv2.waitKey(0)
