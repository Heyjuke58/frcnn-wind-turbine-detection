import os
import random
import shutil

dir = r'D:\_nefino\turbine_detection\turbine_detection\images'

dst_dir=r'D:\_nefino\turbine_detection\turbine_detection\image_sample'

images = [os.path.join(dir, img_name) for img_name in os.listdir(dir)]

sample = random.sample(images, 2000)

for img in sample:
    img_name = img.split('\\')[-1]
    shutil.copyfile(img, os.path.join(dst_dir, img_name))


# Auswertung fehlerhafte Bilder:
# Total random smaple
# Sample size: 2000
#
# Bounding Box in der keine Anlage ist: 104
# Anlage ohne Bounding Box: 209
# 1: 78
# 2: 34
# 3: 7
# 4: 5
# 5: 6
# 6: 3
# 7: 3
# 9 : 1
# 10: 1
# 11: 1
# 13: 1
# 18: 1

# High loss RPN sample
# Sample size: 2000
# Bounding Box in der keine Anlage ist: 590
# Anlage ohne Bounding Box: 636
# 1: 132
# 2: 97
# 3: 36
# 4: 30
# 5: 21
# 6: 23
# 7: 11
# 8: 9
# 9 : 6
# 10: 5
# 11: 2
# 12: 1
# 13: 1
# 14: 1
# 15: 2
# 16: 0
# 17: 2
# 18: 3

# Low loss RPN sample
# Sample size: 2000
# Bounding Box in der keine Anlage ist: 3
# Anlage ohne Bounding Box: 133
# 1: 83
# 2: 17
# 3: 6
# 4:
# 5:
# 6:
# 7:
# 8:
# 9 :
# 10:
# 11:
# 12:
# 13:
# 14:
# 15:
# 16:
# 17:
# 18: