import pandas as pd
import matplotlib.pyplot as plt
from optparse import OptionParser
import shutil
import os
import random

df = pd.read_csv('log/losses.csv', delimiter=';')

parser = OptionParser()

parser.add_option("-w", "--write", dest="write", help="Write images with high loss into folder.", action='store_true')
parser.add_option("-l", dest="low_loss", help="Write images with low loss into folder", action='store_true')
(options, args) = parser.parse_args()

dst_dir = r'D:\_nefino\turbine_detection\turbine_detection\bbox\high_loss'
if options.low_loss:
    dst_dir = r'D:\_nefino\turbine_detection\turbine_detection\bbox\low_loss'
src_dir = r'D:\_nefino\turbine_detection\turbine_detection\bbox'

if options.write:
    if not options.low_loss:
        for index, row in df[df['loss'] > 3.8].iterrows():
            img_name = row['img'].split('/')[-1]
            shutil.copyfile(os.path.join(src_dir, img_name), os.path.join(dst_dir, img_name))
    else:
        img_names = []
        for index, row in df[df['loss'] < 1].iterrows():
            img_names.append(row['img'].split('/')[-1])
        sample = random.sample(img_names, 2000)

        for img_name in sample:
            shutil.copyfile(os.path.join(src_dir, img_name), os.path.join(dst_dir, img_name))


#plt.figure(figsize=(16, 10))
plt.figure(figsize=(8, 5))

#plt.subplot(221)
plt.hist(df['loss'], range=(0, 10), rwidth=0.25)
plt.xlabel('Verlust')
plt.ylabel('Anzahl an Bildern')
plt.grid(axis='y')

# plt.subplot(222)
# plt.hist(df['loss_rpn_cls'], range=(0, 10))
# plt.title('RPN cls Loss')
# plt.xlabel('cls Loss')
# plt.ylabel('# Training instances')
#
# plt.subplot(223)
# plt.hist(df['loss_rpn_regr'], range=(0, 10))
# plt.title('RPN regr Loss')
# plt.xlabel('regr Loss')
# plt.ylabel('# Training instances')

plt.show()