# -*- coding: utf-8 -*-
"""
Created on Thu Jul 2018

@author: Yuhao Ye
"""

import os
import numpy as np
import cv2
from tqdm import tqdm
from sklearn.externals import joblib
import glob as gb

rootDir = 'D:\AI2thsemester\Gait_Database\GEI_CASIA_B'
gait_video_path_length = len("D:\\AI2thsemester\\Gait_Database\\GEI_CASIA_B\\001\\nm-01")
print(gait_video_path_length)

path_list = []


def visitDir(path):
    if not os.path.isdir(path):
        print('Error: "', path, '" is not a directory or does not exist.')
        return
    else:
        global num
        try:
            for lists in os.listdir(path):
                sub_path = os.path.join(path, lists)
                num += 1
                # print('No.', x, ' ', sub_path)
                if os.path.isdir(sub_path):
                    visitDir(sub_path)
        except:
            pass


def path_generator(rootDir):
    count = 0
    for root, dirs, files in os.walk(rootDir, topdown=False):
        for name in dirs:
            if len(os.path.join(root, name)) == gait_video_path_length and os.path.join(root, name)[-4] == "m":
                count += 1
                path_list.append(os.path.join(root, name))
                # print(os.path.join(root, name), )
    print(count, "Gait Videos in Total")


path_generator(rootDir)
for i in range(len(path_list)):
    print(path_list[i])
#
img_list = []
label_list = []
for index in tqdm(range(len(path_list))):
    num = 0
    visitDir(path_list[index])
    img_path = gb.glob(path_list[index] + "\\*.png")

    for path in img_path:

        img = cv2.imread(path)
        img = cv2.resize(img, (32, 32))
        label = int(path[43:46])
        label_list.append(label)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_list.append(img.flatten())

X_train_orig = np.array(img_list)
Y_train_orig = np.array(label_list)

X_train_orig = X_train_orig / 255
print("X_train_orig.shape", X_train_orig.shape)
print("Y_train_orig.shape", Y_train_orig.shape)
joblib.dump(X_train_orig, 'X_train_orig_casia_b_official.pkl')
joblib.dump(Y_train_orig, 'Y_train_orig_casia_b_official.pkl')
print(Y_train_orig)
