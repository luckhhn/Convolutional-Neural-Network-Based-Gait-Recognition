from sklearn.externals import joblib
import glob as gb
from tqdm import tqdm
import numpy as np
import cv2
from sklearn.model_selection import train_test_split, ShuffleSplit
import tensorflow as tf
from keras.utils import to_categorical
from sklearn.externals import joblib

img_path = gb.glob('D:\\AI2thsemester\\GEI' + "\\*.png")
img_list = []
label_list = []
for path in tqdm(img_path):
    img = cv2.imread(path)
    # img = cv2.resize(img, (40, 80))
    label = int(path[23:26])
    label_list.append(label)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_list.append(img.flatten())

X_train_orig = np.array(img_list)
Y_train_orig = np.array(label_list)

X_train_orig = X_train_orig / 255
print("X_train_orig.shape", X_train_orig.shape)
print("Y_train_orig.shape", Y_train_orig.shape)
joblib.dump(X_train_orig, 'X_train_orig.pkl')
joblib.dump(Y_train_orig, 'Y_train_orig.pkl')