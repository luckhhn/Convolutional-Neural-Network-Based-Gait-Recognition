from sklearn.externals import joblib
import glob as gb
from tqdm import tqdm
import numpy as np
# import cv2
from sklearn.model_selection import train_test_split, ShuffleSplit
import tensorflow as tf
from keras.utils import to_categorical
from sklearn.externals import joblib
import math


X_train_orig = joblib.load('X_train_orig.pkl')
Y_train_orig = joblib.load('Y_train_orig.pkl')
print("X_train_orig.shape", X_train_orig.shape)
print("Y_train_orig.shape", Y_train_orig.shape)

X_train, X_test, Y_train, Y_test = train_test_split(X_train_orig, Y_train_orig, test_size=0.1, random_state=1)
###########################
number_of_subjects = 124
###########################
# X_train = X_train.astype(np.float32)
Y_train = to_categorical(Y_train)
Y_train = np.delete(Y_train, 0, 1)
Y_test = to_categorical(Y_test)
Y_test = np.delete(Y_test, 0, 1)
print("X_train.shape", X_train.shape)
print("X_test.shape", X_test.shape)
print("Y_train(after one hot).shape", Y_train.shape)
print("Y_test(after one hot).shape", Y_test.shape)


def next_batch(data, label_, size):
    data_list = []
    label_list = []
    for i in range(math.ceil(data.shape[0] / size)):
        # for i in range(int((data.shape[0] - (data.shape[0] % size) / size))):
        head = ((i * size) % data.shape[0])
        # head = (i * size) % (data.shape[0] - (data.shape[0] % size))
        tail = min((head + size - 1), data.shape[0])
        print('abc', head, tail)
        data_list.append(data[head:tail, :])
        label_list.append(label_[head:tail, :])

        # data_list = np.array(data_list)
        # label_list = np.array(label_list)
    return data_list, label_list


sess = tf.InteractiveSession()
x = tf.placeholder("float32", [None, 3200])
y_ = tf.placeholder("float32", [None, 124])


#
# def generate_batch(self):
#     features_placeholder = tf.placeholder(self.features.dtype, self.features.shape)
#     labels_placeholder = tf.placeholder(self.labels.dtype, self.labels.shape)
#     dataset = tf.data.Dataset.from_tensor_slices((self.features, self.labels))
#     dataset = dataset.repeat(100)
#     batched_dataset = dataset.batch(100)
#     iterator = batched_dataset.make_initializable_iterator()
#     batch_xs, batch_ys = iterator.get_next()
#     return iterator.initializer,batch_xs, batch_ys


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')


W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

x_image = tf.reshape(x, [-1, 40, 80, 1])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

W_fc1 = weight_variable([round(40 / 4) * round(80 / 4) * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, round(40 / 4) * round(80 / 4) * 64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([1024, 124])
b_fc2 = bias_variable([124])

y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

cross_entropy = -tf.reduce_sum(y_ * tf.log(tf.clip_by_value(y_conv, 1e-8, 1.0)))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
sess.run(tf.initialize_all_variables())
num_epoch = 100
minibatch_size = 13
# batch = generate_batch()
iteration = round(X_train.shape[0] / minibatch_size) * num_epoch
print("Number of iteration:", iteration)
print("1 epoch = ", X_train.shape[0] / minibatch_size, "iteration")
print(18 % (X_train.shape[0] / minibatch_size))
Xtrain_batch, Ytrain_batch = next_batch(X_train, Y_train, minibatch_size)

for iteration in tqdm(range(iteration)):
    index = iteration % len(Xtrain_batch)
    if iteration % round(X_train.shape[0] / minibatch_size) == 0:
        train_cross_entropy = cross_entropy.eval(feed_dict={
            x: Xtrain_batch[index], y_: Ytrain_batch[index], keep_prob: 1.0})
        print("Iteration %d, cross_entropy_print %g" % (iteration, train_cross_entropy))

        train_accuracy = accuracy.eval(feed_dict={
            x: Xtrain_batch[index], y_: Ytrain_batch[index], keep_prob: 1.0})
        print("training accuracy %g" % train_accuracy)
    train_step.run(feed_dict={x: Xtrain_batch[index], y_: Ytrain_batch[index], keep_prob: 0.5})

print("test accuracy %g" % accuracy.eval(feed_dict={
    x: X_test, y_: Y_test, keep_prob: 1.0}))
