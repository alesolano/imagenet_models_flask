import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # ignore debugging logs

import tensorflow as tf
import numpy as np
import cv2

from config import *

### GRAPH ###
tf.reset_default_graph()

saver = tf.train.import_meta_graph('./models/inception_resnet_v2/model.ckpt.meta')

input_image_string = tf.get_default_graph().get_tensor_by_name('input_image_string:0')
probs = tf.get_default_graph().get_tensor_by_name('probs:0')


### CLASSES ###
with open('./models/imagenet-classes.txt') as f:
    classes = f.read().splitlines()
    classes = np.array(classes) # if we want to access multiple elements, knowing their index


def evaluate(filename):

    ### LOAD IMAGE ###
    with open(UPLOAD_FOLDER + '/' + filename, 'rb') as f:
        image_string = f.read()

    ### SESSION ###
    with tf.Session() as sess:
        saver.restore(sess, './models/inception_resnet_v2/model.ckpt')

        prob_values = sess.run(probs, feed_dict={
            input_image_string: image_string
            })

        pred_idx = prob_values[0].argsort()[-5:][::-1]
        pred_class = classes[pred_idx - 1] # from 1001 to 1000 classes
        pred_score = np.around(100*prob_values[0][pred_idx], decimals=2) # two decimals

        return list(pred_class), list(pred_score
