import tensorflow as tf
import numpy as np
import cv2
from config import *

### GRAPH ###
tf.reset_default_graph()

saver = tf.train.import_meta_graph('./models/googlenet_model.ckpt.meta')

inputs = tf.get_default_graph().get_tensor_by_name('Placeholder:0')
prob = tf.get_default_graph().get_tensor_by_name('prob:0')
print('The input placeholder is expecting an array of shape {} and type {}'.format(inputs.shape, inputs.dtype))


### CLASSES ###
with open('./models/imagenet-classes.txt') as f:
    classes = f.read().splitlines()
    classes = np.array(classes) # if we want to access multiple elements, knowing their index


def evaluate(filename):

    ### IMAGE PREPROCESSING ###
    img = cv2.imread('./static_files/'+filename)
    prep_img = cv2.resize(img, (224, 224), interpolation = cv2.INTER_CUBIC)
    prep_img = prep_img.reshape([1, 224, 224, 3])
    print("The input image has been resized from {} to {}".format(img.shape, prep_img.shape))

    ### SESSION ###
    with tf.Session() as sess:
        saver.restore(sess, './models/googlenet_model.ckpt')

        prob_values = sess.run(prob, feed_dict={
            inputs: prep_img
            })

        pred_idx = prob_values[0].argsort()[-5:][::-1]
        pred_class = classes[pred_idx]
        pred_certain = np.around(100*prob_values[0][pred_idx], decimals=2) # two decimals
        
        return list(pred_class), list(pred_certain)