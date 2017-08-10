from os import environ
environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # ignore debugging logs

import tensorflow as tf
import numpy as np

from config import *

class Predictor():

    def __init__(self):
        self.loaded_graph_type = None
        self.loaded_model_name = None
        self.saver = None
        self.input_image_string = None
        self.probs = None

        ### CLASSES ###
        with open('./models/imagenet-classes.txt') as f:
            classes = f.read().splitlines()
            self.classes = np.array(classes) # if we want to access multiple elements, knowing their index


    def load_graph_checkpoints(self, model_name):
        tf.reset_default_graph()

        #try:
        self.saver = tf.train.import_meta_graph('./models/'+model_name+'/model.ckpt.meta')

        self.input_image_string = tf.get_default_graph().get_tensor_by_name('input_image_string:0')
        self.probs = tf.get_default_graph().get_tensor_by_name('probs:0')

        self.loaded_graph_type = 'checkpoints'
        self.loaded_model_name = model_name


    def evaluate_checkpoints(self, filename):
        # Load image
        with open(UPLOAD_FOLDER + '/' + filename, 'rb') as f:
            image_string = f.read()

        # Session
        with tf.Session() as sess:
            # Restore variables values
            self.saver.restore(sess, './models/'+self.loaded_model_name+'/model.ckpt')

            prob_values = sess.run(self.probs, feed_dict={
                self.input_image_string: image_string
                })

            pred_idx = prob_values[0].argsort()[-5:][::-1]
            pred_class = self.classes[pred_idx - 1] # from 1001 to 1000 classes
            pred_score = np.around(100*prob_values[0][pred_idx], decimals=2) # two decimals

            return list(pred_class), list(pred_score)


    def load_graph_frozen(self, model_name):
        tf.reset_default_graph()

        from tensorflow.core.framework import graph_pb2
        graph_def = graph_pb2.GraphDef()
        with open("./models/"+model_name+"/frozen_graph.pb", "rb") as f:
            graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name='')

        #try:
        self.input_image_string = tf.get_default_graph().get_tensor_by_name('input_image_string:0')
        self.probs = tf.get_default_graph().get_tensor_by_name('probs:0')

        self.loaded_graph_type = 'frozen'
        self.loaded_model_name = model_name


    def evaluate_frozen(self, filename):
        # Load image
        with open(UPLOAD_FOLDER + '/' + filename, 'rb') as f:
            image_string = f.read()

        # Session
        with tf.Session() as sess:
            prob_values = sess.run(self.probs, feed_dict={
                self.input_image_string: image_string
                })

            pred_idx = prob_values[0].argsort()[-5:][::-1]
            pred_class = self.classes[pred_idx - 1] # from 1001 to 1000 classes
            pred_score = np.around(100*prob_values[0][pred_idx], decimals=2) # two decimals

            return list(pred_class), list(pred_score)


    def evaluate_compiled(self, filename, model_name):
        pred_class = []
        pred_score = []

        import subprocess
        output = subprocess.check_output(['./models/imagenet_cc', './models/'+model_name+'/frozen_graph.pb', UPLOAD_FOLDER+'/'+filename])
        output = output.splitlines()
        for index, value in zip(output[0::2], output[1::2]):
            pred_class.append(self.classes[int(index)])
            pred_score.append(float(value)*100)

        return pred_class, np.around(pred_score, decimals=2)


    def evaluate(self, filename, model_name, graph_type):
        if graph_type == 'checkpoints':
            if (graph_type != self.loaded_graph_type) or (model_name != self.loaded_model_name):
                self.load_graph_checkpoints(model_name)
            pred_class, pred_score = self.evaluate_checkpoints(filename)

        elif graph_type == 'frozen':
            if (graph_type != self.loaded_graph_type) or (model_name != self.loaded_model_name):
                self.load_graph_frozen(model_name)
            pred_class, pred_score = self.evaluate_frozen(filename)

        elif graph_type == 'compiled':
            pred_class, pred_score = self.evaluate_compiled(filename, model_name)

        # if pred_class != None
        return pred_class, pred_score