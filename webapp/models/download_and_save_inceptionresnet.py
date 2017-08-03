import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import tensorflow as tf
from tensorflow.contrib import slim

from sys import path as syspath

models_slim_dir = None

if models_slim_dir is None:
  print("Replace 'models_slim_dir' with path of clone_dir/models/slim from: git clone https://github.com/tensorflow/models.git")
  exit()
else:
  syspath.append(models_slim_dir)

# These 3 modules are not included in TensorFlow
from datasets import dataset_utils, imagenet
from nets import inception_resnet_v2
from preprocessing import inception_preprocessing # see preprocessing/preprocessing_factory.py

image_size = inception_resnet_v2.inception_resnet_v2.default_image_size


### DOWNLOAD CHECKPOINTS ###
net_name = 'inception_resnet_v2'
net_version = '_2016_08_30'

url = "http://download.tensorflow.org/models/"+net_name+net_version+".tar.gz"
checkpoints_dir = '/tmp/' + net_name

if not tf.gfile.Exists(checkpoints_dir):
    tf.gfile.MakeDirs(checkpoints_dir)

if not tf.gfile.Exists(checkpoints_dir+'/'+net_name+net_version+'.ckpt'):
    dataset_utils.download_and_uncompress_tarball(url, checkpoints_dir)


with tf.Graph().as_default():

    ### PLACEHOLDER + PREPROCESSING ###

    image_string = tf.placeholder(tf.string, name='input_image_string')
    image = tf.image.decode_jpeg(image_string, channels=3)
    # image is a Tensor of shape (?, ?, 3)
    
    processed_image = inception_preprocessing.preprocess_image(image, image_size, image_size, is_training=False)
    processed_images  = tf.expand_dims(processed_image, 0)


    ### LOAD GRAPH ###
    # Create the model, use the default arg scope to configure the batch norm parameters.
    with slim.arg_scope(inception_resnet_v2.inception_resnet_v2_arg_scope()):
        logits, _ = inception_resnet_v2.inception_resnet_v2(processed_images, num_classes=1001, is_training=False)
    

    ### OUTPUT ###
    # Get the top 5 predictions
    probabilities = tf.nn.softmax(logits, name='probs')
    values, indices = tf.nn.top_k(probabilities[0], 5, name='top_k')
    
    init_fn = slim.assign_from_checkpoint_fn(
        checkpoints_dir+'/'+net_name+net_version+'.ckpt',
        slim.get_model_variables('InceptionResnetV2'))
    

    ### SESSION ###
    with tf.Session() as sess:
        # Restore variables
        init_fn(sess)

        # Tensorboard
        writer = tf.summary.FileWriter(checkpoints_dir + "/1")
        writer.add_graph(sess.graph)
        print("Tensorboard files saved in: %s" % checkpoints_dir + "/1")

        # Save
        saver = tf.train.Saver()
        save_path = saver.save(sess, checkpoints_dir + "/model.ckpt")
        tf.train.write_graph(sess.graph.as_graph_def(), checkpoints_dir, "graph.pb")
        print("Model and graph saved in: %s\n" % checkpoints_dir)