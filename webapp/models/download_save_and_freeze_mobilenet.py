#!/usr/bin/python3

# Based on https://gist.github.com/StanislawAntol/656e3afe2d43864bb410d71e1c5789c1
# Author @StanislawAntol

from sys import exit
from sys import stdout
from os import path as osp
from os import stat, environ
import tarfile

from six.moves import urllib

environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Ignore debugging logs

import tensorflow as tf
slim = tf.contrib.slim
from tensorflow.python.framework import graph_util

from sys import path as syspath

models_slim_dir = None
if models_slim_dir is None:
  print("Replace 'models_slim_dir' with path of clone_dir/models/slim from: git clone https://github.com/tensorflow/models.git")
  exit()
else:
  syspath.append(models_slim_dir)

net_name = 'mobilenet_v1'

download_dir = None
if download_dir is None:
    print("Replace 'download_dir' with your path for downloading. Recommended for Linux and MacOS:  '/tmp/'+net_name  ")
    exit()

if not tf.gfile.Exists(download_dir):
    tf.gfile.MakeDirs(download_dir)

checkpoints_dir = net_name
if not tf.gfile.Exists(checkpoints_dir):
    tf.gfile.MakeDirs(checkpoints_dir)


# From tensorflow/models/slim
from nets import mobilenet_v1
from datasets import imagenet
from preprocessing import inception_preprocessing # see preprocessing/preprocessing_factory.py



def download_and_uncompress_tarball(base_url, filename, data_dir):
  
  def _progress(count, block_size, total_size):
    stdout.write('\r>> Downloading %s %.1f%%' % (
        filename, float(count * block_size) / float(total_size) * 100.0))
    stdout.flush()
    
  tarball_url = base_url + filename
  filepath = osp.join(data_dir, filename)

  if not tf.gfile.Exists( osp.join(download_dir, model_dl) ):
    filepath, _ = urllib.request.urlretrieve(tarball_url, filepath, _progress)
    print()
    statinfo = stat(filepath)
    print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
  else:
    print('{} tarball already exists -- not downloading'.format(filename))

  tarfile.open(filepath, 'r:*').extractall(data_dir)


def freeze_mobilenet(meta_file, img_size=224, factor=1.0, num_classes=1001):

  tf.reset_default_graph()
  
  image_string = tf.placeholder(tf.string, name='input_image_string')
  image = tf.image.decode_jpeg(image_string, channels=3)
  # image is a Tensor of shape (?, ?, 3)
    
  processed_image = inception_preprocessing.preprocess_image(image, img_size, img_size, is_training=False)
  processed_images  = tf.expand_dims(processed_image, 0)

  is_training=False  
  weight_decay = 0.0
  arg_scope = mobilenet_v1.mobilenet_v1_arg_scope(weight_decay=weight_decay)
  with slim.arg_scope(arg_scope):
    logits, _ = mobilenet_v1.mobilenet_v1(processed_images, 
                                          num_classes=num_classes, 
                                          is_training=is_training, 
                                          depth_multiplier=factor)

  # Get the top 5 predictions
  probabilities = tf.nn.softmax(logits, name='probs')
  values, indices = tf.nn.top_k(probabilities[0], 5, name='top_k')

  ckpt_file = meta_file.replace('.meta', '')
  output_graph_fn = osp.join(checkpoints_dir, "frozen_graph.pb")
  output_node_names = "top_k"

  rest_var = slim.get_variables_to_restore()

  with tf.Session() as sess:
    graph = tf.get_default_graph()
    input_graph_def = graph.as_graph_def()
    
    saver = tf.train.Saver(rest_var)
    saver.restore(sess, ckpt_file)

    # Tensorboard
    writer = tf.summary.FileWriter(checkpoints_dir + "/1")
    writer.add_graph(sess.graph)
    print("Tensorboard files saved in: %s" % checkpoints_dir + "/1")

    # Save checkpoints
    save_path = saver.save(sess, osp.join(checkpoints_dir, "model.ckpt"))
    print("Model and graph saved in: %s\n" % checkpoints_dir)

    # Freeze
    # We use a built-in TF helper to export variables to constant
    output_graph_def = graph_util.convert_variables_to_constants(
        sess, # The session is used to retrieve the weights
        input_graph_def, # The graph_def is used to retrieve the nodes 
        output_node_names.split(",") # The output node names are used to select the usefull nodes
    )

    # Finally we serialize and dump the output graph to the filesystem
    with tf.gfile.GFile(output_graph_fn, "wb") as f:
        f.write(output_graph_def.SerializeToString())
    print("{} ops in the final graph.".format(len(output_graph_def.node)))


#factors = ['0.25', '0.50', '0.75', '1.0']
#img_sizes = [128, 160, 192, 224]
factor = '1.0'
img_size = 224

base_url = 'http://download.tensorflow.org/models/'

model_date = '2017_06_14'
model_base_fmt = net_name+'_{}_{}'
model_dl_fmt = model_base_fmt + '_{}.tar.gz'
model_pb_fmt = model_base_fmt + '.pb'


json_fn = osp.join(download_dir, 'labels.json')

model_dl = model_dl_fmt.format(factor, img_size, model_date)
model_pb = model_pb_fmt.format(factor, img_size)
    
if not tf.gfile.Exists( osp.join(checkpoints_dir, model_pb) ):

  download_and_uncompress_tarball(base_url, model_dl, download_dir)

  try:  
    meta_file = osp.join(download_dir, model_pb.replace('.pb', '.ckpt.meta'))
    if tf.gfile.Exists( meta_file ):
      print('Processing meta_file {}'.format(meta_file))
      freeze_mobilenet(meta_file, img_size, float(factor), num_classes=1001)
    else:
      print('Skipping meta file {}'.format(meta_file))
      pass
  except:
    print('Failed to process meta_file {}'.format(meta_file))
else:
  print('{} frozen model already exists -- skipping'.format(model_pb))