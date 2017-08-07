# Ignore debugging logs
from os import environ
environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

from tensorflow.python.tools import freeze_graph

# Set net name here
net_name = 'inception_resnet_v2'

input_graph_path = net_name+"/graph.pb"
input_saver_def_path = ""
input_binary = False
checkpoint_path = net_name+"/model.ckpt"
output_node_names = "top_k"
restore_op_name = "save/restore_all"
filename_tensor_name = "save/Const:0"
output_graph_path = net_name+"/frozen_graph.pb"
clear_devices = False

freeze_graph.freeze_graph(input_graph_path, input_saver_def_path, input_binary, checkpoint_path, output_node_names, restore_op_name, filename_tensor_name, output_graph_path, clear_devices, "")
