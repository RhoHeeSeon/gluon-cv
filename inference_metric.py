import time
from matplotlib import pyplot as plt
import numpy as np
import mxnet as mx
from mxnet import autograd, gluon
import gluoncv as gcv
from gluoncv.utils import download, viz



network = 'ssd_512_resnet50_v1_custom'              # SSD
# network = 'faster_rcnn_resnet50_v1b_custom'         # Faster RCNN
# network = 'yolo3_darknet53_custom'                  # YOLOv3

classes = ['pikachu']  # only one foreground class here
net = gcv.model_zoo.get_model(network, classes=classes, pretrained_base=False)
net.load_parameters(f'{network}_pikachu.params')


