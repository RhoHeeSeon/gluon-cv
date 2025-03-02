import time
from matplotlib import pyplot as plt
import numpy as np
import mxnet as mx
from mxnet import autograd, gluon
import gluoncv as gcv
from gluoncv.utils import download, viz


classes = ['pikachu']

network = 'ssd_512_resnet50_v1_custom'

net = gcv.model_zoo.get_model(network, classes=classes, pretrained_base=False)
net.load_parameters(f'{network}_pikachu.params')

x, image = gcv.data.transforms.presets.ssd.load_test('src/pikachu_test.jpg', 512)
cid, score, bbox = net(x)

ax = viz.plot_bbox(image, bbox[0], score[0], cid[0], class_names=classes)
plt.show()