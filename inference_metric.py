import time
from matplotlib import pyplot as plt
import numpy as np
import mxnet as mx
from mxnet import autograd, gluon
import gluoncv as gcv
from gluoncv.utils import download, viz
from gluoncv.data.batchify import Tuple, Stack, Pad
from gluoncv.utils.metrics.voc_detection import VOC07MApMetric
from gluoncv.data.transforms.presets.ssd import SSDDefaultValTransform

## network setting

classes = ['pikachu']  # only one foreground class here
# network = 'ssd_512_resnet50_v1_custom'              # SSD
# network = 'ssd_512_vgg16_atrous_custom'             # VGG16
network = 'ssd_512_mobilenet1.0_voc'                # MobileNet

# net = gcv.model_zoo.get_model(network, classes=classes, pretrained_base=False)
net = gcv.model_zoo.get_model(network, pretrained=True)

net.load_parameters(f'{network}_pikachu.params')
print('-'*100)
print(type(net.params))

## validation dataset

url = 'https://apache-mxnet.s3-accelerate.amazonaws.com/gluon/dataset/pikachu/val.rec'
idx_url = 'https://apache-mxnet.s3-accelerate.amazonaws.com/gluon/dataset/pikachu/val.idx'
download(url, path='pikachu_val.rec', overwrite=False)
download(idx_url, path='pikachu_val.idx', overwrite=False)

val_dataset = gcv.data.RecordFileDetection('pikachu_val.rec')
classes = ['pikachu']

# ## visualzie validation dataset

# image, label = val_dataset[5]
# print('label:', label)
# # display image and label
# ax = viz.plot_bbox(image, bboxes=label[:, :4], labels=label[:, 4:5], class_names=classes)
# plt.show()
# print(len(val_dataset))

## dataloader

def get_dataloader(val_dataset, data_shape, batch_size, num_workers):
    """Get dataloader."""
    width, height = data_shape, data_shape
    batchify_fn = Tuple(Stack(), Pad(pad_val=-1))
    val_loader = gluon.data.DataLoader(
        val_dataset.transform(SSDDefaultValTransform(width, height)), batchify_fn=batchify_fn,
        batch_size=batch_size, shuffle=False, last_batch='rollover', num_workers=num_workers)
    return val_loader


val_data = get_dataloader(
    val_dataset, data_shape=512, batch_size=1, num_workers=0)
# classes = val_dataset.classes  # class names

try:
    a = mx.nd.zeros((1,), ctx=mx.gpu(0))
    ctx = [mx.gpu(0)]
except:
    ctx = [mx.cpu()]

val_metric = VOC07MApMetric(iou_thresh=0.5, class_names=classes)


net.collect_params().reset_ctx(ctx)
val_metric.reset()
net.hybridize(static_alloc=True, static_shape=True)

start = time.time()
total = 0
for i, batch in enumerate(val_data):
    batch_size = batch[0].shape[0]
    total += batch_size
    data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0, even_split=False)
    label = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0, even_split=False)

    det_bboxes = []
    det_ids = []
    det_scores = []
    gt_bboxes = []
    gt_ids = []
    gt_difficults = []

    for x, y in zip(data, label):
        ids, scores, bboxes = net(x)
        det_ids.append(ids)
        det_scores.append(scores)
        # clip to image size
        det_bboxes.append(bboxes.clip(0, batch[0].shape[2]))
        # split ground truths
        gt_ids.append(y.slice_axis(axis=-1, begin=4, end=5))
        gt_bboxes.append(y.slice_axis(axis=-1, begin=0, end=4))
        gt_difficults.append(y.slice_axis(axis=-1, begin=5, end=6) if y.shape[-1] > 5 else None)
    
    val_metric.update(det_bboxes, det_ids, det_scores, gt_bboxes, gt_ids, gt_difficults)

end = time.time()
speed = total / (end - start)
print('Throughput is %f img/sec.'% speed)
print(val_metric.get())
