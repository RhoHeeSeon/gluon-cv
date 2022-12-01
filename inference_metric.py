import time
from matplotlib import pyplot as plt
import numpy as np
import mxnet as mx
from mxnet import autograd, gluon
import gluoncv as gcv
from gluoncv.utils import download, viz



classes = ['pikachu']  # only one foreground class here
network = 'ssd_512_resnet50_v1_custom'              # SSD

net = gcv.model_zoo.get_model(network, classes=classes, pretrained_base=False)
net.load_parameters(f'{network}_pikachu.params')



url = 'https://apache-mxnet.s3-accelerate.amazonaws.com/gluon/dataset/pikachu/val.rec'
idx_url = 'https://apache-mxnet.s3-accelerate.amazonaws.com/gluon/dataset/pikachu/val.idx'
download(url, path='pikachu_val.rec', overwrite=False)
download(idx_url, path='pikachu_val.idx', overwrite=False)

dataset = gcv.data.RecordFileDetection('pikachu_val.rec')

image, label = dataset[5]
print('label:', label)
# display image and label
ax = viz.plot_bbox(image, bboxes=label[:, :4], labels=label[:, 4:5], class_names=classes)
plt.show()
print(len(dataset))

def get_dataloader(net, train_dataset, data_shape, batch_size, num_workers):
    from gluoncv.data.batchify import Tuple, Stack, Pad
    from gluoncv.data.transforms.presets.ssd import SSDDefaultTrainTransform
    width, height = data_shape, data_shape
    # use fake data to generate fixed anchors for target generation
    with autograd.train_mode():
        _, _, anchors = net(mx.nd.zeros((1, 3, height, width)))
    batchify_fn = Tuple(Stack(), Stack(), Stack())  # stack image, cls_targets, box_targets
    train_loader = gluon.data.DataLoader(
        train_dataset.transform(SSDDefaultTrainTransform(width, height, anchors)),
        batch_size, True, batchify_fn=batchify_fn, last_batch='rollover', num_workers=num_workers)
    return train_loader



eval_data = get_dataloader(net, dataset, 512, 16, 0)


try:
    a = mx.nd.zeros((1,), ctx=mx.gpu(0))
    ctx = [mx.gpu(0)]
except:
    ctx = [mx.cpu()]

mbox_loss = gcv.loss.SSDMultiBoxLoss()
ce_metric = mx.metric.Loss('CrossEntropy')
smoothl1_metric = mx.metric.Loss('SmoothL1')

ce_metric.reset()
smoothl1_metric.reset()
net.hybridize(static_alloc=True, static_shape=True)
for i, batch in enumerate(eval_data):
    batch_size = batch[0].shape[0]
    data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0)
    cls_targets = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0)
    box_targets = gluon.utils.split_and_load(batch[2], ctx_list=ctx, batch_axis=0)
    with autograd.record():
        cls_preds = []
        box_preds = []
        for x in data:
            cls_pred, box_pred, _ = net(x)
            cls_preds.append(cls_pred)
            box_preds.append(box_pred)
        sum_loss, cls_loss, box_loss = mbox_loss(
            cls_preds, box_preds, cls_targets, box_targets)
        autograd.backward(sum_loss)
    # since we have already normalized the loss, we don't want to normalize
    # by batch-size anymore
    ce_metric.update(0, [l * batch_size for l in cls_loss])
    smoothl1_metric.update(0, [l * batch_size for l in box_loss])
    name1, loss1 = ce_metric.get()
    name2, loss2 = smoothl1_metric.get()
    print('[', batch, '] loss1', loss1, 'loss2', loss2)