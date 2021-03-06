# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Xinlei Chen
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
import torchvision.models as models
from model.faster_rcnn.faster_rcnn import _fasterRCNN
import pdb
from torchsummary import summary

class vgg16(_fasterRCNN):
  def __init__(self, classes, pretrained=False, class_agnostic=False):
    #self.model_path = 'models/vgg16/pretrained_weights/vgg16-397923af.pth'
    self.model_path ='/mnt/nfs/scratch1/shasvatmukes/model_weights/models/vgg16/pretrained_weights/vgg16-397923af.pth'
    self.dout_base_model = 512
    self.pretrained = pretrained
    self.class_agnostic = class_agnostic

    _fasterRCNN.__init__(self, classes, class_agnostic)

  def _init_modules(self):
    vgg = models.vgg16().to('cuda')
    if self.pretrained:
	print('pretrained')
        print("Loading pretrained weights from %s" %(self.model_path))
        state_dict = torch.load(self.model_path)
        vgg.load_state_dict({k:v for k,v in state_dict.items() if k in vgg.state_dict()})

    vgg.classifier = nn.Sequential(*list(vgg.classifier._modules.values())[:-1])
    #for child in vgg.children():	
    #vgg.features=nn.Sequential(*list(vgg.features._modules.values())[:-1])
    self.RCNN_base = nn.Sequential(*list(vgg.features._modules.values())[:-1])

    #self.RCNN_base_before_pool = vgg.features[0:23]
    #self.RCNN_base_after_pool = vgg.features[24:]
    # Fix the layers before conv3:
    ##print(len(*list(child[:])))
    #print(len(self.RCNN_base_before_pool))
    #print(summary(self.RCNN_base_before_pool,(3,512,512)))
    for layer in range(10):
        for p in self.RCNN_base[layer].parameters(): p.requires_grad = False

    # self.RCNN_base = _RCNN_base(vgg.features, self.classes, self.dout_base_model)

    self.RCNN_top = vgg.classifier

    # not using the last maxpool layer
    self.RCNN_cls_score = nn.Linear(4096, self.n_classes)

    if self.class_agnostic:
        self.RCNN_bbox_pred = nn.Linear(4096, 4)
    else:
        self.RCNN_bbox_pred = nn.Linear(4096, 4 * self.n_classes)      

  def _head_to_tail(self, pool5):
    
    pool5_flat = pool5.view(pool5.size(0), -1)
    fc7 = self.RCNN_top(pool5_flat)

    return fc7

