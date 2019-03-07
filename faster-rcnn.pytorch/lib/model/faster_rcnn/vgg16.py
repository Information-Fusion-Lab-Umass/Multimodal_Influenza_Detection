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
#import vgg16_modified
from torchvision.models.vgg import model_urls
from model.faster_rcnn.modified_vgg16 import vgg16 as vgg16_
class vgg16(_fasterRCNN):
  
  def __init__(self, classes, pretrained=False, class_agnostic=False):
    self.model_path = '/mnt/nfs/scratch1/shasvatmukes/model_weights/models/vgg16/pretrained_weights/vgg16-397923af.pth'
    #self.model_path = '/home/dghose/Project/Influenza_Detection/Code/Multimodal_Influenza_Detection/faster-rcnn.pytorch/models/vgg16/pretrained_weights/vgg16-397923af.pth'
    self.dout_base_model = 512
    self.pretrained = pretrained
    self.class_agnostic = class_agnostic

    _fasterRCNN.__init__(self, classes, class_agnostic)

  def _init_modules(self):
    ################------------------##########################
    model_urls['vgg16'] = model_urls['vgg16'].replace('https://', 'http://')
    pretrained = models.vgg16(pretrained=True)
    print(pretrained)
    #pretrained=models.vgg16()
    #print('printing pretrained')
    #print(pretrained)
    #pretrained_dict=torch.load(self.model_path)
    #print("original state dict")
    pretrained_dict = pretrained.state_dict()
    #print(pretrained_dict.keys())

    vgg = vgg16_()
    vgg_dict = vgg.state_dict()
    idx = 0
    #print("no max pool",vgg_dict.keys())

    for k, v in pretrained_dict.items():
        if int(k.split('.')[1]) > 22:
            break

        vgg_dict[k] = v
        #print("in loop", k)
        #print('vgg_dict[k]')
        #print(type(vgg_dict[k]))
  	#print('tyoe v')
	#print(type(v)) 
        #if int(k.split('.')[1]) > 22:
        #    break
    vgg_dict['features.23.weight'] = pretrained_dict['features.24.weight']
    vgg_dict['features.23.bias'] = pretrained_dict['features.24.bias']
    vgg_dict['features.25.weight'] = pretrained_dict['features.26.weight']
    vgg_dict['features.25.bias'] = pretrained_dict['features.26.bias']
    vgg_dict['features.27.weight'] = pretrained_dict['features.28.weight']
    vgg_dict['features.27.bias'] = pretrained_dict['features.28.bias']
    #print (vgg_dict['features.0.bias'] == pretrained_dict['features.0.bias'])
    #print('pretrained_dict before updation')
    #print(pretrained_dict.keys())
    #pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in vgg_dict}
    #print('pretrained_dict.keys()')
    #print(pretrained_dict.keys())
    #print('vgg_dict.keys()')
    #print(vgg_dict.keys())
    #vgg_dict.update(pretrained_dict) 
    vgg.load_state_dict(vgg_dict)
 

    ######################################-________###################################
    '''
    if self.pretrained:
        print("Loading pretrained weights from %s" %(self.model_path))
        state_dict = torch.load(self.model_path)
        vgg.load_state_dict({k:v for k,v in state_dict.items() if k in vgg.state_dict()})
    '''
    #vgg.classifier = nn.Sequential(*list(pretrained.classifier._modules.values())[:-1])
    #print(pretrained.features)
    part_one = list(pretrained.features.children())[0:23]
    part_two = list(pretrained.features.children())[24:30]
    part_one.extend(part_two)
    self.RCNN_base  = nn.Sequential(*part_one)
    #self.RCNN_base = nn.Sequential(*list(pretrained.features.children())[:-1])
    print(self.RCNN_base)
    class_part_one = list(pretrained.classifier.children())[0:2]
    class_part_two = list(pretrained.classifier.children())[3:5] 
    class_part_one.extend(class_part_two)
    self.RCNN_top  = nn.Sequential(*class_part_one)

    #self.RCNN_top=nn.Sequential(*list(pretrained.classifier.children())[:-1])

    print(self.RCNN_top)
    # not using the last maxpool layer
    ##############removing 4th max pool ---starts here##################################
    #print(vgg.features)
    #print(vgg.features._modules)
    #self.RCNN_base = nn.Sequential(*list(pretrained.features._modules.values())[:-1])##original
    #print('priting original vgg without 5th pool')
    #print(self.RCNN_base)
    
    #self.RCNN_base=vgg.features
    #self.RCNN_base=pretrained.features
    #self.RCNN_base_before_pool = nn.Sequential(*list(vgg.features._modules.values())[:23])# as per the count 24the layer is 4th max pool=23rd index[0 based indexing]
    #self.RCNN_base_after_pool = nn.Sequential(*list(vgg.features._modules.values())[24:-1])
    #print(self.RCNN_base_before_pool)
    #print(self.RCNN_base_after_pool)
    
    '''
    list1 = list(vgg.features._modules.values())[:23] 
    list1.extend(list(vgg.features._modules.values())[24:-1])
    self.RCNN_base_new = nn.Sequential(*list1)
    print(self.RCNN_base_new)
    '''

    
    #self.RCNN_new=nn.Sequential(
    #print(summary(self.RCNN_base_before_pool,(3,512,640)))
    #print(summary(self.RCNN_base__pool,(3,512,512)))

    ###################changes end here#####################################################
    ###########due to above change########################
    # Fix the layers before conv3:
    for layer in range(10):
      for p in self.RCNN_base[layer].parameters(): p.requires_grad = False

    # self.RCNN_base = _RCNN_base(vgg.features, self.classes, self.dout_base_model)
    #for layer in range(10):
    #  for p in self.RCNN_base_new[layer].parameters(): p.requires_grad = False
    #######################################################3
    #self.RCNN_top = vgg.classifier

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

