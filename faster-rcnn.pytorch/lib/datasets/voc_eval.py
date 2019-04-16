#t/er R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Bharath Hariharan
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import xml.etree.ElementTree as ET
import os
import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import math
from matplotlib.ticker import FormatStrFormatter

def parse_rec(filename):

  """ Parse a PASCAL VOC xml file """
  '''
  tree = ET.parse(filename)
  objects = []
  for obj in tree.findall('object'):
    obj_struct = {}
    obj_struct['name'] = obj.find('name').text
    obj_struct['pose'] = obj.find('pose').text
    obj_struct['truncated'] = int(obj.find('truncated').text)
    obj_struct['difficult'] = int(obj.find('difficult').text)
    bbox = obj.find('bndbox')
    obj_struct['bbox'] = [int(bbox.find('xmin').text),
                          int(bbox.find('ymin').text),
                          int(bbox.find('xmax').text),
                          int(bbox.find('ymax').text)]
    objects.append(obj_struct)
  '''
  objects = []
  with open(filename, 'rw+') as fo:
    lines = fo.readlines()
    for i in range(1, len(lines)):
        data = lines[i].split(' ')
        #print("inside parse_rec")
        #print(data)
        data[3] =str(int(data[3]) +int(data[1]))
        data[4] = str(int(data[4]) + int(data[2]))
        obj_struct = {}
        obj_struct['name'] = data[0]
        obj_struct['bbox'] = [data[1], data[2], data[3], data[4]]
        obj_struct['difficult'] = data[5]
        objects.append(obj_struct)
  return objects


def voc_ap(rec, prec, use_07_metric=False):
  """ ap = voc_ap(rec, prec, [use_07_metric])
  Compute VOC AP given precision and recall.
  If use_07_metric is true, uses the
  VOC 07 11 point method (default:False).
  """
  if use_07_metric:
    # 11 point metric
    ap = 0.
    for t in np.arange(0., 1.1, 0.1):
      if np.sum(rec >= t) == 0:
        p = 0
      else:
        p = np.max(prec[rec >= t])
      ap = ap + p / 11.
  else:
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.], rec, [1.]))
    mpre = np.concatenate(([0.], prec, [0.]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
      mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
  return ap



def voc_eval(imageset,detpath,
             annopath,
             imagesetfile,
             classname,
             cachedir,
             ovthresh=0.5,#0.5
             use_07_metric=False):
  """rec, prec, ap = voc_eval(detpath,
                              annopath,
                              imagesetfile,
                              classname,
                              [ovthresh],
                              [use_07_metric])
  Top level function that does the PASCAL VOC evaluation.
  detpath: Path to detections
      detpath.format(classname) should produce the detection results file.
  annopath: Path to annotations
      annopath.format(imagename) should be the xml annotations file.
  imagesetfile: Text file containing the list of images, one image per line.
  classname: Category name (duh)
  cachedir: Directory for caching the annotations
  [ovthresh]: Overlap threshold (default = 0.5)
  [use_07_metric]: Whether to use VOC07's 11 point AP computation
      (default False)
  """
  # assumes detections are in detpath.format(classname)
  # assumes annotations are in annopath.format(imagename)
  # assumes imagesetfile is a text file with each line an image name
  # cachedir caches the annotations in a pickle file

  # first load gt
  if not os.path.isdir(cachedir):
    os.mkdir(cachedir)
  annotations='annotations'
  cachefile = os.path.join(cachedir, '%s_annots.pkl' % annotations)
  #print(cachefile)
  # read list of images
  with open(imagesetfile, 'r') as f:
    lines = f.readlines()
  imagenames = [x.strip() for x in lines]
  #print(imagenames)
  if not os.path.isfile(cachefile):
    # load annotations
    recs = {}
    for i, imagename in enumerate(imagenames):
      recs[imagename] = parse_rec(annopath.format(imagename))
      if i % 100 == 0:
        print('Reading annotation for {:d}/{:d}'.format(
          i + 1, len(imagenames)))
    # save
    print('Saving cached annotations to {:s}'.format(cachefile))
    with open(cachefile, 'wb') as f:
      pickle.dump(recs, f)
  else:
    # load
    with open(cachefile, 'rb') as f:
      try:
        recs = pickle.load(f)
      except:
        recs = pickle.load(f, encoding='bytes')

  # extract gt objects for this class
  class_recs = {}
  npos = 0 #0.0001#0
  for imagename in imagenames:
    R = [obj for obj in recs[imagename] if obj['name'] == classname]
    bbox = np.array([x['bbox'] for x in R])
    difficult = np.array([x['difficult'] for x in R]).astype(np.bool)
    det = [False] * len(R)
    npos = npos + sum(~difficult)
    #print("npos")
    #print(npos)
    class_recs[imagename] = {'bbox': bbox,
                              'difficult':difficult,
                             'det': det}

  # read dets
  detfile = detpath.format(classname)
  with open(detfile, 'r') as f:
    lines = f.readlines()
    #print("detections")
    #print(lines)
  splitlines = [x.strip().split(' ') for x in lines]
  image_ids = [x[0] for x in splitlines]
  confidence = np.array([float(x[1]) for x in splitlines])
  BB = np.array([[float(z) for z in x[2:]] for x in splitlines])
  fppi={}# a dict mapping image name to the number of false positives the image
   
  nd = len(image_ids)
  #print('nd')
  #print(nd)
  tp = np.zeros(nd)
  fp = np.zeros(nd)

  if BB.shape[0] > 0:
    # sort by confidence
    sorted_ind = np.argsort(-confidence)
    sorted_scores = np.sort(-confidence)
    BB = BB[sorted_ind, :]
    image_ids = [image_ids[x] for x in sorted_ind]

    # go down dets and mark TPs and FPs
    for d in range(nd):
      R = class_recs[image_ids[d]]
      bb = BB[d, :].astype(float)
      ovmax = -np.inf
      BBGT = R['bbox'].astype(float)
      #print ("BBGT")
      #print (BBGT)

      if BBGT.size > 0:
        # compute overlaps
        ixmin = np.maximum(BBGT[:, 0], bb[0])
        iymin = np.maximum(BBGT[:, 1], bb[1])
        ixmax = np.minimum(BBGT[:, 2], bb[2])
        iymax = np.minimum(BBGT[:, 3], bb[3])
        iw = np.maximum(np.absolute(ixmax - ixmin + 1.), 0.)
        ih = np.maximum(np.absolute(iymax - iymin + 1.), 0.)
        inters = iw * ih
    
        # union
        uni = ((bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) +
               (BBGT[:, 2] - BBGT[:, 0] + 1.) *
               (BBGT[:, 3] - BBGT[:, 1] + 1.) - inters)
        overlaps = inters / uni
        ovmax = np.max(overlaps)
        #print("ovmax")
        #print(ovmax)
        jmax = np.argmax(overlaps)
        #print("jmax")
        #print(jmax)
        

         	
      if ovmax > ovthresh:
        if not R['det'][jmax]:
           tp[d] = 1.
           R['det'][jmax] = 1
        else:
           fp[d] = 1.
      else:
        fp[d] = 1.
  # compute precision recall
  fp = np.cumsum(fp)
  #fp=np.sum(fp)  
  tp = np.cumsum(tp)
  rec = tp / float(npos)


  prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)

  ap = voc_ap(rec, prec, use_07_metric)
  print("MAP")
  print(ap)
  
  lamr, mr, fppi = log_average_miss_rate(imageset,rec, fp, nd)

  return rec, prec

def log_average_miss_rate(imageset,recall, fp_cumsum, num_images):
    """
        log-average miss rate:
            Calculated by averaging miss rates at 9 evenly spaced FPPI points
            between 10e-2 and 10e0, in log-space.

        output:
                lamr | log-average miss rate
                mr | miss rate
                fppi | false positives per image

        references:
            [1] Dollar, Piotr, et al. "Pedestrian Detection: An Evaluation of the
               State of the Art." Pattern Analysis and Machine Intelligence, IEEE
               Transactions on 34.4 (2012): 743 - 761.
    """

    # if there were no detections of that class
    if recall.size == 0:
        lamr = 0
        mr = 1
        fppi = 0
        return lamr, mr, fppi

    fppi = fp_cumsum / float(num_images)
    mr = (1 - recall)

    fppi_tmp = np.insert(fppi, 0, -1.0)
    mr_tmp = np.insert(mr, 0, 1.0)

    # Use 9 evenly spaced reference points in log-space
    ref = np.logspace(-2.0, 0.0, num = 9)
    #print(ref)
    ref_tmp=np.copy(ref)
    for i, ref_i in enumerate(ref):
        # np.where() will always find at least 1 index, since min(ref) = 0.01 and min(fppi_tmp) = -1.0
        j = np.where(fppi_tmp <= ref_i)[-1][-1]
        ref[i] = mr_tmp[j]

    # log(0) is undefined, so we use the np.maximum(1e-10, ref)
    #print(ref)
    np.save('/mnt/nfs/scratch1/dghose/Kaist_test_30X/fppi_mr_plots/vgg16/'+imageset+'.npy',ref)
    plt.xscale('log')
    plt.yscale('log')
    plt.title('Miss rate vs FPPI on log-log scale')
    plt.xlabel('False Positives Per Image')
    plt.ylabel('Miss Rate')
    #print(ref_tmp)
    #print(ref)
    plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

    plt.plot(ref_tmp,ref)
    plt.savefig('/mnt/nfs/scratch1/dghose/Kaist_test_30X/fppi_mr_plots/vgg16/'+imageset+'.png')
    lamr = math.exp(np.mean(np.log(np.maximum(1e-10, ref))))
    print('lamr')
    print(lamr)	
    return lamr, mr, fppi

