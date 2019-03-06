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


def voc_eval(detpath,
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
  cachefile = os.path.join(cachedir, '%s_annots.pkl' % imagesetfile)
  print(cachefile)
  # read list of images
  with open(imagesetfile, 'r') as f:
    lines = f.readlines()
  imagenames = [x.strip() for x in lines]

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
    #recs is a dict with key:imagename, val:list of dicts[each dict in the list having 3 keys-- bbox, diff , name]  
    #recs[imagename] is a list
    R = [obj for obj in recs[imagename]]
    # R is a list of dicts
    bbox=np.array([x['bbox'] for x in R])# x is a dict[each element of list is a dict]
    difficult = np.array([x['difficult'] for x in R]).astype(np.bool)
    det = [False] * len(R)
    npos=npos+len(R)
    # mapping: imagename: array of all bbox in that image
    class_recs[imagename] = {'bbox': bbox,
                             'difficult': difficult,
                             'det': det}

  # read dets
  detfile = detpath.format(classname)
  with open(detfile, 'r') as f:
    lines = f.readlines()
    print("detections")
    #print(lines)
  splitlines = [x.strip().split(' ') for x in lines]
  image_ids = [x[0] for x in splitlines]
  confidence = np.array([float(x[1]) for x in splitlines])
  BB = np.array([[float(z) for z in x[2:]] for x in splitlines])
  
  nd = len(image_ids)
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
      # R is a dict
      bb = BB[d, :].astype(float)
      ovmax = -np.inf
      # R['bbox'] is an numoy array of all the bbox in the given image
      BBGT = R['bbox'].astype(float)
      #print ("BBGT")
      #print (BBGT)

      if BBGT.size > 0:
  
        ixmin = np.maximum(BBGT[:, 0], bb[0])
        #print("ixmin")
        #print(ixmin)
        iymin = np.maximum(BBGT[:, 1], bb[1])
        #print("iymin")
        #print(iymin)
        ixmax = np.minimum(BBGT[:, 2], bb[2])
        #print("ixmax")
        #print(ixmax)
        iymax = np.minimum(BBGT[:, 3], bb[3])
        #print("iymax")
        #print(iymax)
        iw = np.maximum(np.absolute(ixmax - ixmin + 1.), 0.)
        #print ("iw")
        #print (iw)
        ih = np.maximum(np.absolute(iymax - iymin + 1.), 0.)
        #print ("ih")
        #print(ih)
        inters = iw * ih
        #print("intersection")
        #print(inters)

        # union
        uni = ((bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) +
               (BBGT[:, 2] - BBGT[:, 0] + 1.) *
               (BBGT[:, 3] - BBGT[:, 1] + 1.) - inters)
        #print("union")
        #print(uni)
        overlaps = inters / uni
        #print("overlaps")
        #print(overlaps)
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
  print(tp.shape)
  tp = np.cumsum(tp)
  print ("true positive")
  print (truep)
 
  #print('npos')
  #print(npos)
  #temp=npos-tp
  #print('false negative')
  #print(temp[:100])
  #print()
  #print()
  rec = tp / float(npos)
  # avoid divide by zero in case the first detection matches a difficult
  # ground truth
  print("recall")
  print (rec)
  prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
  print("prec")
  print(prec)
  ap = voc_ap(rec, prec, use_07_metric)
  print("ap")
  print(ap)

  return rec, prec, ap


def voc_eval_miss_rate(detpath,
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
  print(cachefile)
  # read list of images
  with open(imagesetfile, 'r') as f:
    lines = f.readlines()
  imagenames = [x.strip() for x in lines]
  print(imagenames)
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
    R = [obj for obj in recs[imagename]]
    bbox = np.array([x['bbox'] for x in R])
    #difficult = np.array([x['difficult'] for x in R]).astype(np.bool)
    det = [False] * len(R)
    npos = npos + len(R)
    #print("npos")
    #print(npos)
    class_recs[imagename] = {'bbox': bbox,
                             'det': det}

  # read dets
  detfile = detpath.format(classname)
  with open(detfile, 'r') as f:
    lines = f.readlines()
    print("detections")
    print(lines)
  splitlines = [x.strip().split(' ') for x in lines]
  image_ids = [x[0] for x in splitlines]
  confidence = np.array([float(x[1]) for x in splitlines])
  BB = np.array([[float(z) for z in x[2:]] for x in splitlines])
  fppi={}# a dict mapping image name to the number of false positives the image
   
  nd = len(image_ids)
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
      print ("BBGT")
      print (BBGT)

      if BBGT.size > 0:
        # compute overlaps
        # intersection
       # print("inside if")  
        ixmin = np.maximum(BBGT[:, 0], bb[0])
        #print("ixmin")
        #print(ixmin)
        iymin = np.maximum(BBGT[:, 1], bb[1])
        #print("iymin")
        #print(iymin)
        ixmax = np.minimum(BBGT[:, 2], bb[2])
        #print("ixmax")
        #print(ixmax)
        iymax = np.minimum(BBGT[:, 3], bb[3])
        #print("iymax")
        #print(iymax)
        iw = np.maximum(np.absolute(ixmax - ixmin + 1.), 0.)
        #print ("iw")
        #print (iw)
        ih = np.maximum(np.absolute(iymax - iymin + 1.), 0.)
        #print ("ih")
        #print(ih)
        inters = iw * ih
        print("intersection")
        print(inters)
    
        # union
        uni = ((bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) +
               (BBGT[:, 2] - BBGT[:, 0] + 1.) *
               (BBGT[:, 3] - BBGT[:, 1] + 1.) - inters)
        #print("union")
        #print(uni)
        overlaps = inters / uni
        #print("overlaps")
        #print(overlaps)
        ovmax = np.max(overlaps)
        print("ovmax")
        print(ovmax)
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
  #fp = np.cumsum(fp)
  fp=np.sum(fp)  
  #tp = np.cumsum(tp)
  print("false positive")
  print(fp)
  tp=np.sum(tp)  
  print ("true positive")
  print(tp)
  print('npos')
  print(npos)
  rec = tp / float(npos)

  print("recall")
  print (rec)

  prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
  print("prec")
  print(prec)

  #ap = voc_ap(rec, prec, use_07_metric)
  #print("ap")
  #print(ap)

  return rec, prec


def voc_eval_thesis(imageset,detpath,
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
        # intersection
       # print("inside if")  
        ixmin = np.maximum(BBGT[:, 0], bb[0])
        #print("ixmin")
        #print(ixmin)
        iymin = np.maximum(BBGT[:, 1], bb[1])
        #print("iymin")
        #print(iymin)
        ixmax = np.minimum(BBGT[:, 2], bb[2])
        #print("ixmax")
        #print(ixmax)
        iymax = np.minimum(BBGT[:, 3], bb[3])
        #print("iymax")
        #print(iymax)
        iw = np.maximum(np.absolute(ixmax - ixmin + 1.), 0.)
        #print ("iw")
        #print (iw)
        ih = np.maximum(np.absolute(iymax - iymin + 1.), 0.)
        #print ("ih")
        #print(ih)
        inters = iw * ih
        #print("intersection")
        #print(inters)
    
        # union
        uni = ((bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) +
               (BBGT[:, 2] - BBGT[:, 0] + 1.) *
               (BBGT[:, 3] - BBGT[:, 1] + 1.) - inters)
        #print("union")
        #print(uni)
        overlaps = inters / uni
        #print("overlaps")
        #print(overlaps)
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
  #print("false positive")
  #print(fp)
  #tp=np.sum(tp)  
  #print ("true positive")
  #print(tp)
  #print('npos')
  #print(npos)
  rec = tp / float(npos)

  #print("recall")
  #print (rec)

  prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
  #print("prec")
  #print(prec)

  ap = voc_ap(rec, prec, use_07_metric)
  print("ap")
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

def voc_eval_fppi_LAMR(detpath,
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
  print(cachefile)
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
  #nd = len(image_ids)

  num_thresh=10
  #tp = np.zeros(num_thresh,nd)
  #fp = np.zeros(num_thresh,nd)
  #nd = len(image_ids)

  for imagename in imagenames:
    R = [obj for obj in recs[imagename] if obj['name'] == classname]
    bbox = np.array([x['bbox'] for x in R])
    difficult = np.array([x['difficult'] for x in R]).astype(np.bool)
    det = [False] *(num_thresh* len(R))#so here we are doing 10 thresholds for every Ground Truth Bounding Box]
    npos = npos + sum(~difficult)
    #print("npos")i
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
  print('length of image ids')
  print(nd)
  #num_scales=10
  tp = np.zeros((num_thresh,nd))
  fp = np.zeros((num_thresh,nd))

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
        # intersection
       # print("inside if")  
        ixmin = np.maximum(BBGT[:, 0], bb[0])
        #print("ixmin")
        #print(ixmin)
        iymin = np.maximum(BBGT[:, 1], bb[1])
        #print("iymin")
        #print(iymin)
        ixmax = np.minimum(BBGT[:, 2], bb[2])
        #print("ixmax")
        #print(ixmax)
        iymax = np.minimum(BBGT[:, 3], bb[3])
        #print("iymax")
        #print(iymax)
        iw = np.maximum(np.absolute(ixmax - ixmin + 1.), 0.)
        #print ("iw")
        #print (iw)
        ih = np.maximum(np.absolute(iymax - iymin + 1.), 0.)
        #print ("ih")
        #print(ih)
        inters = iw * ih
        #print("intersection")
        #print(inters)
    
        # union
        uni = ((bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) +
               (BBGT[:, 2] - BBGT[:, 0] + 1.) *
               (BBGT[:, 3] - BBGT[:, 1] + 1.) - inters)
        #print("union")
        #print(uni)
        overlaps = inters / uni
        #print("overlaps")
        #print(overlaps)
        ovmax = np.max(overlaps)
        #print("ovmax")
        #print(ovmax)
        jmax = np.argmax(overlaps)
        #print("jmax")
        #print(jmax)
      counter=0
      increment=len(R['det'])/float(10)
      #print(increment)
      increment=int(increment)
      for i in range(num_thresh):
	ovthresh=i/float(num_thresh)#0.0 to 0.9
        #print(counter,jmax,counter+jmax) 
        if ovmax > ovthresh:
            if not R['det'][counter+jmax]:
              tp[i][d] = 1.
              R['det'][counter+jmax] = 1
            else:
              fp[i][d] = 1.
        else:
          fp[i][d] = 1.
	counter=counter+increment

  # compute precision recall
  #fp = np.cumsum(fp)
  #print(tp.shape)
  #print(fp.shape)
  all_fp=np.sum(fp,axis=1)
  all_tp=np.sum(tp,axis=1)
  #print(fp)
  for i in range(num_thresh):
    #print(i)
    fp=all_fp[i]
    #fp=np.sum(fp) 
    #tp = np.cumsum(tp)
    print("false positive with ovthresh",i/float(num_thresh))
    print(fp)
    tp=all_tp[i]  
    #tp=np.sum(tp)
    print ("true positive with ovthresh",i/float(num_thresh))
    print(tp)
    print('npos')
    print(npos)
    print('fppi')
    fppi=fp/float(npos)
    print(fppi)
    rec = tp / float(npos)
    miss_rate=1-rec
    print("recall")
    print (rec)
	
    print('miss rate with thresh',i/float(num_thresh))
    print(miss_rate)
    print('miss rate vs fppi')
    print(miss_rate/float(np.maximum(fppi, np.finfo(np.float64).eps)))
		
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    print("prec")
    print(prec)
    print('#################################################################')


  return rec, prec
