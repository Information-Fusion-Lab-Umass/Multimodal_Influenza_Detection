#t R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

import datasets.kaist_fusion
import os
from datasets.imdb import imdb
import xml.dom.minidom as minidom
import numpy as np
import scipy.sparse
import scipy.io as sio
import model.utils.cython_bbox
import cPickle
import subprocess
from model.utils.config import cfg
from .voc_eval import voc_eval
from .voc_eval import voc_eval_miss_rate

class kaist_thermal(imdb):
    def __init__(self, image_set, devkit_path='/home/dghose/Project/Influenza_Detection/Data/KAIST/Train/combined_train'):
        
        imdb.__init__(self, image_set)  
        '''
         image_set: ['combined_train','combined_test','day_train','day_test','night_train','night_test', \
         'train_combined_salient_ir','test_combined_salient_ir',
         'train_subset','salient_combined_train','salient_combined_test']
        '''
        self._image_set = image_set
        self._devkit_path = self._get_default_path()#mnt/nfs/scratch1/dghose/Kaist/
        self._data_path = (self._devkit_path)
        self._classes = ('__background__', # always index 0
                         'person')
        self._class_to_ind = dict(zip(self.classes, xrange(self.num_classes)))
        self._image_ext = '.jpg'
        self._image_index = self._load_image_set_index()
        # Default to roidb handler
        self._roidb_handler = self.selective_search_roidb
	    # PASCAL specific config options
        self.config = {'cleanup'     : True,
                       'use_salt'    : True,
                       'use_diff'    : False,
                       'matlab_eval' : False,
                       'rpn_file'    : None,
                       'min_size'    : 2}

        assert os.path.exists(self._devkit_path), \
                'VOCdevkit path does not exist: {}'.format(self._devkit_path)
        assert os.path.exists(self._data_path), \
                'Path does not exist: {}'.format(self._data_path)

    def image_path_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        return self.image_path_from_index(self._image_index[i])

    def image_path_from_index(self, index):
        """
        Construct an image path from the image's "index" identifier.
        """
        #print(index)
        image_path=(self._data_path +'/'+ self._image_set + '/' + index + self._image_ext) #train
        #print(image_path)
        return image_path

    def _load_image_set_index(self):
        """
        Load the indexes listed in this dataset's image set file.
        """
        image_set_file =(self._data_path+ '/Imagesetfiles/' + self._image_set+ '_imagesetfile.txt')
	assert os.path.exists(image_set_file), \
                'Path does not exist: {}'.format(image_set_file)
        with open(image_set_file) as f:
            image_index = [x.strip() for x in f.readlines()]
        #print(image_index)
        return image_index


    def _get_default_path(self):
        """
        Return the default path where kaist dataset is expected to be installed.
        """
        return (cfg.DATA_DIR)

    def gt_roidb(self):
        """
        Return the database of ground-truth regions of interest.

        This function loads/saves from/to a cache file to speed up future calls.
        """
        cache_file =(self.cache_path + '/' +self.name +  '_gt_roidb.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = cPickle.load(fid)   
            print '{} gt roidb loaded from {}'.format(self.name, cache_file)
            return roidb

        gt_roidb = [self._load_revised_annotation(index)
                    for index in self.image_index]
        with open(cache_file, 'wb') as fid:
            cPickle.dump(gt_roidb, fid, cPickle.HIGHEST_PROTOCOL)
        print 'wrote gt roidb to {}'.format(cache_file)

        return gt_roidb

    def selective_search_roidb(self):
        """
        Return the database of selective search regions of interest.
        Ground-truth ROIs are also included.

        This function loads/saves from/to a cache file to speed up future calls.
        """
        cache_file = (self.cache_path,
                                  self.name + '_selective_search_roidb.pkl')

        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = cPickle.load(fid)
            print '{} ss roidb loaded from {}'.format(self.name, cache_file)
            return roidb

        if self._image_set != 'test-all':
	   # print('in test all---calling load revised annotation and then slective search db')
            gt_roidb = self.gt_roidb()
            ss_roidb = self._load_selective_search_roidb(gt_roidb)
            roidb = imdb.merge_roidbs(gt_roidb, ss_roidb)
        else:
            roidb = self._load_selective_search_roidb(None)
        with open(cache_file, 'wb') as fid:
            cPickle.dump(roidb, fid, cPickle.HIGHEST_PROTOCOL)
        print 'wrote ss roidb to {}'.format(cache_file)

        return roidb

    def _load_selective_search_roidb(self, gt_roidb):
        filename = os.path.abspath(os.path.join(self.cache_path, '..',
                                                'selective_search_data',
                                                self.name + '.mat'))
        assert os.path.exists(filename), \
               'Selective search data not found at: {}'.format(filename)
        raw_data = sio.loadmat(filename)['boxes'].ravel()
	#print('in selective searchDB')
        box_list = []
        for i in xrange(raw_data.shape[0]):
            box_list.append(raw_data[i][:, :] - 1)

        return self.create_roidb_from_box_list(box_list, gt_roidb)

    def selective_search_IJCV_roidb(self):
        """
        Return the database of selective search regions of interest.
        Ground-truth ROIs are also included.

        This function loads/saves from/to a cache file to speed up future calls.
        """
        cache_file = os.path.join(self.cache_path,
                '{:s}_selective_search_IJCV_top_{:d}_roidb.pkl'.
                format(self.name, self.config['top_k']))

        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = cPickle.load(fid)
            print '{} ss roidb loaded from {}'.format(self.name, cache_file)
            return roidb

        gt_roidb = self.gt_roidb()
        ss_roidb = self._load_selective_search_IJCV_roidb(gt_roidb)
        roidb = datasets.imdb.merge_roidbs(gt_roidb, ss_roidb)
        with open(cache_file, 'wb') as fid:
            cPickle.dump(roidb, fid, cPickle.HIGHEST_PROTOCOL)
        print 'wrote ss roidb to {}'.format(cache_file)

        return roidb

    def rpn_roidb(self):
        if self._image_set != 'test-all02':            #initially test-all
            gt_roidb = self.gt_roidb()
            rpn_roidb = self._load_rpn_roidb(gt_roidb)
            roidb = imdb.merge_roidbs(gt_roidb, rpn_roidb)
        else:
            roidb = self._load_rpn_roidb(None)

        return roidb

    def _load_rpn_roidb(self, gt_roidb):
        filename = self.config['rpn_file']
        print 'loading {}'.format(filename)
        assert os.path.exists(filename), \
               'rpn data not found at: {}'.format(filename)
        with open(filename, 'rb') as f:
            box_list = cPickle.load(f)
        return self.create_roidb_from_box_list(box_list, gt_roidb)

    def _load_selective_search_IJCV_roidb(self, gt_roidb):
        IJCV_path = os.path.abspath(os.path.join(self.cache_path, '..',
                                                 'selective_search_IJCV_data',
                                                 'voc_' + self._year))
        assert os.path.exists(IJCV_path), \
               'Selective search IJCV data not found at: {}'.format(IJCV_path)

        top_k = self.config['top_k']
        box_list = []
        for i in xrange(self.num_images):
            filename = os.path.join(IJCV_path, self.image_index[i] + '.mat')
            raw_data = sio.loadmat(filename)
            box_list.append((raw_data['boxes'][:top_k, :]-1).astype(np.uint16))
        return self.create_roidb_from_box_list(box_list, gt_roidb)

    def _load_revised_annotation(self, index):
        """
        Load image and bounding boxes info from text file in the kaist dataset
        format.
        """
	#print(self._image_set)
        filename =(self._data_path+ '/anno_'+self._image_set +'/' + index +'.txt') #train
        
        with open(filename) as f:
            lines = f.readlines()

        num_objs = len(lines)
	
	
        boxes = np.zeros((num_objs-1, 4), dtype=np.uint16)
        gt_classes = np.zeros((num_objs-1), dtype=np.int32)
        overlaps = np.zeros((num_objs-1, self.num_classes), dtype=np.float32)
        seg_areas = np.zeros((num_objs-1), dtype=np.float32)

        # Load object bounding boxes into a data frame.
        ix = 0
        for obj in lines:
            # Make pixel indexes 0-based
            
            info = obj.split()
            if info[0]== "%":
                continue
            x1 = float(info[1]) 
            y1 = float(info[2])
            x2 = float(info[3])
            y2 = float(info[4])
            cls = self._class_to_ind['person']
            # adjusting for annotation format 
            x2 = x1 + x2
            y2 = y1 + y2
            
            boxes[ix, :] = [x1, y1, x2, y2] #removed -1 from each coordinate
            gt_classes[ix] = cls
            overlaps[ix, cls] = 1.0
            seg_areas[ix] = (x2 - x1 + 1) * (y2 - y1 + 1)
            ix = ix + 1

		
        overlaps = scipy.sparse.csr_matrix(overlaps)
        return {'boxes' : boxes,
                'gt_classes': gt_classes,
                'gt_overlaps' : overlaps,
                'flipped' : False,
                'seg_areas' : seg_areas}

    def _write_voc_results_file(self, all_boxes):
        #use_salt = self.config['use_salt']
        #comp_id = 'comp4'
        #if use_salt:
        #    comp_id += '-{}'.format(os.getpid())

        #path = os.path.join(self._devkit_path, 'results', 'kaist',
        #                    'Main', comp_id + '_')
        for cls_ind, cls in enumerate(self.classes):
            if cls == '__background__':
                continue
            print 'Writing {} Kaist results file'.format(cls)
            # save the predictions here
            #print(all_boxes)
            output_dir=self._data_path+'/output'
            filename=output_dir+'/' + 'det_' + self._image_set + '.txt'
            with open(filename, 'wt') as f:
                for im_ind, index in enumerate(self.image_index):
                    dets = all_boxes[cls_ind][im_ind]
                    #print('dets')
                    #print(dets)
		    if dets == []:
                        continue
                    # the VOCdevkit expects 1-based indices
                    for k in xrange(dets.shape[0]):
                     	#print(dets[k-1])
 		 	# since we are writing to fle with only 3 decimal points :3f[see below], we filter out all detections with confidence less than thaand do not write them to the file
			if(dets[k,-1]<1e-3):
			    continue
                        f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                                format(index, dets[k, -1],
                                       dets[k, 0] + 1, dets[k, 1] + 1,
                                       dets[k, 2] + 1, dets[k, 3] + 1))
        #return comp_id

    def _get_voc_results_file_template(self):
        filename=output_dir+'/' + 'det_' + self._image_set + '.txt'
        path=(filename)
        return path



    def _do_python_eval(self, output_dir='output'):
        annopath = (self._data_path +'/anno_'+ self._image_set+ '/' + '{:s}.txt') #test
        
        cachedir = (self._data_path + '/' + self._image_set+'_annotations_cache')

        imagesetfile = (self._data_path + '/Imagesetfiles/' + self._image_set+ '_imagesetfile.txt')
        
        aps = []
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)
        for i, cls in enumerate(self._classes):
            if cls == '__background__':
                continue
            output_dir=self._data_path+'/output'

            filename=output_dir+'/' + 'det_' + self._image_set + '.txt'
            rec, prec = voc_eval_miss_rate(
                filename, annopath, imagesetfile, cls, cachedir, ovthresh=0.5,
                use_07_metric = False)
            #aps += [ap]
            #print("aps")
            #print(aps)
            ap=0
            print('AP for {} = {:.4f}'.format(cls, ap))
            with open((output_dir+'/'+ self._image_set + '_pr.pkl'), 'wb') as f:
                cPickle.dump({'rec': rec, 'prec': prec, 'ap': ap}, f)
        print('~~~~~~~~')
        print('Results:')
        print('~~~~~~~~')
        print('')
        print('--------------------------------------------------------------')
        print('Results computed with the **unofficial** Python eval code.')
        print('Results should be very close to the official MATLAB eval code.')
        print('Recompute with `./tools/reval.py --matlab ...` for your paper.')
        print('-- Thanks, The Management')
        print('--------------------------------------------------------------')





    def _do_matlab_eval(self, comp_id, output_dir='output'):
        rm_results = self.config['cleanup']

        path = os.path.join(os.path.dirname(__file__),
                            'VOCdevkit-matlab-wrapper')
        cmd = 'cd {} && '.format(path)
        cmd += '{:s} -nodisplay -nodesktop '.format(cfg.MATLAB)
        cmd += '-r "dbstop if error; '
        cmd += 'voc_eval(\'{:s}\',\'{:s}\',\'{:s}\',\'{:s}\',{:d}); quit;"' \
               .format(self._devkit_path, comp_id,
                       self._image_set, output_dir, int(rm_results))
        print('Running:\n{}'.format(cmd))
        status = subprocess.call(cmd, shell=True)

    def evaluate_detections(self, all_boxes, output_dir):
        comp_id = self._write_voc_results_file(all_boxes)
        #self._do_matlab_eval(comp_id, output_dir)
        self._do_python_eval(output_dir)
        if self.config['cleanup']:
            for cls in self._classes:
                if cls == '__background__':
                    continue
                filename = self._get_voc_results_file_template().format(cls)
                os.remove(filename)
    

    def competition_mode(self, on):
        if on:
            self.config['use_salt'] = False
            self.config['cleanup'] = False
        else:
            self.config['use_salt'] = True
            self.config['cleanup'] = True

if __name__ == '__main__':
    from datasets.kaist_rgb import kaist_rgb
    d = kaist('train-all02')
    res = d.roidb
    from IPython import embed; embed()

