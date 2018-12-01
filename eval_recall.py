import _init_paths
from model.utils.config import cfg, cfg_from_file, cfg_from_list
from datasets.factory import get_imdb
from datasets.imdb import imdb
#from datasets.imdb import imdb.imdb.evaluate_recall
#from imdb import evaluate_recall
import argparse
import time, os, sys
import numpy as np

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Test a Fast R-CNN network')
    parser.add_argument('--imdb', dest='imdb_name',
                        help='dataset to test',
                        default='kaist_train-all02', type=str)
    parser.add_argument('--method', dest='method',
                        help='proposal method',
                        default='selective_search', type=str) #Default = selective_search
    parser.add_argument('--rpn-file', dest='rpn_file',
                        default=None, type=str)#default = None

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()

    print('Called with args:')
    print(args)
    classes = ('__background__', 'pedestrian')
    imdb_1 = get_imdb(args.imdb_name)
    print(imdb_1)
    imdb_2 = imdb(args.imdb_name, classes)
    print("imdb_2.classes")
    print(imdb_2.classes)
    print("get imdb called")
    imdb_1.set_proposal_method(args.method)
    print("set proposal method called")
    if args.rpn_file is not None:
        print('inside if not')
        imdb_1.config['rpn_file'] = args.rpn_file

    candidate_boxes = None
    if 0:
        print("inside if 0")
        import scipy.io as sio
        filename = 'debug/stage1_rpn_voc_2007_test.mat'
        raw_data = sio.loadmat(filename)['aboxes'].ravel()
        candidate_boxes = raw_data
        print("candidate boxes")
        print(candidate_boxes)

    ar, gt_overlaps, recalls, thresholds = \
        imdb_1.evaluate_recall(candidate_boxes=candidate_boxes)
    print("ar")
    print(ar)
    print("gt_overlaps")
    print(gt_overlaps)
    print("recalls")
    print(recalls)
    print("thresholds")
    print(thresholds)
    print("after evaluate recall")
    print 'Method: {}'.format(args.method)
    print 'AverageRec: {}'.format(ar)
    print("after evaluate recall")
'''
    def recall_at(t):
        ind = np.where(thresholds > t - 1e-5)[0][0]
        assert np.isclose(thresholds[ind], t)
        return recalls[ind]

    print 'Recall@0.5: {:.3f}'.format(recall_at(0.5))
    print 'Recall@0.6: {:.3f}'.format(recall_at(0.6))
    print 'Recall@0.7: {:.3f}'.format(recall_at(0.7))
    print 'Recall@0.8: {:.3f}'.format(recall_at(0.8))
    print 'Recall@0.9: {:.3f}'.format(recall_at(0.9))
    # print again for easy spreadsheet copying
    print '{:.3f}'.format(ar)
    print '{:.3f}'.format(recall_at(0.5))
    print '{:.3f}'.format(recall_at(0.6))
    print '{:.3f}'.format(recall_at(0.7))
    print '{:.3f}'.format(recall_at(0.8))
    print '{:.3f}'.format(recall_at(0.9))
'''
