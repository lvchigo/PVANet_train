#!/usr/bin/env python

# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""
Demo script showing detections in sample images.

See README.md for installation instructions before running.
"""

import _init_paths
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer
import numpy as np
import scipy.io as sio
import caffe, os, sys, cv2
import argparse
from utils.cython_bbox import bbox_vote

#voc2007
CLASSES = ('__background__', # always index 0
         'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 
         'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor')

# mutilabel
#CLASSES = ('__background__', # always index 0
#'animal.bird', 'animal.cat', 'animal.cow', 'animal.dog', 'animal.horse', 
#'animal.other', 'animal.panda', 'animal.rabbit', 'animal.sheep', 'electronics.camera', 
#'electronics.cellphone', 'electronics.cursormouse', 'electronics.keyboard', 'electronics.monitor', 'electronics.notebook', 
#'electronics.other', 'food.barbecue', 'food.cake', 'food.coffee', 'food.cook', 
#'food.fruit', 'food.hotpot', 'food.icecream', 'food.other', 'food.pizza', 
#'food.sushi', 'furniture.bed', 'furniture.chair', 'furniture.other', 'furniture.pottedplant', 
#'furniture.sofa', 'furniture.table', 'goods.bag', 'goods.ball', 'goods.book', 
#'goods.bottle', 'goods.clock', 'goods.clothes', 'goods.cosmetics', 'goods.cup', 
#'goods.drawbar', 'goods.flower', 'goods.glass', 'goods.guitar', 'goods.hat', 
#'goods.jewelry', 'goods.other', 'goods.puppet', 'goods.shoe', 'other.2dcode', 
#'other.logo', 'other.other', 'other.sticker', 'other.text', 'person.body', 
#'person.face', 'vehicle.airplane', 'vehicle.bicycle', 'vehicle.boat', 'vehicle.bus', 
#'vehicle.car', 'vehicle.motorbike', 'vehicle.other', 'vehicle.train')

COLORS = ( (0,0,255), (0,255,255), (0,255,0),   (255,255,0),
           (255,0,0), (255,0,255), (0,128,255), (255,128,0))

def demo(net, im_file):
    """Detect object classes in an image using pre-computed object proposals."""

    # Load the demo image
    #im_file = os.path.join(cfg.DATA_DIR, 'demo', image_name)
    im = cv2.imread(im_file)

    # timers
    t = {'im_preproc': Timer(), 'im_net' : Timer(), 'im_postproc': Timer(), 'misc' : Timer()}

    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(net, im, t)
    timer.toc()
    print ('Detection took {:.3f}s for '
           '{:d} object proposals').format(timer.total_time, boxes.shape[0])

    # Visualize detections for each class
    CONF_THRESH = 0.8
    NMS_THRESH = 0.3
    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1 # because we skipped background
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]
        #vis_detections(im, cls, dets, thresh=CONF_THRESH)

def demo_oneimage_mutilabel(net, im_file, output_path, im_name, isPrint):
    """Detect object classes in an image using pre-computed object proposals."""

    # Load the demo image
    im = cv2.imread(im_file)

    font = cv2.FONT_HERSHEY_SIMPLEX

    # timers
    t = {'im_preproc': Timer(), 'im_net' : Timer(), 'im_postproc': Timer(), 'misc' : Timer()}

    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(net, im, t)
    timer.toc()
    if isPrint == 1:
        print ('Detection took {:.3f}s for '
           '{:d} object proposals').format(timer.total_time, boxes.shape[0])
    #print ('len(scores):{},len(boxes):{}').format(len(scores),len(boxes))

    # Visualize detections for each class
    CONF_THRESH = 0.8

    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1 # because we skipped background
        #print ('cls_ind:{}').format(cls_ind)
        inds = np.where(scores[:, cls_ind] >= CONF_THRESH)[0]
        if len(inds) == 0:
            continue
            
        cls_boxes = boxes[inds, 4*cls_ind:4*(cls_ind + 1)]
        cls_scores = scores[inds, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = nms(dets, cfg.TEST.NMS)
        
        dets_NMSed = dets[keep, :]
        if cfg.TEST.BBOX_VOTE:
            dets = bbox_vote(dets_NMSed, dets)
        else:
            dets = dets_NMSed

        if len(dets) == 0:
            continue

        for i in range(len(dets)):
            bbox = dets[i][:4]
            score = dets[i][4]
            #print ('bbox:{},score:{}').format(bbox,score)
            
            cv2.rectangle(im,(bbox[0],bbox[1]),(bbox[2],bbox[3]),COLORS[i%8],2)
            text='{:s}_{:.3f}'.format(cls, score)
            cv2.putText(im,text,(int(bbox[0]+5),int(bbox[1]+15)), font, 0.5,COLORS[i%8],2,8) 

    # save image
    output_file = os.path.join( output_path, '{:s}'.format(im_name) )      
    #print 'output_file:{}'.format(output_file)
    cv2.imwrite(output_file, im)


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Faster R-CNN demo')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--cpu', dest='cpu_mode',
                        help='Use CPU mode (overrides --gpu)',
                        action='store_true')
    parser.add_argument('--def', dest='prototxt', default='models/pvanet/lite/lite_src/test.pt')
    parser.add_argument('--net', dest='caffemodel', default='models/pvanet/lite/test.model')
    parser.add_argument('--input', dest='input_file', default='/home/chigo/image/test/list_street.txt')
    parser.add_argument('--output', dest='output_path', default='data/output_img/')

    args = parser.parse_args()

    return args

if __name__ == '__main__':                          
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals

    args = parse_args()
    prototxt = os.path.join(args.prototxt)
    caffemodel = os.path.join(args.caffemodel)

    if not os.path.isfile(caffemodel):
        raise IOError(('{:s} not found.\nDid you run ./data/script/'
                       'fetch_faster_rcnn_models.sh?').format(caffemodel))

    if args.cpu_mode:
        caffe.set_mode_cpu()
    else:
        caffe.set_mode_gpu()
        caffe.set_device(args.gpu_id)
        cfg.GPU_ID = args.gpu_id
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)

    print '\n\nLoaded network {:s}'.format(caffemodel)

    # Warmup on a dummy image
    im = 128 * np.ones((300, 500, 3), dtype=np.uint8)
    # timers
    t = {'im_preproc': Timer(), 'im_net' : Timer(), 'im_postproc': Timer(), 'misc' : Timer()}
    for i in xrange(2):
        _, _= im_detect(net, im, t)

    output_dir = args.output_path[:args.output_path.rfind('/')]
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
#    for cls_ind, cls in enumerate(CLASSES[1:]):
#        cls_dir = '{:s}/{:s}'.format(output_dir,cls)
#        if not os.path.exists(cls_dir):
#            os.makedirs(cls_dir)

    file = open(args.input_file, "r")
    alllines=file.readlines();
    file.close();

    nCount = 0;
    for line in alllines:

        #print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
        #print 'line:{}'.format(line)
        im_name = line[(line.rfind('/')+1):];
        im_name = im_name[:im_name.rfind('.')];
        im_name = '{:s}.jpg'.format(im_name)
        input_path = line[:line.rfind('/')];
        im_file = '{:s}/{:s}'.format(input_path,im_name)
        #print 'input_file:{}'.format(im_file)

        isPrint = 0;
        nCount = nCount+1;    
        if nCount%50 == 0:
            print 'load img:{}...'.format(nCount)
            isPrint = 1;
        else:
            isPrint = 1;#0

        #add by chigo                    
        #demo(net, im_name)
        demo_oneimage_mutilabel(net, im_file, args.output_path, im_name, isPrint)

    print 'All load img:{}!!'.format(nCount)

