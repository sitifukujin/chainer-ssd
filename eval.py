#! /usr/bin/env python3
# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------
# This pascal_voc_mini class came from 
#   https://github.com/mitmul/chainer-faster-rcnn
# --------------------------------------------------------
import os
import argparse
import numpy as np
from lib import VOCDataset
from lib import voc_eval


class pascal_voc_mini(object):
    def __init__(self, year, image_set, root, outputdir):
        self._year = year
        self._image_set = image_set
        self._root = root
        self._annopath = os.path.join(
                        self._root,
                        'VOC' + self._year,
                        'Annotations') + os.sep + '{:s}.xml'
        self._imagesetfile = os.path.join(
                            self._root,
                            'VOC' + year,
                            'ImageSets',
                            'Main',
                            self._image_set + '.txt')
        self._outputdir = outputdir

    def _get_voc_results_file_template(self):
        # /results/VOC2007/Main/comp4_det_test_{label}.txt
        filename = 'comp4' + '_det_' + self._image_set + '_{:s}.txt'
        path = self._outputdir + os.sep + filename
        return path

    def _do_python_eval(self):
        cachedir = os.path.join(args.output, 'cache')
        aps = []
        # The PASCAL VOC metric changed in 2010
        use_07_metric = True if int(self._year) < 2010 else False
        print('VOC07 metric? ' + ('Yes' if use_07_metric else 'No'))

        for i, cls in enumerate(VOCDataset.labels):
            filename = self._get_voc_results_file_template().format(cls)
            rec, prec, ap = voc_eval(
                filename, self._annopath, self._imagesetfile, 
                cls, cachedir, ovthresh=0.5,
                use_07_metric=use_07_metric)
            aps += [ap]
            print('AP for {} = {:.4f}'.format(cls, ap))
        print('Mean AP = {:.4f}'.format(np.mean(aps)))
        print('~~~~~~~~')
        print('Results:')
        for ap in aps:
            print('{:.3f}'.format(ap))
        print('{:.3f}'.format(np.mean(aps)))
        print('~~~~~~~~')
        print('')
        print('--------------------------------------------------------------')
        print('Results computed with the **unofficial** Python eval code.')
        print('Results should be very close to the official MATLAB eval code.')
        print('Recompute with matlab for your paper.')
        print('-- Thanks, The Management')
        print('--------------------------------------------------------------')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default='VOCdevkit')
    parser.add_argument('--output', default='result')
    parser.add_argument('test')

    args = parser.parse_args()
    year, image_set = args.test.split('-')
    p_eval = pascal_voc_mini(year, image_set, args.root, args.output)
    p_eval._do_python_eval()

