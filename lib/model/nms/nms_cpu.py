from __future__ import absolute_import

import numpy as np
import torch
import math
def nms_cpu(dets, thresh):
    dets = dets.numpy()
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order.item(0)
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h

        xx3 = np.minimum(x1[i], x1[order[1:]])
        yy3 = np.minimum(y1[i], y1[order[1:]])
        xx4 = np.maximum(x2[i], x2[order[1:]])
        yy4 = np.maximum(y2[i], y2[order[1:]])
        wg = np.maximum(0.0, xx4 - xx3 + 1)
        hg = np.maximum(0.0, yy4 - yy3 + 1)
        oa=wg*hg
        '''
        k=(oa-inter)/oa
        ovr = (inter / (areas[i] + areas[order[1:]] - inter))-k-0.00001

        #scores[i]=scores[i]*np.e**(math.log2(1-ovr)-1)#over=1score=0over=-1score=si
        inds = np.where(over<=thresh)[0]
        order = order[inds + 1]
        #效果较好 0.85~0.87
        '''
        k=(oa-(areas[i] + areas[order[1:]] - inter))/oa
        ovr = (inter / (areas[i] + areas[order[1:]] - inter))-k
		
        scores[i]=scores[i]*np.e**(math.log2(1-ovr)-1)
        inds = np.where(scores[i]>=0.5)[0]
        order = order[inds + 1]
    return torch.IntTensor(keep)
