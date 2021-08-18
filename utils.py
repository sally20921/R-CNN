from torch_snippets import *
import selectivesearch
from torchvision import transforms, models, datasets
from torch_snippets import Report
from torchvision.ops import nms
import numpy as np
from cv2 import cv2

## regions[0] = {'labels': [0.0], 'rect': (0,0,15,24), 'size': 260}

def extract_candidates(img):
    '''
    Parameter
    ______
    img: np.array
        W-by-H-by-3

    Returns
    ____
    candidates: list of list
        [[x,y,w,h], [x,y,w,h]]
    '''
    img_lbl, regions = selectivesearch.selective_search(img, scale=200, min_size=100)
    img_area = np.prod(img.shape[:2]) #W-by-H
    candidates = []
    # select candidates out of regions 
    for r in regions:
        # if it is already in candidates
        if r['rect'] in candidates: continue
        if r['size'] < (0.05 * img_area): continue
        if r['size'] > (1 * img_area): continue
        x, y, w, h = r['rect']
        candidates.append(list(r['rect']))
    return candidates
    
def extract_iou(boxA, boxB, epsilon=1e-5):
    '''
    Parameter
    _____
    boxA: [x,y, x+w, y+h]
    boxB: [x,y, x+w, y+h]
    epsilon: in case area_combined becomes 0

    Returns
    _____
    iou = area_overlap / (area_combined + epsilon)
    '''
    x1 = max(boxA[0], boxB[0])
    y1 = max(boxA[1], boxB[1])
    x2 = min(boxA[2], boxB[2])
    y2 = min(boxA[3], boxB[3])
    width = (x2 - x1)
    height = (y2 - y1)
    if (width < 0) or (height < 0):
        return 0.0
    area_overlap = width * height
    area_a = (boxA[2]-boxA[0])*(boxA[3]-boxA[1])
    area_b = (boxB[2]-boxB[0])*(boxB[3]-boxB[1])
    area_combined = area_a + area_b - area_overlap
    iou = area_overlap / (area_combined + epsilon)
    return iou 


