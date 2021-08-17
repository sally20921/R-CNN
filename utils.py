from torch_snippets import *
import selectivesearch
from torchvision import transforms, models, datasets
from torch_snippets import Report
from torchvision.ops import nms
import numpy as np
from cv2 import cv2

def extract_candidates(img):

