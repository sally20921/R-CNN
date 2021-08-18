from torch_snippets import *
import selectivesearch
from torchvision import transforms, models, datasets
from torch_snippets import Report
from torchvision.ops import nms
from utils import *

FPATHS, GTBBS, CLSS, DELTAS, ROIS, IOUS = [], [], [], [], [], []
 
IMAGE_ROOT = 'images/images'
DF_RAW = pd.read_csv('df.csv')


