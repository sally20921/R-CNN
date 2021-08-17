from torch_snippets import *
import selectivesearch
from torchvision import transforms, models, datasets
from torch_snippets import Report
from torchvision.ops import nms
from cv2 import cv2

device = 'cuda' if torch.cuda.is_available() else 'cpu'

IMAGE_ROOT = 'images/images'
DF_RAW = pd.read_csv('df.csv')




if __name__=="__main__":
    test()
