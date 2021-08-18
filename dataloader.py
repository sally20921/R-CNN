from torch_snippets import *
import selectivesearch
from torchvision import transforms, models, datasets
from torch_snippets import Report
from torchvision.ops import nms
from utils import *
import torch.utils.data as data
import pandas as pd
'''
XMin, XMax, YMin, YMax are available as a proportion of the shape of images
in the dataframe
'''

# list of per image
FPATHS, GTBBS, CLSS, DELTAS, ROIS, IOUS = [], [], [], [], [], []
# GTBBS: ground truth bounding boxes
# DELTAS: delta offset of a bb with region proposal 
# IOUS: IoU of region proposals with ground truth

IMAGE_ROOT = 'images/images'
DF_RAW = pd.read_csv('df.csv')

normalize = transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])

def preprocess_image(img):
    img = torch.tensor(img).permute(2,1,0) # switch channels back to BGR
    img = normalize(img)
    return img.to(device).float()


class OpenImages(data.Dataset):
    '''
    get each image from each image path
    read df.csv and return image, boxes, classes, image_path 
    for each image
    '''
    def __init__(self, df, image_folder=IMAGE_ROOT):
        self.root = image_folder
        self.df = df
        self.unique_images = df['ImageID'].unique()
        # returns the unique values as a NumPy array
        # hash table-based unique, therefore does NOT sort
        # pd.Series([2,1,3,3], name='A').unique()
        # array([2,1,3])

    def __len__(self):
        '''
        return the number of images
        '''
        return len(self.unique_images)

    def __getitem__(self, ix):
        '''
        Parameter
        _____
        ix: int
            index
        
        Returns
        ____
        image: ndarray H-by-W-by-C
        boxes: list of ndarray [xmin,ymin,xmax,ymax]
        classes: list of string labelname
        image_path: string
        '''
        # cv2.imread(): 3-d matrix (height, width, channels)
        # imread() stored in the order B-G-R-A
        image_id =self.unique_images[ix]
        image_path = f'{self.root}/{image_id}.jpg'
        # cv2.IMREAD_COLOR (reads the image but no transparency)
        # BGR to RGB
        image = cv2.imread(image_path, 1)[..., ::-1] 
        h, w, _ = img.shape

        # return a NumPy representation of the DataFrame
        df = self.df.copy()
        df = df[df['ImageID'] == image_id]
        boxes = df['XMin,YMin,XMax,YMax'.split(',')].values 
        # ndarray of [xmin,ymin,xmax,ymax]
        boxes = (boxes * np.array([w,h,w,h])).astype(np.uint16).tolist()

        classes = df['LabelName'].values.tolist()
        return image, boxes, classes, image_path



def prepare_data():
    '''
    create input and output values 

    input: candidates from selectivesearch
    output: class corresponding to candidates
            offset of the candidate with respect to the bb it overlaps the most with (if candidate contains an object)
    '''       
    ds = OpenImages(df=DF_RAW)
    # for one image: map each candidate to each bb and label 
    for ix, (im, bbs, labels, fpath) in enumerate(ds):
        candidates = extract_candidates(im)
        candidates = np.array([x,y,x+w,y+h] for x,y,w,h in candidates])
        # candidates: ndarray of [xmin,ymin,xmax,ymax]
        # for each candidate
        ious, rois, clss, deltas = [], [], [], []
        
        # go through all the proposals from SelectiveSearch
        # store those with a high IoU as bus/truck proposal
        # rest is the background proposal
        ious = np.array([extract_ious(candidate, _bb_) for _bb_ in bbs] for candidate in candidates]).T

        for jx, candidate in enumerate(candidates):
            cx, cy, cX, cY = candidate
            # extract IoU for each cdd with respect to all gtbbs
            candidate_ious = ious[jx] # for each candidate list of bbs
            best_iou_at = np.argmax(candidate_ious)
            best_iou = candidate_iou[best_iou_at]
            best_bb = _x, _y, _X, _Y = bbs[best_iou_at]
            if best_iou > 0.3: clss.append(labels[best_iou_at])
            else: cls.append('background')

            # offset needed to transform cdd to gtbb
            delta = np.array([_x-cx, _y-cy, _X-cX, _Y-cY]) / np.array([W,H,W,H])
            deltas.append(delta)
            rois.append(candidate/np.array([W,H,W,H]))

        FPATHS.append(fpath)
        IOUS.append(ious)
        ROIS.append(rois)
        CLSS.append(clss)
        DELTAS.append(deltas)
        GTBBS.append(candidates)

    FPATHS = [f'{IMAGE_ROOT}/{stem(f)}.jpg' for f in FPATHS]
    
    # numpy.ndarray.flatten(order='C')
    # return a copy of the array collapsed into one dimension
    targets = pd.DataFrame(flatten(CLSS), columns=['label'])
    # list of list of strings
    label2target = {l:t for t,l in enumerate(targets['label'].unique())}
    target2label = {t:l for l,t in label2target.items()}
    background_class = label2target['background']

class FRCNNDataset(data.Dataset):
    def __init__(self, fpaths, rois, labels, deltas, gtbbs):
        self.fpaths = fpaths
        self.gtbbs = gtbbs
        self.rois = rois
        self.labels = labels
        self.deltas = deltas

    def __len__(self):
        return len(self.fpaths)

    def __getitem__(self, ix):
        fpath = str(self.fpaths[ix])
        image = cv2.imread(fpath, 1)[...,::-1] # channel to rgb
        H, W, _ = image.shape
        sh = np.array([W,H,W,H])

        gtbbs = self.gtbbs[ix]
        rois = self.rois[ix]
        bbs = (np.array(rois)*sh).astype(np.uint16) #(xmin,ymin,xmax,ymax)
        labels = self.labels[ix]
        deltas = self.deltas[ix]
        crops = [image[y:Y, x:X] for (x,y,X,Y) in bbs]
        # crops: list of numpy arrays [HxW]
        return image, crops, bbs, labels, deltas, gtbbs, fpath

    def collate_fn(self, batch):





 



