from torch_snippets import *
import selectivesearch
from torchvision import transforms, models, datasets
from torch_snippets import Report
from torchvision.ops import nms
from torchvision.ops import RoIPool
from torch import nn

class FRCNN(nn.Module):
    def __init__(self):
        super().__init__()
        rawnet = models.vgg16_bn(pretrained=True)
        # vgg16 with batch norm

        # VGG.features = nn.Module
        # nn.Module.parameters(recurse=True)
        # returns an iterator over module parameters
        # typically passed to an optimizer
        # recurse: +all the submodules
        for param in rawnet.features.parameters():
            param.requires_grad = True
        
        # nn.Module.children()
        # returns an iterator over immediate children modules
        self.seq = nn.Sequential(*list(rawnet.features.children())[:-1])
        
        self.roipool = RoIPool(7, spatial_scale=14/224)
        '''
        torchvision.ops.roi_pool(input:torch.Tensor, boxes:torch.Tensor,output_size=None,spatial_scale:float=1.0) -> torch.Tensor

        Parameters
        ______
        input: Tensor[N,C,H,W]
            batch with N elements, each containing C feature maps of dimensions HxW
        boxes: List[Tensor[L,4]]
            the box coordinates (x1,y1,x2,y2) 
        output_size: Tuple[int, int]
            the size of the output
        spatial_scale: float
            scaling factor that maps input coordinates to box coordinates
        '''
        feature_dim = 512*7*7

