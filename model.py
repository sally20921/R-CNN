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
        self.seq = nn.Sequential

        
