import torch.nn as nn
import torch
from backbone.resnet import resnet18
from utils import Conv2d, SPP, SAM


class MyYolo(nn.Module):
    def __init__(self, input_size, num_classes=80, trainable=False, conf_thresh=0.01,
                 nms_threshold=0.5, hr=False):
        super(MyYolo, self).__init__()
        self.num_classes = num_classes
        self.trainable = trainable
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_threshold
        self.stride = 49
        self.grid_cell = self.create_grid(input_size)
        self.input_size = input_size

        self.backbone = resnet18(pretrained=True)
        self.conv_set = nn.Sequential(
            Conv2d(512, 256, 1, leakyReLU=True),
            Conv2d(256, 512, 3, padding=1, leakyReLU=True),
            Conv2d(512, 256, 1, leakyReLU=True),
            Conv2d(256, 512, 3, padding=1, leakyReLU=True),
        )
        self.pred = nn.Conv2d(512, 1 + self.num_classes + 4, 1)
    def create_grid(self, input_size):
        h, w = input_size
        hs, ws = h // self.stride, w // self.stride
        grid_y, grid_x = torch.meshgrid([torch.arange(hs), torch.arange(ws)])
        grid_xy = torch.stack([grid_x, grid_y], dim=-1).float()
        grid_xy = grid_xy.view(1, hs * ws, 2).to(self.device)

        return grid_xy
    def forward(self):
        pass
