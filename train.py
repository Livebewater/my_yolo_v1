from data.cocodataset import *
from models.yolo import MyYolo
import torch

input_size = [608, 608]
data_dir = "/home/LiuRunJi/Document/Dataset"
cocoDataset = COCODataset(
    data_dir=data_dir,
    img_size=608,
    transform=None,
    debug=False
)
