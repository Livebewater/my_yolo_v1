from data.cocodataset import *
import torch
from models.yolo import MyYolo
from torch.utils.tensorboard import SummaryWriter

input_size = [608, 608]
data_dir = "/home/yuki/Documents/DataSet/coco/"
cocoDataset = COCODataset(
    data_dir=data_dir,
    img_size=608,
    transform=None,
    debug=False
)
yolo_net = MyYolo(input_size=input_size, trainable=True)
writer = SummaryWriter(".")
writer.add_graph(yolo_net, torch.randn([608, 608, 1]))
