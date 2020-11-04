from data.cocodataset import *
import torch
from models.yolo import MyYolo
from torch.utils.tensorboard import SummaryWriter
from data import *
import tools
import torch.optim as optim
import torch.backends.cudnn as cudnn

if torch.cuda.is_available():
    cudnn.benchmark = True
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
input_size = [608, 608]
data_dir = "/home/LiuRunJi/Document/Dataset/coco/"
cocoDataset = COCODataset(
    root_dir=data_dir,
    img_size=608,
    transform=None,
    debug=False
)
yolo_net = MyYolo(input_size=input_size, trainable=True)
batch_size = 1
train_dataloader = torch.utils.data.DataLoader(
    cocoDataset,
    batch_size=batch_size,
    shuffle=True,
    collate_fn=detection_collate,
    num_workers=8
)
optimizer = optim.SGD(yolo_net.parameters(), lr=1e-4, momentum=0.9, weight_decay=5e-4)
for data, label in train_dataloader:
    labels = [_.tolist() for _ in label]
    images = data.to(device)
    targets = torch.FloatTensor(tools.gt_creator(input_size=input_size, stride=yolo_net.stride, label_lists=labels)).to(
        device)
    conf_loss, class_loss, box_loss, loss = yolo_net(images, targets=targets)
    break