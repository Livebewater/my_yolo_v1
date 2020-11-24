import torch
from models.yolo import MyYolo
from data.vocdataset import *
from data import *
from torch.utils.data import DataLoader
import random
import cv2

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

data_dir = "/home/LiuRunJi/Documents/Dataset/VOC/"

batch_size = 256
input_size = [416, 416]
vocDataset = VOCDataset(root_dir=data_dir, transform=BaseTransform(input_size), dataset_type=["val"], year=["2007"])
yolo_net = MyYolo(input_size=input_size, device=device, trainable=False, num_classes=20, conf_thresh=0.5).to(device)
yolo_net.load_state_dict(torch.load(r"model/voc/20-11-23/160.pth"))
f=1
while true:
    index = random.randint(0, len(vocDataset))
    image, targets = vocDataset[index]
    image = image.unsqueeze(dim=0).to(device)
    pred_targets = yolo_net(image)

    show_image, _ = vocDataset.get_test_items(index)
    show_image = np.array(show_image).transpose([1, 2, 0])
    print(pred_targets)
    targets[:, :-1] *= 416
    draw(show_image, targets, index, pred_targets)

