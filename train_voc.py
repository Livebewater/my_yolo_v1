from data.vocdataset import *
from data import *
import torch
import time
from models.yolo import MyYolo
from torch.utils.tensorboard import SummaryWriter
from utils.augmentations import *
from torch.utils.data import DataLoader
import tools
import os
import torch.optim as optim
import torch.backends.cudnn as cudnn

train_epoch = 250
use_board = True
input_size = [416, 416]
data_dir = "/home/LiuRunJi/Documents/Dataset/VOC/"
saving_model = True

if torch.cuda.is_available():
    print("using gpu")
    cudnn.benchmark = True
    device = torch.device("cuda")
else:
    print("using cpu")
    device = torch.device("cpu")
if use_board:
    print("using board")
    log_path = os.path.join("log", "voc", time.strftime("%y-%m-%d", time.localtime()))
    if not os.path.exists(log_path):
        os.makedirs(log_path, exist_ok=True)  # if exist_ok  True will ignore the folder exits situation
    writer = SummaryWriter(log_path)
if saving_model:
    print("saving model")
    save_path = os.path.join("model", "voc", time.strftime("%y-%m-%d", time.localtime()))
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)

vocDataset = VOCDataset(root_dir=data_dir, transform=SSDAugmentation([416, 416], mean=(0.406, 0.456, 0.485), std=(0.225, 0.224, 0.229)))
yolo_net = MyYolo(input_size=input_size, device=device, trainable=True, num_classes=20).to(device)
batch_size = 32
train_dataLoader = DataLoader(
    vocDataset,
    batch_size=batch_size,
    shuffle=True,
    collate_fn=detection_collate,
    num_workers=8
)
iteration = len(vocDataset) // batch_size
print(len(vocDataset))
optimizer = optim.SGD(yolo_net.parameters(), lr=3e-4, momentum=0.9, weight_decay=5e-4)
t0 = time.time()
for epoch in range(train_epoch):

    # if (epoch + 1) % eval_epoch == 0:
    #     yolo_net.trainable = False

    for iter_i, (data, labels) in enumerate(train_dataLoader):
        optimizer.zero_grad()
        images = data.to(device)
        labels = [label.tolist() for label in labels]
        targets = torch.FloatTensor(
            tools.gt_creator(input_size=input_size, stride=yolo_net.stride, label_lists=labels)).to(
            device)
        conf_loss, class_loss, box_loss, loss = yolo_net(images, targets=targets)
        loss.backward()
        optimizer.step()
        if iter_i % 10 == 0:
            t1 = time.time()
            print(
                f"[{epoch}/{train_epoch}] [{iter_i} / {iteration}] conf loss: [{conf_loss.item():.3f}] class loss: [{class_loss.item():.3f}] box loss:"
                f"[{box_loss.item():3f}] total loss: [{loss.item():.3f}] time {t1-t0:.3f}")
            t0 = time.time()
        if iter_i % 50 == 0 and use_board:
            writer.add_scalar("object conf loss", conf_loss.item(), iter_i + epoch * batch_size)
            writer.add_scalar("class loss", class_loss.item(), iter_i + epoch * batch_size)
            writer.add_scalar("box loss", box_loss.item(), iter_i + epoch * batch_size)
            writer.add_scalar("total loss", loss.item(), iter_i + epoch * batch_size)
    if (epoch + 1) % 10 == 0 and saving_model:
        print("saving model", epoch + 1)
        torch.save(yolo_net.state_dict(), f"{save_path}/{epoch + 1}.pth")
