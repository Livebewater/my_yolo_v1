import numpy as np
import os
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset
import cv2
from pycocotools.coco import COCO

coco_class_labels = ('background',
                     'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
                     'boat', 'traffic light', 'fire hydrant', 'street sign', 'stop sign',
                     'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
                     'elephant', 'bear', 'zebra', 'giraffe', 'hat', 'backpack', 'umbrella',
                     'shoe', 'eye glasses', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
                     'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
                     'skateboard', 'surfboard', 'tennis racket', 'bottle', 'plate', 'wine glass',
                     'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
                     'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
                     'couch', 'potted plant', 'bed', 'mirror', 'dining table', 'window', 'desk',
                     'toilet', 'door', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
                     'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'blender', 'book',
                     'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush')

coco_class_index = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20,
                    21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44,
                    46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 67,
                    70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90]


class COCODataset(Dataset):
    def __init__(self, data_dir="/home/LiuRunJi/Document/Dataset", json_file="instances_train2017.json",
                 name="train2017", img_size=416, transform=None,
                 min_size=1,
                 debug=False):
        self.data_dir = data_dir
        self.json_file = json_file
        self.coco = COCO(os.path.join(self.data_dir, 'annotations', self.json_file))
        self.ids = self.coco.getImgIds()  # get all image id
        if debug:
            self.ids = self.ids[1:2]
            print(f"debug model: id {self.ids}")
        self.class_ids = sorted(self.coco.getCatIds())  # t all class id
        self.name = name
        self.max_labels = 50
        self.img_size = img_size
        self.min_size = min_size
        self.transform = transform

    def __len__(self):
        return len(self.ids)

    def pull_image(self, index):
        id_ = self.ids[index]
        img_file = os.path.join(self.data_dir, self.name, f"{id_:012}.jpg")  # 012
        img = cv2.imread(img_file)

        if self.json_file == 'instances_val5k.json' and img is None:  # solve val and test status
            img_file = os.path.join(self.data_dir, 'train2017',
                                    f"{id_:012}.jpg")

        elif self.json_file == 'image_info_test-dev2017.json' and img is None:
            img_file = os.path.join(self.data_dir, 'test2017',
                                    f"{id_:012}.jpg")

        elif self.json_file == 'image_info_test2017.json' and img is None:
            img_file = os.path.join(self.data_dir, 'test2017',
                                    f"{id_:012}.jpg")

        if img is None:
            img = cv2.imread(img_file)
        return img, id_


def visual(pic, annotation):
    rgb_pic = np.zeros_like(pic)
    rgb_pic[:, :, 0] = pic[:, :, 2]
    rgb_pic[:, :, 1] = pic[:, :, 1]
    rgb_pic[:, :, 2] = pic[:, :, 0]
    plt.figure(1)
    box = annotation["bbox"]
    rgb_pic[int(box[1]), int(box[0]):int(box[0] + box[2]), :] = [255, 0, 0]
    rgb_pic[int(box[1]):int(box[1] + box[3]), int(box[0]), :] = [255, 0, 0]
    rgb_pic[int(box[1] + box[3]), int(box[0]):int(box[0] + box[2]), :] = [255, 0, 0]
    rgb_pic[int(box[1]):int(box[1] + box[3]), int(box[0] + box[2]), :] = [255, 0, 0]
    plt.title()
    plt.imshow(rgb_pic)


def __getitem__(self, index):
    img, id_ = self.pull_image(index)
    anno_id_ = self.coco.getAnnIds(imgIds=[int(id_)], iscrowd=None)
    annotations = self.coco.loadAnns(anno_id_)  # get annotation information id

    assert img is not None

    h, w, c = img.shape

    target = []

    for anno in annotations:
        x1 = np.max((0, anno["bbox"][0]))
        y1 = np.max((0, anno["bbox"][1]))
        x2 = np.min((w - 1, x1 + np.max((0, anno["bbox"][2] - 1))))
        y2 = np.min((h - 1, y1 - np.max((0, anno["bbox"][3] - 1))))


if __name__ == "__main__":
    cocoDataset = COCODataset()
    img, _ = cocoDataset.pull_image(0)

    visual(img, cocoDataset[0])

    plt.show()

    plt.show()
