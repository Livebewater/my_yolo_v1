import xml.etree.ElementTree as ET
import torch
import os.path as op
from torch.utils.data import Dataset
import cv2
import numpy as np

VOC_CLASSES = (
    'aeroplane', 'bicycle', 'bird', 'boat',
    'bottle', 'bus', 'car', 'cat', 'chair',
    'cow', 'diningtable', 'dog', 'horse',
    'motorbike', 'person', 'pottedplant',
    'sheep', 'sofa', 'train', 'tvmonitor')


class VOCAnnotationTransform(object):
    def __init__(self, class_to_ind=None, keep_difficult=False):
        if class_to_ind is not None:
            self.class_to_index = class_to_ind
        else:
            self.class_to_index = dict(zip(VOC_CLASSES, range(len(VOC_CLASSES))))
        self.keep_difficult = keep_difficult

    def __call__(self, target, width, height):
        res = []
        for obj in target.iter("object"):
            difficult = int(obj.find("difficult").text) == 1  # difficult代表待检测目标很难识别, == 1可以令difficult确定值为0, 1
            if not self.keep_difficult and difficult:
                continue
            name = obj.find("name").text.lower().strip()
            box = obj.find("bndbox")
            boundbox = [int(box.find(i).text) - 1 for i in ["xmin", "ymin", "xmax", "ymax"]]
            boundbox[0] /= width
            boundbox[1] /= height
            boundbox[2] /= width
            boundbox[3] /= height
            boundbox.append(self.class_to_index[name])
            res.append(boundbox)
        return res


class VOCDataset(Dataset):

    def __init__(self, root_dir, dataset_type=["trainval", "trainval"], year=["2007", "2012"], transform=None,
                 target_transform=VOCAnnotationTransform, size=[416, 416]):
        """

        Args:
            root_dir:
                -JPEGImages
                -Annotations
            dataset_type:
            transform:
            target_transform:
        """
        self.root = root_dir
        self.dataset_type = dataset_type
        if transform is not None:
            self.transform = transform(size)
        else:
            self.transform = transform
        self.size = size
        self.target_transform = target_transform
        self.image_path = {}
        self.annotations_path = {}
        for i in year:
            self.image_path[i] = op.join(root_dir, "VOC" + i, "JPEGImages")
        for i in year:
            self.annotations_path[i] = op.join(root_dir, "VOC" + i, "Annotations")
        self._id = []
        for y, n in zip(year, dataset_type):
            root_path = op.join(self.root, "VOC" + y)
            for line in open(op.join(root_path, "ImageSets", "Main", n + ".txt")):
                self._id.append((y, line.strip()))

    def __len__(self):
        return len(self._id)

    def __getitem__(self, index):
        images, targets, H, W = self.get_items(index)
        return torch.from_numpy(images), targets

    def get_test_items(self, index):
        year, id = self._id[index]

        target = ET.parse(op.join(self.annotations_path[year], f"{id}.xml")).getroot()
        image = cv2.imread(op.join(self.image_path[year], f"{id}.jpg"))
        H, W, C = image.shape
        target = self.target_transform()(target, W, H)
        return image.transpose([2, 0, 1]), target

    def get_items(self, index):
        year, id = self._id[index]

        target = ET.parse(op.join(self.annotations_path[year], f"{id}.xml")).getroot()
        image = cv2.imread(op.join(self.image_path[year], f"{id}.jpg"))
        H, W, C = image.shape
        if self.target_transform is not None:
            target = self.target_transform()(target, W, H)
        if self.transform is not None:
            target = np.array(target)
            image = image[:, :, (2, 1, 0)]
            image, box, label = self.transform(image, target[:, :4], target[:, 4])
            target = np.hstack((box, np.expand_dims(label, axis=1)))
        return image.transpose([2, 0, 1]), target, H, W


if __name__ == "__main__":
    vocDataset = VOCDataset("/home/yuki/Documents/DataSet/VOC", dataset_type=["train"], year=["2007"])
    images, targets = vocDataset[10]
    print(targets)
    print(len(vocDataset))
    cv2.imshow("test", np.array(images).transpose([1, 2, 0]))
    cv2.waitKey(0)
