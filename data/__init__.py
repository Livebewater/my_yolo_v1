import torch
import cv2
import numpy as np


def detection_collate(batch):
    """
    解决一张图像上可能有多个box的情况,此时label的个数不一致,如果用默认的collate_fn会无法stack起来

    Arguments:
        batch: (tuple) A tuple of tensor images and lists of annotations

    Return:
        A tuple containing:
            1) (tensor) batch of images stacked on their 0 dim
            2) (list of tensors) annotations for a given image are stacked on
                                 0 dim
    """

    targets = []
    imgs = []
    for sample in batch:
        imgs.append(sample[0])
        targets.append(torch.FloatTensor(sample[1]))
    return torch.stack(imgs, 0), targets


def base_transform(image, size, mean, std):
    x = cv2.resize(image, (size[1], size[0])).astype(np.float32)
    x /= 255.
    x -= mean
    x /= std
    return x


class BaseTransform:
    def __init__(self, size, mean=(0.406, 0.456, 0.485), std=(0.225, 0.224, 0.229)):
        self.size = size
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)

    def __call__(self, image, boxes=None, labels=None):
        return base_transform(image, self.size, self.mean, self.std), boxes, labels
