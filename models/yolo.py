import torch.nn as nn
import torch
from backbone.resnet import resnet18
from utils.modules import Conv2d, SPP, SAM
import tools


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
        self.SPP = SPP(512, 512)
        self.SAM = SAM(512)
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
        grid_y, grid_x = torch.meshgrid([torch.arange(hs), torch.arange(ws)])  # 注意顺序
        grid_xy = torch.stack([grid_x, grid_y], dim=-1).float()
        grid_xy = grid_xy.view(1, hs * ws, 2)  # 得到图片每个grid左上角的坐标 [0,1]-[hs-1, ws-1]
        return grid_xy

    def forward(self, data, target=None):
        out = self.backbone(data)
        out = self.SPP(out)
        out = self.SAM(out)
        out = self.conv_set(out)

        pred = self.pred(out).view(out.shape[0], 1 + self.num_classes + 4, -1)
        Batch, HW, C = pred.shape

        conf = pred[:, :, :1]  # 置信度
        class_prob_pred = pred[:, :, 1: self.num_classes + 1]  # 类别概率
        box = pred[:, :, self.num_classes + 1:]  # box中心位置,偏移量
        print(conf.shape)
        if self.trainable:
            conf_loss, class_loss, box_loss, loss = tools.loss(pred_conf=conf, pred_cls=class_prob_pred,
                                                               pred_txtytwth=box, label=target)
            return conf_loss, class_loss, box_loss, loss
        else:
            conf = torch.sigmoid(conf)
            box = torch.clamp()
            class_pred = torch.softmax(class_prob_pred, 1) * conf
            box, scores, class_pred = self.process(box, class_pred)
            return box, scores, class_pred

    def set_grid(self, input_size):
        self.input_size = input_size
        self.grid_cell = self.create_grid(input_size)

    def decode_boxes(self, pred):
        output = torch.zeros_like(pred)  # 得到的pred也是按照grid顺序的
        pred[:, :, 2] = torch.sigmoid(pred[:, :, 2]) + self.grid_cell
        pred[:, :, 2:] = torch.exp(pred[:, :, 2:])

        output[:, :, 0] = pred[:, :, 0] * self.stride - pred[:, :, 2] / 2
        output[:, :, 1] = pred[:, :, 1] * self.stride - pred[:, :, 3] / 2
        output[:, :, 2] = pred[:, :, 0] * self.stride + pred[:, :, 2] / 2
        output[:, :, 3] = pred[:, :, 1] * self.stride + pred[:, :, 3] / 2
        return output

    def nms(self, boxes, scores):
        order = torch.argsort(scores)[::-1].squeeze()  # 大到小
        keep = []
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)  # 所有box的面积
        while order.size()[1] > 0:
            i = order[0]
            keep.append(i)
            min_x = torch.maximum(x1[i], x1[order[1:]])  # 只与score比它小的框计算iou
            min_y = torch.maximum(y1[i], y1[order[1:]])
            max_x = torch.maximum(x2[i], x2[order[1:]])
            max_y = torch.maximum(x1[i], x1[order[1:]])
            inner_area = (max_x - min_x) * (max_y - min_y)

            iou = inner_area / (areas[i] + areas[order[1:]] - inner_area)
            index = torch.where(torch.Tensor(iou <= self.nms_thresh))[0]
            # 找到满足阀值要求且得到最大的一个点, 删掉了从起始点开该点中间那些亢余部分, 从该点开始
            # 又因为iou的长度比order小1, 故索引需要加1
            order = order[index + 1]
        return keep

    def process(self, box, class_pred):
        pred = torch.argmax(class_pred, dim=1)
        prob_pred = class_pred[torch.arange(pred.shape[0]), pred]  # 置信度
        scores = prob_pred.copy()

        keep = torch.where(torch.Tensor(scores > self.conf_thresh))  # 先筛除置信度小于阀值的框
        box = box[keep]
        scores = scores[keep]
        pred = pred[keep]
        keep = torch.zeros_like(pred).int()
        for i in range(self.num_classes):
            index = torch.where(torch.Tensor(pred == i))[0]  # where会返回一个元祖,[0]取出索引
            if len(index) == 0:
                continue
            class_box = box[index]
            class_scores = scores[index]
            class_keep = self.nms(class_box, class_scores)
            keep[index[class_keep]] = 1
        box = box[keep]
        pred = pred[keep]
        scores = scores[keep]
        return box, scores, pred
