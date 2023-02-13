import torch
from .model import YOLOPAFPN, YOLOX, YOLOXHead


def get_detector(
    depth: float = 1.00,
    width: float = 1.00,
    num_classes: int = 80
):
    def init_yolo(M):
        for m in M.modules():
            if isinstance(m, torch.nn.BatchNorm2d):
                m.eps = 1e-3
                m.momentum = 0.03

    in_channels = [256, 512, 1024]
    backbone = YOLOPAFPN(depth, width, in_channels=in_channels)
    head = YOLOXHead(num_classes, width, in_channels=in_channels)
    model = YOLOX(backbone, head)

    model.apply(init_yolo)
    model.head.initialize_biases(1e-2)
    return model
