from abc import ABC
from typing import Tuple

import torch


class YoloXTrackerConfig(ABC):
    num_classes: int
    depth: float
    width: float
    input_size: Tuple[int, int]
    test_conf: float
    nmsthre: float
    track_thresh: float
    fps: float
    track_buffer: float
    match_thresh: float
    aspect_ratio_thresh: float
    min_box_area: float
    mot20: bool

    device: str
    fp16: bool
    fuse: bool
    trt: bool

    def get_torch_device(self):
        if self.device == "directml":
            raise NotImplementedError()
            import torch_directml

            dml = torch_directml.device()
            return dml
        return torch.device(self.device)
