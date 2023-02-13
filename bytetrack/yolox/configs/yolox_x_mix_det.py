from .base_config import YoloXTrackerConfig


class ConfigXMixDet(YoloXTrackerConfig):
    def __init__(self):
        super().__init__()
        self.num_classes = 1
        self.depth = 1.33
        self.width = 1.25
        self.input_size = (800, 1440)
        self.test_conf = 0.001
        self.nmsthre = 0.7
        self.track_thresh = 0.5
        self.track_buffer =  30
        self.match_thresh = 0.8
        self.aspect_ratio_thresh = 1.6
        self.min_box_area = 10
        self.mot20 = True
        self.device = "cuda"
        self.fp16 = True
        self.fuse = True
        self.trt = False


