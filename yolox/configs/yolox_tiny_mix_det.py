from yolox.configs.base_config import YoloXTrackerConfig


class ConfigTinyMixDet(YoloXTrackerConfig):
    def __init__(self):
        super().__init__()
        self.num_classes = 1
        self.depth = 0.33
        self.width = 0.375
        self.scale = (0.5, 1.5)
        self.input_size = (608, 1088)
        self.test_size = (608, 1088)
        self.test_conf = 0.001
        self.nmsthre = 0.7
