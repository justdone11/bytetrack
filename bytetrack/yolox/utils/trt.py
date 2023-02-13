from loguru import logger

import tensorrt as trt
import torch
from torch2trt import torch2trt

import argparse
import os
import shutil
from tracker.bytetrack_wrapper import get_bytetrack_tracker
from yolox.builder import get_detector

from yolox.configs.yolox_x_mix_det import ConfigXMixDet


def make_parser():
    parser = argparse.ArgumentParser("YOLOX ncnn deploy")
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    parser.add_argument("-n", "--name", type=str, default=None, help="model name")

    parser.add_argument(
        "-f",
        "--exp_file",
        default=None,
        type=str,
        help="pls input your expriment description file",
    )
    parser.add_argument("-c", "--ckpt", default=None, type=str, help="ckpt path")
    return parser


@logger.catch
def main():
    # args = make_parser().parse_args()
    parameters = ConfigXMixDet()
    model = get_detector(
        depth=parameters.depth,
        width=parameters.width,
        num_classes=parameters.num_classes,
    )
    ckpt_file = "./pretrained/bytetrack_x_mot17.pth.tar"
    file_name = os.path.join("pretrined")
    os.makedirs(file_name, exist_ok=True)
    # if args.ckpt is None:
    #     ckpt_file = os.path.join(file_name, "best_ckpt.pth.tar")
    # else:
    #     ckpt_file = args.ckpt

    ckpt = torch.load(ckpt_file, map_location="cpu")
    # load the model state dict

    model.load_state_dict(ckpt["model"])
    logger.info("loaded checkpoint done.")
    model.eval()
    model.cuda()
    model.head.decode_in_inference = False
    x = torch.ones(1, 3, parameters.input_size[0], parameters.input_size[1]).cuda()
    model_trt = torch2trt(
        model,
        [x],
        fp16_mode=True,
        log_level=trt.Logger.INFO,
        max_workspace_size=(1 << 32),
    )
    torch.save(model_trt.state_dict(), os.path.join(file_name, "model_trt.pth"))
    logger.info("Converted TensorRT model done.")
    engine_file = os.path.join(file_name, "model_trt.engine")
    # engine_file_demo = os.path.join("deploy", "TensorRT", "cpp", "model_trt.engine")
    with open(engine_file, "wb") as f:
        f.write(model_trt.engine.serialize())

    # shutil.copyfile(engine_file, engine_file_demo)

    logger.info("Converted TensorRT model engine file is saved for C++ inference.")


if __name__ == "__main__":
    main()
