
import time
import cv2 
import numpy as np
import os
import os.path as osp
import math

from tracker.visualize import plot_tracking
from tracker.bytetrack_wrapper import get_bytetrack_tracker
from yolox.configs.yolox_x_mix_det import ConfigXMixDet
# python3 tools/demo_track.py video --path ../data/videos/vlc-record-2022-12-13-11h28m59s-rtsp___192.168.60.13_554_live1.sdp-.mp4 -f exps/example/mot/yolox_x_mix_det.py -c ../pretrained/bytetrack_x_mot17.pth.tar --fp16 --fuse --save_result


def main():
    video = "data/videos/vlc-record-2022-12-13-11h28m59s-rtsp___192.168.60.13_554_live1.sdp-.mp4"
    pretrained = "./pretrained/bytetrack_x_mot17.pth.tar"

    params = ConfigXMixDet()
    params.fp16=False
    wrapper = get_bytetrack_tracker(params, pretrained)

    cap = cv2.VideoCapture(video)
    frame_id = 0
    while True:
        ret, oframe = cap.read()
        if not ret:
            break

        frame = cv2.cvtColor(oframe, cv2.COLOR_BGR2RGB)

        start = time.time()
        online_tlwhs, online_ids, online_scores = wrapper.inference(frame)
        fps = 1/(time.time() - start)

        online_im = plot_tracking(oframe, online_tlwhs, online_ids, frame_id=frame_id, fps=fps)

        cv2.imshow("frame", cv2.resize(online_im, (online_im.shape[1]//2,online_im.shape[0]//2)))
        cv2.waitKey(30)

        frame_id += 1

        print(fps)

    cap.release()



if __name__ == "__main__":
    main()
