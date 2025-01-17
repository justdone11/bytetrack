
import time
import cv2 
import numpy as np
import os
import os.path as osp
import math
import torch
import pickle

# from tracker.visualize import plot_tracking
# from tracker.bytetrack_wrapper import get_bytetrack_tracker
# from yolox.configs.yolox_x_mix_det import ConfigXMixDet
# python3 tools/demo_track.py video --path ../data/videos/vlc-record-2022-12-13-11h28m59s-rtsp___192.168.60.13_554_live1.sdp-.mp4 -f exps/example/mot/yolox_x_mix_det.py -c ../pretrained/bytetrack_x_mot17.pth.tar --fp16 --fuse --save_result
from bytetrack import plot_tracking, get_bytetrack_tracker, ConfigXMixDet


def main():
    print(torch.cuda.is_available())
    video = "data/C9_short.mp4"
    pretrained = "./pretrained/bytetrack_x_mot17.pth.tar"

    params = ConfigXMixDet()
    params.fp16=False
    wrapper = get_bytetrack_tracker(params, pretrained)

    cap = cv2.VideoCapture(video)
    frame_id = 0
    bbxoes_list_per_frame = []

    while True:
        ret, oframe = cap.read()
        if not ret:
            break

        frame = cv2.cvtColor(oframe, cv2.COLOR_BGR2RGB)

        start = time.time()
        online_tlwhs, online_ids, online_scores = wrapper.inference(frame)
        fps = 1/(time.time() - start)

        online_im, online_bboxes = plot_tracking(oframe, online_tlwhs, online_ids, online_scores, frame_id=frame_id, fps=fps)

        #cv2.imshow("frame", cv2.resize(online_im, (online_im.shape[1]//2,online_im.shape[0]//2)))
        #cv2.waitKey(1)
        print(frame_id)

        if len(online_bboxes) > 0:
            bbxoes_list_per_frame.append((frame_id, online_bboxes)) #to save list of bbxes

        frame_id += 1

        print(fps)

    # bboxes estratte nella forma di tupla (frame_id, [list of (id, bbox, score)]) per ogni frame.
    # se per un frame non sono presenti bboxes avrò una tupla vuota
    with open("output/C9_short_bboxes_dump_withScores.obj", "wb") as file:
        pickle.dump(bbxoes_list_per_frame, file)

    cap.release()


if __name__ == "__main__":
    main()
