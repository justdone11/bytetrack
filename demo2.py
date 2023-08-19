
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

    folder1 = "data/C8-1/images"
    folder2 = "data/C8-2/images"
    pretrained = "./pretrained/bytetrack_x_mot17.pth.tar"

    params = ConfigXMixDet()
    params.fp16 = False
    wrapper = get_bytetrack_tracker(params, pretrained)

    frame_id = 0

    bbxoes_list_per_frame = []

    for folder in [folder1, folder2]:
        images = os.listdir(folder)

        images.sort(key=lambda x: int((x.split("-")[1] if "-" in x else x).split("/")[-1].split(".")[0]))

        for image in images:

            oframe = cv2.imread(os.path.join(folder, image))

            frame = cv2.cvtColor(oframe, cv2.COLOR_BGR2RGB)

            start = time.time()
            online_tlwhs, online_ids, online_scores = wrapper.inference(frame)
            fps = 1/(time.time() - start)

            online_im, online_bboxes = plot_tracking(oframe, online_tlwhs, online_ids, online_scores, frame_id=frame_id, fps=fps)

            cv2.imshow("frame", cv2.resize(online_im, (online_im.shape[1]//2,online_im.shape[0]//2)))
            cv2.waitKey(1)
            print(frame_id)

            bbxoes_list_per_frame.append((online_tlwhs, online_ids, online_scores)) #to save list of bbxes

            frame_id += 1

            print(fps)

    # bboxes estratte nella forma di tupla ([list of bboxes], [list of id], [list of scores]) per ogni frame (l'indice è il frame).
    # se per un frame non sono presenti bboxes avrò una tupla vuota
    with open("output/bboxes_dump.obj", "wb") as file:
        pickle.dump(bbxoes_list_per_frame, file)


if __name__ == "__main__":
    main()
