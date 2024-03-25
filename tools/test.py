import os
import sys
import argparse

sys.path.append("./")

import cv2
import torch
import numpy as np

from pysot.core.config import cfg
from pysot.models.model_builder import ModelBuilder
from pysot.tracker.prl_tracker import PRLTrack
from pysot.utils.bbox import get_axis_aligned_bbox
from pysot.utils.model_load import load_pretrain
from toolkit.datasets import DatasetFactory

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def main():
    parser = argparse.ArgumentParser(description="PRL-Track tracking")
    parser.add_argument(
        "--dataset", default="UAVTrack112_L", type=str, help="datasets"
    ),
    parser.add_argument(
        "--dataset-root", default="path to test_dataset", type=str, help="dataset root"
    )
    parser.add_argument(
        "--config", default="./experiments/config.yaml", type=str, help="config file"
    )
    parser.add_argument(
        "--snapshot",
        default="./snapshot/best.pth",
        type=str,
        help="snapshot of models to eval",
    )
    parser.add_argument("--video", default="", type=str, help="eval one special video")
    parser.add_argument(
        "--vis", default="", action="store_true", help="whether visualzie result"
    )
    args = parser.parse_args()

    torch.set_num_threads(1)

    cfg.merge_from_file(args.config)

    dataset_root = os.path.join(args.dataset_root, args.dataset)
    model = ModelBuilder()
    model = load_pretrain(model, args.snapshot).cuda().eval()

    tracker = PRLTrack(model)

    dataset = DatasetFactory.create_dataset(
        name=args.dataset, dataset_root=dataset_root, load_img=False
    )

    model_name = args.snapshot.split("/")[-1].split(".")[0]

    for v_idx, video in enumerate(dataset):
        if args.video != "":
            if video.name != args.video:
                continue

        toc = 0
        pred_bboxes = []
        scores = []
        track_times = []
        for idx, (img, gt_bbox) in enumerate(video):
            tic = cv2.getTickCount()
            if idx == 0:
                cx, cy, w, h = get_axis_aligned_bbox(np.array(gt_bbox))
                gt_bbox_ = [cx - (w - 1) / 2, cy - (h - 1) / 2, w, h]
                tracker.init(img, gt_bbox_)
                pred_bbox = gt_bbox_
                scores.append(None)
                if "VOT2018-LT" == args.dataset:
                    pred_bboxes.append([1])
                else:
                    pred_bboxes.append(pred_bbox)
            else:
                outputs = tracker.track(img)
                pred_bbox = outputs["bbox"]
                pred_bboxes.append(pred_bbox)
                scores.append(outputs["best_score"])
            toc += cv2.getTickCount() - tic
            track_times.append((cv2.getTickCount() - tic) / cv2.getTickFrequency())
            if idx == 0:
                cv2.destroyAllWindows()
            if args.vis:
                try:
                    gt_bbox = list(map(int, gt_bbox))
                except:
                    gt_bbox = [0, 0, 0, 0]
                    pred_bbox = list(map(int, pred_bbox))
                    cv2.rectangle(
                        img,
                        (gt_bbox[0], gt_bbox[1]),
                        (gt_bbox[0] + gt_bbox[2], gt_bbox[1] + gt_bbox[3]),
                        (0, 255, 0),
                        3,
                    )
                    cv2.rectangle(
                        img,
                        (pred_bbox[0], pred_bbox[1]),
                        (pred_bbox[0] + pred_bbox[2], pred_bbox[1] + pred_bbox[3]),
                        (0, 255, 255),
                        3,
                    )
                    cv2.putText(
                        img,
                        str(idx),
                        (40, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 255),
                        2,
                    )
                    cv2.imshow(video.name, img)
                    cv2.waitKey(1)
        toc /= cv2.getTickFrequency()

        model_path = os.path.join("results", args.dataset, model_name)
        if not os.path.isdir(model_path):
            os.makedirs(model_path)
        result_path = os.path.join(model_path, "{}.txt".format(video.name))
        with open(result_path, "w") as f:
            for x in pred_bboxes:
                f.write(",".join([str(i) for i in x]) + "\n")
        print(
            "({:3d}) Video: {:12s} Time: {:5.1f}s Speed: {:3.1f}fps".format(
                v_idx + 1, video.name, toc, idx / toc
            )
        )


if __name__ == "__main__":
    main()
