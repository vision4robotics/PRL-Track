import os
import sys
import argparse

sys.path.append("./")

from tqdm import tqdm
from multiprocessing import Pool

from toolkit.visualization import draw_success_precision
from toolkit.evaluation import OPEBenchmark
from toolkit.datasets import *


parser = argparse.ArgumentParser(description="Single Object Tracking Evaluation")
parser.add_argument(
    "--datasetpath",
    default="./test_dataset/",
    type=str,
    help="dataset root directory",
)
parser.add_argument("--dataset", default="UAV123", type=str, help="dataset name")
parser.add_argument(
    "--tracker_result_dir",
    default="./results",
    type=str,
    help="tracker result root",
)
parser.add_argument("--tracker_path", default="./results", type=str)
parser.add_argument("--vis", default="", dest="vis", action="store_true")
parser.add_argument(
    "--show_video_level", default=True, dest="show_video_level", action="store_true"
)
parser.add_argument("--num", default=1, type=int, help="number of processes to eval")
args = parser.parse_args()


def main():
    tracker_dir = os.path.join(args.tracker_path, args.dataset)
    trackers = os.listdir(tracker_dir)

    root = args.datasetpath + args.dataset

    assert len(trackers) > 0
    args.num = min(args.num, len(trackers))

    if "UAV123_10fps" in args.dataset:
        dataset = UAV10Dataset(args.dataset, root)
    elif "UAV123_20L" in args.dataset:
        dataset = UAV20Dataset(args.dataset, root)
    elif "UAV123" in args.dataset:
        dataset = UAVDataset(args.dataset, root)
    elif "DTB70" in args.dataset:
        dataset = DTBDataset(args.dataset, root)
    elif "UAVDT" in args.dataset:
        dataset = UAVDTDataset(args.dataset, root)
    elif "VISDRONED" in args.dataset:
        dataset = VISDRONEDDataset(args.dataset, root)
    elif "UAVTrack112_L" in args.dataset:
        dataset = UAV112LDataset(args.dataset, root)
    elif "UAVTrack112" in args.dataset:
        dataset = UAV112Dataset(args.dataset, root)

    dataset.set_tracker(tracker_dir, trackers)
    benchmark = OPEBenchmark(dataset)
    success_ret = {}
    with Pool(processes=args.num) as pool:
        for ret in tqdm(
            pool.imap_unordered(benchmark.eval_success, trackers),
            desc="eval success",
            total=len(trackers),
            ncols=18,
        ):
            success_ret.update(ret)

    norm_precision_ret = {}
    with Pool(processes=args.num) as pool:
        for ret in tqdm(
            pool.imap_unordered(benchmark.eval_norm_precision, trackers),
            desc="eval norm precision",
            total=len(trackers),
            ncols=25,
        ):
            norm_precision_ret.update(ret)

    precision_ret = {}
    with Pool(processes=args.num) as pool:
        for ret in tqdm(
            pool.imap_unordered(benchmark.eval_precision, trackers),
            desc="eval precision",
            total=len(trackers),
            ncols=20,
        ):
            precision_ret.update(ret)

    benchmark.show_result(
        success_ret,
        precision_ret,
        norm_precision_ret,
        show_video_level=args.show_video_level,
    )

    if args.vis:
        for attr, videos in dataset.attr.items():
            draw_success_precision(
                success_ret,
                name=dataset.name,
                videos=videos,
                attr=attr,
                precision_ret=precision_ret,
            )


if __name__ == "__main__":
    main()
