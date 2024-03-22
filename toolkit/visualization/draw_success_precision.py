import matplotlib.pyplot as plt
import numpy as np

from .draw_utils import COLOR, LINE_STYLE


def draw_success_precision(
    success_ret,
    name,
    videos,
    attr,
    precision_ret=None,
    norm_precision_ret=None,
    bold_name=None,
    axis=[0, 1],
):
    # success plot
    fig, ax = plt.subplots()
    ax.grid()
    ax.set_aspect(1)
    plt.xlabel("Overlap threshold")
    plt.ylabel("Success rate")
    if attr == "ALL":
        plt.title(r"\textbf{Success plots of OPE on %s}" % (name))
    else:
        plt.title(r"\textbf{Success plots of OPE - %s}" % (attr))
    plt.axis([0, 1] + axis)
    success = {}
    thresholds = np.arange(0, 1.05, 0.05)
    for tracker_name in success_ret.keys():
        value = [v for k, v in success_ret[tracker_name].items() if k in videos]
        success[tracker_name] = np.mean(value)
    for idx, (tracker_name, auc) in enumerate(
        sorted(success.items(), key=lambda x: x[1], reverse=True)
    ):
        if tracker_name == bold_name:
            label = r"\textbf{[%.3f] %s}" % (auc, tracker_name)
        else:
            label = "[%.3f] " % (auc) + tracker_name
        value = [v for k, v in success_ret[tracker_name].items() if k in videos]
        plt.plot(
            thresholds,
            np.mean(value, axis=0),
            color=COLOR[idx],
            linestyle=LINE_STYLE[idx],
            label=label,
            linewidth=2,
        )
    ax.legend(loc="lower left", labelspacing=0.2)
    ax.autoscale(enable=True, axis="both", tight=True)
    xmin, xmax, ymin, ymax = plt.axis()
    ax.autoscale(enable=False)
    ymax += 0.03
    plt.axis([xmin, xmax, ymin, ymax])
    plt.xticks(np.arange(xmin, xmax + 0.01, 0.1))
    plt.yticks(np.arange(ymin, ymax, 0.1))
    ax.set_aspect((xmax - xmin) / (ymax - ymin))
    # plt.show()
    plt.savefig(
        f"/mnt/sdc/V4R/LX/2024/HiFT/result_{attr}.png",
        format="png",
        bbox_inches="tight",
    )
