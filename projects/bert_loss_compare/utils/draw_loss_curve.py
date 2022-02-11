import os
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np


def draw_result(
    save_dir: str,
    name,
    xlabel: str,
    ylabel: str,
    data: Dict[str, np.ndarray],
) -> None:
    # Setup matplotlib
    plt.rcParams["figure.dpi"] = 100
    plt.clf()
    for data_name, values in data.items():
        axis = np.arange(1, len(values) + 1)
        # Draw Line Chart
        plt.plot(axis, values, "-", linewidth=1.5, label=data_name)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(loc="best", frameon=True, fontsize=8)
    plt.savefig(os.path.join(save_dir, name + ".png"))


if __name__ == "__main__":
    with open("of_bert_loss.txt", "r") as f:
        flow_total_loss = [float(line) for line in f.readlines()]
    with open("/workspace/idea_model/idea_bert/megatron_bert_loss.txt", "r") as f:
        megatron_total_loss = [float(line) for line in f.readlines()]

    draw_result(
        "./",
        "of_meg_loss",
        "steps",
        "loss",
        {
            "oneflow": flow_total_loss,
            "megatron": megatron_total_loss,
        },
    )

    diff = [flow_total_loss[i] - megatron_total_loss[i] for i in range(len(flow_total_loss))]

    draw_result("./", "of_meg_diff", "steps", "diff", {"diff": diff})
