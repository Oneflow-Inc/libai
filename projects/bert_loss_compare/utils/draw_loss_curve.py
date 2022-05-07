import argparse
import os
from datetime import date
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
    parser = argparse.ArgumentParser(description="Draw loss curve.")
    parser.add_argument(
        "--torch-loss-path",
        type=str,
        default="projects/bert_loss_compare/torch_loss.txt",
        help="torch loss file path",
    )
    parser.add_argument(
        "--of-loss-path",
        type=str,
        default="loss_align/bert_loss_compare",
        help="oneflow loss file path",
    )
    parser.add_argument(
        "--compare-item", type=int, default=1000, help="number of loss items to be compared"
    )
    args = parser.parse_args()

    with open(args.torch_loss_path, "r") as f:
        torch_total_loss = [float(line) for line in f.readlines()][: args.compare_item]

    import glob

    of_loss_pathes = glob.glob(os.path.join(args.of_loss_path, f"{date.today()}_*"))

    for of_loss_path in of_loss_pathes:
        of_loss_path = os.path.join(of_loss_path, "of_loss.txt")

        with open(of_loss_path, "r") as f:
            flow_total_loss = [float(line) for line in f.readlines()][: args.compare_item]

        draw_result(
            os.path.dirname(of_loss_path),
            f"loss_{args.compare_item}",
            "steps",
            "loss",
            {
                "oneflow": flow_total_loss,
                "torch": torch_total_loss,
            },
        )

        diff = [flow_total_loss[i] - torch_total_loss[i] for i in range(len(flow_total_loss))]

        draw_result(
            os.path.dirname(of_loss_path),
            f"loss_diff_{args.compare_item}",
            "steps",
            "diff",
            {"diff": diff},
        )
