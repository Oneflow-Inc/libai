import json
import csv
import sys
import re
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import os


our_times = []
nlr_times = []
no_allo_times = []
no_lr_times = []


cwd = Path.cwd()


def get_theo_time_from_json_file(fn):
    with open(fn) as f:
        j = json.load(f)
        return float(j["theoretically time"])


def get_real_time_from_json_file(fn):
    with open(fn) as f:
        j = json.load(f)
        return float(j["real time"])


def avg(x):
    return sum(x) / len(x)


def get_searching_time_from_json_file(fn):
    with open(fn) as f:
        j = json.load(f)
        j = list(map(lambda x: x[2], j["overhead"]))
        if len(j) == 0:
            return 0
        return sum(j)


def get_mem_frag_rate_from_json_file(fn):
    with open(fn) as f:
        j = json.load(f)
        mem_frag_rate = j["mem frag rate"]
        if mem_frag_rate is None:
            mem_frag_rate = np.nan
        mem_frag_rate = float(mem_frag_rate)
        # return %
        return mem_frag_rate * 100


def get_dataset_time_from_json_file(fn):
    with open(fn) as f:
        j = json.load(f)
        if 'real time' in j:
            x = j["real time"]
            if x != 0:
                return x * 10**6
        x = j["dataset time"]
        if x is not None:
            x = float(x)
        return x


def get_threshold_from_json_file(fn):
    return int(fn[-9:][:4])
    # return int(fn[-10:][:5]) for a800
    if fn[:9] == 'overhead-':
        fn = fn[:-5]
        return int(fn.split('-')[-1])
    with open(fn) as f:
        j = json.load(f)
        t = float(j["threshold"][:-2])
        if t == 9900:
            t = 10000
        return t


model_name = sys.argv[1]


def draw_one(ax, data, label, i, total, kind):
    assert kind in ["step", "line", "bar"]
    colors = {
            'Coop (Ours)': 'tab:blue',
            'Coop': 'tab:blue',
            'DTE': 'tab:orange',
            'DTR': 'tab:green',
            'No op-guided allocation': 'tab:brown',
            'No recomputable in-place': 'tab:purple',
            'No layout-aware eviction': 'tab:red',
            }
    markers = ["o", "*", "D", "^", "s"]
    zorder = 200 - i
    length = len(data)
    data = list(zip(*data))
    if kind == "step":
        x = ax.step(
            *data,
            where="post",
            label=label,
            linewidth=4,
            marker=markers[i],
            markevery=[True] + [False] * (length - 1),
            ms=10,
            zorder=zorder,
            color=colors[label],
        )
    elif kind == "line":
        ax.plot(
            *data,
            label=label,
            linewidth=3,
            marker=markers[i],
            markevery=1,
            ms=10,
            zorder=zorder,
        )
    elif kind == "bar":
        width = 0.25
        x = np.arange(len(data[0]))
        bars = ax.bar(
            x + (i - total / 2 + 0.5) * width,
            np.where(np.isnan(data[1]), 0.5, data[1]),
            width=width * 0.88,
            label=label,
            zorder=zorder,
        )
        for i, bar in enumerate(bars):
            if np.isnan(data[1][i]):
                bar.set_color('gray')


def draw_from_files_and_draw(
    *,
    xlabel,
    ylabel,
    get_y,
    pic_name,
    name_to_legend_and_fn_patterns,
    ncols,
    nrows,
    pic_kind,
    data_kind,
    imgcat=False,
):
    assert len(name_to_legend_and_fn_patterns) == ncols * nrows
    if data_kind in [DK_FRAG, DK_ABLA_FRAG]:
        fig, axs = plt.subplots(ncols=ncols, nrows=nrows, figsize=(14, 6))
    elif data_kind == DK_TIME:
        fig, axs = plt.subplots(ncols=ncols, nrows=nrows, figsize=(28, 9), sharex=True, sharey='row')
    elif data_kind == DK_ABLA:
        fig, axs = plt.subplots(ncols=ncols, nrows=nrows, figsize=(28, 9), sharex=True, sharey='row')
    elif data_kind == DK_OH:
        fig, axs = plt.subplots(ncols=ncols, nrows=nrows, figsize=(14, 6), sharex=True, sharey='row')
    else:
        raise ValueError("data_kind")
    axs = axs.flatten()

    for i, ((name, resolution), legend_and_fn_patterns) in enumerate(name_to_legend_and_fn_patterns.items()):
        ax = axs[i]
        _draw_from_files_and_draw_in_subplot(
            ax=ax,
            xlabel=xlabel,
            ylabel=ylabel,
            get_y=get_y,
            legend_and_fn_pattern=legend_and_fn_patterns,
            pic_kind=pic_kind,
            data_kind=data_kind,
            name=name,
            resolution=resolution,
            index=i,
        )
    left = 0.075 if data_kind in [DK_FRAG, DK_ABLA_FRAG] else 0.06
    if data_kind == DK_OH:
        left = 0
    right = 1
    top = 1
    bottom = {DK_FRAG: 0.15, DK_ABLA: 0.18, DK_TIME: 0.11, DK_OH: 0.15}[data_kind]
    subplot_x_center = (left + right) / 2
    subplot_y_center = (top + bottom) / 2
    if data_kind in [DK_TIME, DK_ABLA]:
        fig.subplots_adjust(top=top, left=left, right=right, bottom=bottom, wspace=0.1, hspace=0.07)
    elif data_kind == [DK_FRAG, DK_OH]:
        fig.subplots_adjust(top=top, left=left, right=right, bottom=bottom, wspace=0.1)
    handles, labels = axs[0].get_legend_handles_labels() # type: ignore
    fig.legend(handles, labels, loc='upper center', ncol=3 if data_kind == DK_FRAG else 4, bbox_to_anchor=(subplot_x_center, 0.01), handlelength=1.5)

    if data_kind == DK_FRAG:
        fig.supylabel(ylabel, y=subplot_y_center, fontsize=YLABEL_FONT_SIZE_FRAG, fontweight='bold')
    else:
        fig.supylabel(ylabel, y=subplot_y_center, fontweight='bold')
    fig.supxlabel(xlabel, x=subplot_x_center, fontweight='bold')

    plt.savefig(pic_name, bbox_inches="tight")
    if imgcat:
        os.system(f"imgcat {pic_name}")


def _draw_from_files_and_draw_in_subplot(
    *, ax, xlabel, ylabel, get_y, legend_and_fn_pattern, pic_kind, data_kind, name, resolution, index
):
    data = {}
    threshold_set = set()
    for label, (fn_pattern, predicate) in legend_and_fn_pattern.items():
        match = re.compile(fn_pattern).match
        fns = list(
            filter(match, (str(x.name) for x in cwd.iterdir()))
        )
        fns = list(filter(lambda fn: predicate(int(match(fn).group(1))), fns))
        thresholds = list(map(get_threshold_from_json_file, fns))
        threshold_set = threshold_set.union(thresholds)
        data[label] = list(zip(thresholds, list(map(get_y, fns))))
    max_threshold = max(threshold_set)
    min_threshold = min(threshold_set)
    if data_kind in [DK_TIME, DK_ABLA]:
        base_time = min(min(x[1] for x in d) for d in data.values())
        # dtr_base_time = min(x[1] for x in data["DTR"])
        for label, d in data.items():
            for i in range(len(d)):
                # if label == 'DTR':
                #     d[i] = (d[i][0], d[i][1] / dtr_base_time)
                # else:
                d[i] = (d[i][0], d[i][1] / base_time)

    if pic_kind == "bar":
        for threshold in threshold_set:
            for label, d in data.items():
                if threshold not in list(zip(*d))[0]:
                    d.append((threshold, np.nan))
    for label in data:
        data[label].sort(key=lambda x: x[0])
        # print(data[label])
        data[label] = list(map(lambda x: (x[0] / max_threshold, x[1]), data[label]))
        # pop max_threshold because it doesn't have mem frag
        # pop min_threshold because most data is None
        if pic_kind == "bar" and data_kind in [DK_FRAG, DK_OH]:
            data[label].pop()
            threshold_set.discard(max_threshold)
            del data[label][0]
            threshold_set.discard(min_threshold)

    # print(f"max_threshold: {max_threshold}")

    for i, (label, d) in enumerate(data.items()):
        draw_one(ax, d, label, i, len(data), pic_kind)

    if name[:5] == 'GPT-2' or name[:4] == 'BERT':
        ax.plot(0.77, 1.25, label='SAR', color='tab:brown', markersize=15, marker='*', linestyle='None')

    if pic_kind in ["step", "line"]:
        if data_kind in [DK_TIME, DK_ABLA]:
            ax.set_xticks(np.arange(0, 1.1, 0.2))
            ax.set_xticks(np.arange(0, 1.1, 0.1), minor=True)
            ax.set_xlim(left=0.15, right=1.1)

            flag = True
            if index >= 4:
                flag = False
            if data_kind == DK_ABLA:
                flag = True
            if flag:
                ax.set_ylim(top=1.57)
                ax.text(1.07, 1.51, name, fontsize=24, fontweight="bold", ha='right')
                ax.text(1.07, 1.49, resolution, fontsize=18, ha='right', va='top')
            else:
                ax.set_ylim(top=1.39)
                ax.text(1.07, 1.34, name, fontsize=24, fontweight="bold", ha='right')
                ax.text(1.07, 1.32, resolution, fontsize=18, ha='right', va='top')

        ax.grid(which='both')
    if pic_kind in ["bar"]:
        x = list(map(lambda x: x / max_threshold, sorted(list(threshold_set))))
        ax.set_xticks(np.arange(len(x)), x)
        ax.grid(axis="y")
        ax.set_title(name, fontsize=21, fontweight='bold', pad=12)


DK_ABLA_FRAG = "abfrag"
DK_FRAG = "frag"
DK_OH = "oh"
DK_TIME = "time"
DK_ABLA = "ablation"
DK_FWS = "fws"
data_kind = sys.argv[3] if len(sys.argv) > 3 else DK_FRAG

DEFAULT_FONT_SIZE = 22
YLABEL_FONT_SIZE_FRAG = 20 if data_kind == DK_FRAG else None
plt.rcParams.update({"font.size": DEFAULT_FONT_SIZE})


def divisible_by(n):
    return lambda x: x % n == 0


def unet_predicate(fn):
    return divisible_by(1000)(fn)

def gpt2_predicate(fn):
    return divisible_by(850)(fn)
    # return divisible_by(7168)(fn)
    # return divisible_by(6500)(fn)

def resnet50_predicate(fn):
    return divisible_by(1000)(fn)


if data_kind == DK_TIME:
    d = {
            # ("GPT-2 (8)", "Sequence length 512"): ("bert-new2", unet_predicate),
            # ("BERT Large (4)", "Sequence length 512"): ("bert-new2", unet_predicate),
            # ("BiLSTM (2048)", "Input dimension 100,\nHidden dimension 256,\nSequence length 128"): ("lstm_text-new2", divisible_by(850)),
            # # ("ResNet-152 (55)", "224x224"): ("resnet152-new2", unet_predicate),
            # ("U-Net (5)", "460x608"): ("unet-new2", unet_predicate),
            # ("Swin-T (40)", "224x224"): ("stn2", unet_predicate),
            # ("ResNet-50 (115)", "224x224"): ("resnet50-new2", unet_predicate),
            # ("Inception V3 (96)", "299x299"): ("inception_v3-new2", unet_predicate),
            # ("DenseNet-121 (70)", "224x224"): ("densenet121-new2", unet_predicate),

            ("gpt", "32x32"): ("gpt2", gpt2_predicate),
            ("gpt1", "32x32"): ("gpt2", gpt2_predicate),
            ("gpt2", "32x32"): ("gpt2", gpt2_predicate),
            ("gpt3", "32x32"): ("gpt2", gpt2_predicate),
            ("gpt4", "32x32"): ("gpt2", gpt2_predicate),
            ("gpt5", "32x32"): ("gpt2", gpt2_predicate),
            ("gpt6", "32x32"): ("gpt2", gpt2_predicate),
            ("gpt7", "32x32"): ("gpt2", gpt2_predicate),
            }
    name_to_legend_and_fn_patterns = {}
    for k, v in d.items():
        if k == ("Swin-T (40)", "224x224"):
            name_to_legend_and_fn_patterns[k] = {
                    "Coop (Ours)": (rf"{v[0]}-ours-(\d{{4,5}}).json", v[1]),
                    "DTE": (rf"{v[0]}-no-gp-(\d{{4,5}}).json", v[1]),
                    "DTR": (rf"{v[0]}-no-fbip-(\d{{4,5}}).json", v[1]),
                }
            continue
        name_to_legend_and_fn_patterns[k] = {
                "Coop (Ours)": (rf"{v[0]}-ours-(\d{{4,5}}).json", v[1]),
                "DTE": (rf"{v[0]}-dte-our-impl-(\d{{4,5}}).json", v[1]),
                "DTR": (rf"{v[0]}-dtr-no-free-(\d{{4,5}}).json", v[1]),
            }

    draw_from_files_and_draw(
        xlabel="Memory Ratio",
        ylabel="Compute Overhead (x)",
        get_y=get_dataset_time_from_json_file,
        pic_name=f"compute-overhead-main.pdf",
        name_to_legend_and_fn_patterns=name_to_legend_and_fn_patterns,
        ncols=4,
        nrows=2,
        imgcat=True,
        pic_kind=sys.argv[2],
        data_kind=data_kind,
    )