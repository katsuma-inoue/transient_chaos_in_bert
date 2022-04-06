#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import re
import sys
import time
import json
import glob
import parse
import joblib
import argparse
import itertools
import numpy as np
import pandas as pd

sys.path.append(".")

import src.library.style
from pyutils.figure import Figure
import seaborn as sns
from matplotlib.colors import LogNorm
from matplotlib.ticker import FormatStrFormatter


parser = argparse.ArgumentParser()
parser.add_argument("load_dir", type=str)
parser.add_argument("--load_file", type=str, default="error.npy")
parser.add_argument("--regex", type=str, default=".*")
parser.add_argument("--figsize", type=float, nargs=2, default=[8, 6])
parser.add_argument("--ci", type=str, default=None)
parser.add_argument("--hide_legend", action="store_true")
parser.add_argument("--use_cache", action="store_true")
args = parser.parse_args()


def load_data(_dir):
    *_, label = list(filter(None, _dir.split("/")))
    if not re.match(args.regex, label):
        return None
    file_path = "{}/{}".format(_dir, args.load_file)
    if not os.path.exists(file_path):
        return None
    data = np.load(file_path)
    return label, data


if __name__ == '__main__':
    dir_list = glob.glob(f"{args.load_dir}/*,*")
    dir_list = sorted(dir_list)

    raw_data = []
    for _dir in dir_list:
        res = load_data(_dir)
        if res is None:
            continue
        label, result = res
        print(label)
        params = parse.parse("{:d},{:d}", label)
        if params is None:
            continue
        id1, id2 = params
        raw_data.append((id1, id2 - id1, result.mean()))
    df = pd.DataFrame(
        raw_data, columns=["from", "len", "error"])
    df = df[df["len"] >= 20]
    df_pivot = df.pivot("len", "from", "error")
    arr = df_pivot.to_numpy()

    # fig = Figure(figsize=(8, 8))
    # fig[0].plot_matrix(arr, contour=True)
    # fig[0].invert_yaxis()

    import src.library.style

    fig = Figure(figsize=(8, 6))
    ax = sns.heatmap(
        df_pivot, ax=fig[0], xticklabels=20, yticklabels=20,
        norm=LogNorm(df["error"].min(), df["error"].max()), cmap="viridis")
    # ax.set_aspect('equal', 'box')
    ax.set_aspect(1.25)
    min_pos = np.array(
        np.unravel_index(np.nanargmin(arr), arr.shape))[::-1] + 0.5
    ax.scatter(*min_pos, marker="*", s=150.0, color="red")
    ax.invert_yaxis()
    print(df.loc[df["error"].idxmin()])
    fig.show()
    fig.savefig("../result/short_rc_task/heatmap.pdf", transpose=True)

    # x_format = ax.xaxis.get_major_formatter()
    # x_format.seq = ["{:0.1f}".format(float(s)) for s in x_format.seq]
    # y_format = ax.yaxis.get_major_formatter()
    # y_format.seq = ["{:0.1f}".format(float(s)) for s in y_format.seq]
    # ax.xaxis.set_major_formatter(x_format)
    # ax.yaxis.set_major_formatter(y_format)
