"""Calculating the effective dimensions of the embedding states 
obtained with src/script/calc_embedding_states.py
"""
#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import gc
import sys
import json
import glob
import argparse
import numpy as np
import pandas as pd
from datetime import datetime
from collections import defaultdict, OrderedDict
from typing import List, Union

import matplotlib.pyplot as plt

sys.path.append(".")

from pyutils.tqdm import tqdm, trange
from pyutils.figure import Figure
from pyutils.interpolate import interp1d, grad

import src.library.style
num_heads = {"xlarge": 16, "large": 16, "base": 12}
num_hidden = {"xlarge": 2048, "large": 1024, "base": 768}

parser = argparse.ArgumentParser()
parser.add_argument("data_type", type=str)
parser.add_argument("--root_dir",
                    type=str,
                    default="../result/embedding_states/")
parser.add_argument("--save_dir",
                    type=str,
                    default="../result/short_effective/")
parser.add_argument("--begin_id", type=int, default=None)
parser.add_argument("--end_id", type=int, default=None)
parser.add_argument("--split_num", type=int, default=1)
parser.add_argument("--node_num", type=int, default=None)
args = parser.parse_args()


def load_data(task_type, time_step, root_dir="../data/embedding_states"):
    states = []
    for _path in sorted(
            glob.glob(f"{root_dir}/{task_type}/states/{time_step}_*.npy")):
        states.append(np.load(_path, allow_pickle=True))
    states = np.concatenate(states, axis=0)
    return states


def effective_dimension(X):
    Xm = X - X.mean(axis=0)
    u, s, vh = np.linalg.svd(Xm, full_matrices=False)
    eigs = abs(s**2)
    eigs *= 1 / sum(eigs)
    return 1 / (sum(eigs**2))


if __name__ == '__main__':
    save_name = f"{args.save_dir}/{args.data_type.replace('/', '_')}"
    eff_dims = []
    for _t in trange(500):
        states = load_data(f"{args.data_type}", _t)
        eff_dim = effective_dimension(states.reshape(states.shape[0], -1))
        eff_dims.append(eff_dim)
    eff_dims = np.array(eff_dims)
    np.save(f"{save_name}.npy", eff_dims)
    fig = Figure()
    fig[0].plot(eff_dims)
    fig[0].set_yscale("log")
    fig.savefig(f"{save_name}.png", dpi=200)
