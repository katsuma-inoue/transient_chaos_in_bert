"""Evaluating the reconstruction errors of the handwriting task.
See the supplementary materials of our paper for the detailed setup
of the handwriting task.
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

sys.path.append(".")

from pyutils.tqdm import tqdm, trange
from pyutils.figure import Figure
from pyutils.parallel import for_each, multi_thread, multi_process
from pyutils.interpolate import interp1d, grad

parser = argparse.ArgumentParser()
parser.add_argument("--save_dir", type=str, default="../result/short_rc_task")
parser.add_argument("--data_dir", type=str, default="../data/short_rc_task")
parser.add_argument("--task_type",
                    type=str,
                    default="STS-B",
                    choices=["STS-B", "QQP"])
parser.add_argument("--state_type", type=str, default="embedding_state")
parser.add_argument("--token_num", type=int, default=1)
parser.add_argument("--write_mode", type=int, default=0, choices=[0, 1])
parser.add_argument("--threshold", type=float, nargs=2, default=None)
parser.add_argument("--step_num", type=int, default=200)
parser.add_argument("--save_token_num", type=int, default=1)
parser.add_argument("--save_head_num", type=int, default=1)

parser.add_argument("--begin_id", type=int, default=0)
parser.add_argument("--end_id", type=int, default=None)
parser.add_argument("--split_num", type=int, default=1)
parser.add_argument("--node_num", type=int, default=None)
parser.add_argument("--debug", action="store_true")
args = parser.parse_args()


def load_data(paths, key="embedding_state"):
    labels = []
    indices = []
    for _path in tqdm(paths):
        npz = np.load(_path)
        labels.append(npz["label"])
        indices.append(npz["ids"])
    labels = np.concatenate(labels)
    indices = np.concatenate(indices)
    num_data = labels.shape[0]
    states = None
    pos1, pos2 = 0, 0
    for _path in tqdm(paths):
        npz = np.load(_path)
        state = npz[key]
        if states is None:
            states = np.zeros((num_data, *state.shape[1:]), dtype=state.dtype)
        pos2 = pos1 + state.shape[0]
        states[pos1:pos2] = state
        pos1 = pos2
    return indices, labels, states


def load_curve(character, draw_step=200, csv_dir="../data/learning_curve/csv"):
    csv_path = os.path.join(csv_dir, character + ".csv")
    df = pd.read_csv(csv_path)
    data = df[["x", "y"]].to_numpy()
    func = interp1d(data, axis=0)
    dfunc = grad(func, dx=1e-5)
    drec = dfunc(np.linspace(0, 1, draw_step))
    rec = np.cumsum(drec, axis=0)
    return drec, rec


def post_process(task_type, layer_range, dim_range, indices, labels, states):
    if task_type == "STS-B" and (args.threshold is not None):
        selected_data = (labels <= args.threshold[0]) | (labels >=
                                                         args.threshold[1])
        I_data = indices[selected_data].astype(int)
        L_data = (labels[selected_data] > 2.5).astype(int)
    else:
        selected_data = np.full(labels.shape, True)
        I_data = indices[selected_data].astype(int)
        L_data = labels[selected_data].astype(int)
    X_data = states[selected_data, layer_range, dim_range]
    X_data = X_data.reshape(*X_data.shape[:2], -1)
    X_data = np.concatenate([X_data, np.ones((*X_data.shape[:2], 1))], axis=-1)
    return I_data, L_data, X_data


def normalized_mse(out, desire, axis=0):
    axes = tuple(_ for _ in range(out.ndim) if _ != axis)
    err = out - desire
    return np.sum(err**2, axis=axes) / np.sum(desire**2, axis=axes)


if __name__ == '__main__':
    step_num = args.step_num
    token_num, head_num = args.save_token_num, args.save_head_num
    root_dir = f"../data/short_rc_task/{args.task_type}"
    train_dir = f"{root_dir}/train({step_num},{token_num},{head_num})"
    eval_dir = f"{root_dir}/dev({step_num},{token_num},{head_num})"
    train_path = list(sorted(glob.glob(f"{train_dir}/*.npz")))
    eval_path = list(sorted(glob.glob(f"{eval_dir}/*.npz")))

    index_t, label_t, state_t = load_data(train_path, args.state_type)
    index_e, label_e, state_e = load_data(eval_path, args.state_type)

    if args.threshold is None:
        base_dir = "{}/continuous".format(args.save_dir)
    else:
        base_dir = "{}/{:.1f},{:.1f}".format(args.save_dir, *args.threshold)
    os.makedirs(base_dir, exist_ok=True)
    with open(f"{base_dir}/args.json", mode="w") as f:
        json.dump(args.__dict__, f, indent=4)
    char_list = ["U", "S"]

    def run(_id1, _id2):
        dim_range = slice(None, 1)
        layer_range = slice(_id1, _id2)
        save_dir = "{}/{:02d},{:02d}".format(base_dir, _id1, _id2)
        print(save_dir)
        if args.debug:
            return
        os.makedirs(save_dir, exist_ok=True)
        I_train, L_train, X_train = post_process(args.task_type, layer_range,
                                                 dim_range, index_t, label_t,
                                                 state_t)
        I_eval, L_eval, X_eval = post_process(args.task_type, layer_range,
                                              dim_range, index_e, label_e,
                                              state_e)

        Y_raw = np.array([
            load_curve(_c, X_train.shape[1],
                       csv_dir="../data/arial/csv/")[args.write_mode]
            for _c in char_list
        ])
        if args.task_type == "STS-B" and args.threshold is None:
            Y_train = np.einsum("ijk,in->njk", Y_raw,
                                np.array([1 - L_train / 5, L_train / 5]))
            Y_eval = np.einsum("ijk,in->njk", Y_raw,
                               np.array([1 - L_eval / 5, L_eval / 5]))
        else:
            Y_train = Y_raw[L_train]
            Y_eval = Y_raw[L_eval]
        solution, _res, _rank, _s = np.linalg.lstsq(
            X_train.reshape(-1, X_train.shape[-1]),
            Y_train.reshape(-1, Y_train.shape[-1]),
            rcond=None)
        Y_out = X_eval.dot(solution)
        error = normalized_mse(Y_out, Y_eval)
        if args.write_mode == 0:
            Z_raw = np.cumsum(Y_raw, axis=1)
            Z_out = np.cumsum(Y_out, axis=1)
        else:
            Z_raw = Y_raw
            Z_out = Y_out

        score = [0.0, 5.0]
        fig = Figure(figsize=(18, 6), grid=(1, 3))
        for _t in [0, 1]:
            fig[_t].set_aspect("equal", "datalim")
            fig[_t].plot(Z_raw[_t, :, 0], Z_raw[_t, :, 1], color="k", ls=":")
            if args.task_type == "STS-B" and args.threshold is None:
                ids = (L_eval == score[_t])
            else:
                ids = (L_eval == _t)
            index = np.argsort(error[ids])
            Z_now = Z_out[ids]
            I_now = I_eval[ids]
            for _ in range(5):
                fig[_t].plot(Z_now[index[_], :, 0],
                             Z_now[index[_], :, 1],
                             label=str(I_now[index[_] - 1]))
            fig[_t].legend()
        fig[2].set_title("error={:.3e}".format(error.mean()))
        fig[2].hist(error)
        fig.savefig(f"{save_dir}/out.png", dpi=200)
        fig.close()
        np.save(f"{save_dir}/error.npy", error)
        np.save(f"{save_dir}/solution.npy", solution)

    min_step, max_step = 1, 101
    arg_list = []
    for _i1 in range(0, 101, 1):
        # for _i2 in range(100, 101, min_step):
        for _i2 in range(_i1 + min_step, _i1 + max_step, 1):
            arg_list.append((_i1, _i2))

    if args.node_num is None:
        for_each(run, arg_list, expand=True, verbose=False)
    else:
        multi_process(run,
                      arg_list,
                      expand=True,
                      verbose=False,
                      nodes=args.node_num)
