"""Calculate the local Lyapunov exponent.
"""
#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import gc
import sys
import json
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
from datetime import datetime
from collections import defaultdict, OrderedDict
from typing import List, Union

sys.path.append(".")

from pyutils.tqdm import tqdm, trange
from pyutils.figure import Figure
from pyutils.parallel import multi_process, multi_thread, for_each

import src.library.style
import seaborn as sns

os.environ["USE_BERT_ATTENTION_PROJECT"] = "0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from src.library.model.albert_system import AlbertSystem
from src.library.utils import wiki_random_extractor
from src.library.utils.data_loader import STSDataLoader, QQPDataLoader, WikipediaDataLoader
from src.library.utils.target_loader import load_curve

num_heads = {"xlarge": 16, "large": 16, "base": 12}
num_hidden = {"xlarge": 2048, "large": 1024, "base": 768}

physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    for k in range(len(physical_devices)):
        tf.config.experimental.set_memory_growth(physical_devices[k], True)
        print('memory growth:',
              tf.config.experimental.get_memory_growth(physical_devices[k]))
else:
    print("Not enough GPU hardware devices available")

parser = argparse.ArgumentParser()
parser.add_argument("--root_dir", type=str, default="../result/long_lle/")
parser.add_argument("--save_name", type=str, default=None)
parser.add_argument("--albert_size",
                    type=str,
                    default="large",
                    choices=["base", "large", "xlarge", "xxlarge"])
parser.add_argument("--albert_version", type=str, default="2")
parser.add_argument("--init_model", action="store_true")
parser.add_argument("--max_seq_len", type=int, default=32)
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--epsilon", type=float, default=1e-0)
parser.add_argument("--tau", type=int, default=50)
parser.add_argument("--T", type=int, default=1010)
parser.add_argument("--batch_size", type=int, default=100)
parser.add_argument("--sample_size", type=int, default=1000)
args = parser.parse_args()


def calc_lle(net, tau, T, inputs, epsilon=1e-6):
    net.set_input(**inputs)
    batch_size, *state_shape = net._embedding_state.shape
    norm_axis = tuple(range(-len(state_shape), 0))
    pert = tf.random.normal((batch_size, *state_shape))
    pert, _norm = tf.linalg.normalize(pert, axis=norm_axis)
    net._embedding_state = tf.concat(
        [net._embedding_state, net._embedding_state + pert * epsilon], axis=0)
    lyap_list = []
    state_list = []
    pbar = trange(int(T / tau), leave=False)
    for _iter in pbar:
        for _t in range(tau):
            net._embedding_state = \
                net.albert.encoders_layer.shared_layer(
                    net._embedding_state,
                    mask=None, training=False)
            _state = net._embedding_state[:, 0, 0].numpy()
            state_list.append(_state)
        diff = net._embedding_state[
            batch_size:] - net._embedding_state[:batch_size]
        diff_norm, _norm = tf.linalg.normalize(diff, axis=norm_axis)
        net._embedding_state = tf.concat([
            net._embedding_state[:batch_size],
            net._embedding_state[:batch_size] + diff_norm * epsilon
        ],
                                         axis=0)
        _lyap = _norm.numpy().flatten()
        _lyap = np.log(_lyap / epsilon) / tau
        lyap_list.append(_lyap)
    lyap_list = np.array(lyap_list)
    state_list = np.array(state_list)
    return lyap_list, state_list


def calc_lle_2(net, tau, T, inputs, epsilon=1e-6):
    net.set_input(**inputs)
    lyap_list, diff, states = net.maximum_lyapunov(
        tau,
        T,
        epsilon=epsilon,
        callback_gpu=lambda x: x.cpu(),
        save_state=True,
        save_one_dim=True,
        save_one_token=False,
        debug=True)
    return lyap_list.T, states


if __name__ == '__main__':
    if args.save_name is None:
        save_name = "init" if args.init_model else "pretrained"
    else:
        save_name = args.save_name
    save_dir = "{}/{}/{},{},{:.1e}".format(args.root_dir, save_name, args.tau,
                                           args.T, args.epsilon)
    os.makedirs(save_dir, exist_ok=True)
    with open(f"{save_dir}/args.json", mode="w") as f:
        json.dump(args.__dict__, f, indent=4)

    net = AlbertSystem(f"albert_{args.albert_size}",
                       args.albert_version,
                       args.max_seq_len,
                       pretrained_weights=not (args.init_model))
    data_loader = WikipediaDataLoader(net.tokenizer,
                                      args.sample_size,
                                      args.batch_size,
                                      args.max_seq_len,
                                      shuffle=False,
                                      seed=args.seed,
                                      sentence_length=2,
                                      data_dir="../data")
    datasets = tqdm(enumerate(data_loader), total=len(data_loader))

    lles, states, indices = [], [], []
    for idx, (ids, inputs) in datasets:
        lle, state = calc_lle(net,
                              args.tau,
                              args.T,
                              inputs,
                              epsilon=args.epsilon)
        lles.append(lle)
        states.append(state)
        indices.append(ids)
    lles = np.concatenate(lles, axis=1)
    states = np.concatenate(states, axis=1)
    indices = np.concatenate(indices)
    np.save(f"{save_dir}/lles.npy", lles)
    np.save(f"{save_dir}/states.npy", states)
    np.save(f"{save_dir}/indices.npy", indices)
    fig = Figure(figsize=(8, 6), grid=(2, 1))
    ts = np.arange(0, lles.shape[0]) * args.tau
    fig[0].fill_std(ts, lles.mean(axis=1), lles.std(axis=1))
    ts = np.arange(0, states.shape[0])
    fig[1].fill_std(ts, states.mean(axis=1), states.std(axis=1))
    fig.savefig(f"{save_dir}/results.png", dpi=200)
