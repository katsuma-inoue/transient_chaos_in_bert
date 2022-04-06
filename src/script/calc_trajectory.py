#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import gc
import csv
import sys
import json
import joblib
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

os.environ["USE_BERT_ATTENTION_PROJECT"] = "0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from src.library.model.albert_system import AlbertSystem
from src.library.utils.target_loader import load_curve
from src.library.utils.data_loader import QQPDataLoader, STSDataLoader

physical_devices = tf.config.experimental.list_physical_devices("GPU")
if len(physical_devices) > 0:
    for k in range(len(physical_devices)):
        tf.config.experimental.set_memory_growth(physical_devices[k], True)
        print("memory growth:",
              tf.config.experimental.get_memory_growth(physical_devices[k]))
else:
    print("Not enough GPU hardware devices available")

num_heads = {"xlarge": 16, "large": 16, "base": 12}
num_hidden = {"xlarge": 2048, "large": 1024, "base": 768}

parser = argparse.ArgumentParser()
parser.add_argument("--save_dir", type=str, default="../data/task_handwrite")
parser.add_argument("--load_dir", type=str, default="../data/glue_data")
parser.add_argument("--task_type", type=str, default="STS-B",
                    choices=["STS-B", "QQP"])
parser.add_argument("--csv_type", type=str, default="train",
                    choices=["train", "dev", "test"])
parser.add_argument("--albert_size", type=str, default="large",
                    choices=["base", "large", "xlarge", "xxlarge"])
parser.add_argument("--albert_version", type=str, default="2")
parser.add_argument("--init_model", action="store_true")
parser.add_argument("--max_seq_len", type=int, default=64)
parser.add_argument("--batch_size", type=int, default=256)
parser.add_argument("--step_num", type=int, default=100)
parser.add_argument("--save_token_num", type=int, default=4)
parser.add_argument("--save_head_num", type=int, default=1)
parser.add_argument("--begin_id", type=int, default=0)
parser.add_argument("--end_id", type=int, default=None)
parser.add_argument("--split_num", type=int, default=1)
parser.add_argument("--node_num", type=int, default=None)
args = parser.parse_args()


def load_sts_data(path, tokenizer, batch_size, max_seq_len, **kwargs):
    sts_cols = [
        "index", "genre", "filename", "year", "old_index", "source1",
        "source2", "sentence1", "sentence2", "score"]
    df = pd.read_csv(path, delimiter='\t', names=sts_cols, quoting=csv.QUOTE_NONE, encoding='utf-8')
    df["score"] = pd.to_numeric(df["score"], errors="coerce")
    df = df[df["score"].notna()]
    return STSDataLoader(
        df, tokenizer, batch_size=batch_size,
        max_seq_len=max_seq_len, shuffle=False)


def load_qqp_data(path, tokenizer, batch_size, max_seq_len, **kwargs):
    sts_cols = ["id", "qid1", "qid2", "question1", "question2", "is_duplicate"]
    # loading dataset.
    df = pd.read_csv(path, delimiter="\t", names=sts_cols, quoting=csv.QUOTE_NONE, encoding='utf-8')
    df["is_duplicate"] = pd.to_numeric(df["is_duplicate"], errors="coerce")
    df = df[df["is_duplicate"].notna()]
    df["is_duplicate"] = df["is_duplicate"].astype(int)
    return QQPDataLoader(
        df, tokenizer, batch_size=batch_size,
        max_seq_len=max_seq_len, shuffle=False)


def emulate(net, step_num, inputs):
    net.set_input(**inputs)
    record = defaultdict(list)
    record["embedding_state"].append(net._embedding_state[:, :args.save_token_num])
    for _t in trange(step_num, leave=False):
        net._embedding_state = \
            net.albert.encoders_layer.shared_layer(
                net._embedding_state,
                mask=None, training=False)
        # net._attention_state = int_vals["attention_probs"]
        record["embedding_state"].append(net._embedding_state[:, :args.save_token_num])
        # record["attention_state"].append(net._attention_state[:, :args.save_head_num])
    for _key, _val in record.items():
        record[_key] = tf.stack(_val, 1).numpy()
    return record


if __name__ == "__main__":
    net = AlbertSystem(
        f"albert_{args.albert_size}", args.albert_version,
        args.max_seq_len, pretrained_weights=~args.init_model)

    if args.task_type == "STS-B":
        load_func = load_sts_data
    elif args.task_type == "QQP":
        load_func = load_qqp_data

    data_loader = load_func(
        f"../data/glue_data/{args.task_type}/{args.csv_type}.tsv",
        net.tokenizer, args.batch_size, args.max_seq_len)
    data_size = data_loader.ids.shape[0]
    save_path = f"{args.save_dir}/{args.task_type}/{args.csv_type}"
    save_path += "({:d},{:d},{:d})".format(
        args.step_num, args.save_token_num, args.save_head_num)
    os.makedirs(save_path, exist_ok=True)
    with open(f"{save_path}/args.json", mode="w") as f:
        json.dump(args.__dict__, f, indent=4)

    iters = [_ for _ in enumerate(tqdm(data_loader, leave=True))]
    pbar = tqdm(iters[args.begin_id:args.end_id:args.split_num],
                desc=f"{args.task_type}/{args.csv_type}")
    for _i, (ids, inputs, labels) in pbar:
        record = emulate(net, args.step_num, inputs)
        record["ids"] = ids
        record["label"] = labels
        np.savez("{}/{:05d}.npz".format(save_path, _i), **record)
