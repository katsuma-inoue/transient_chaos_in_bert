"""Solve STS-B tasks with from 'layer_min' ALBERT encoder layers to 'layer_max' layers.
"""
#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import json
import glob
import string
import argparse
import platform
import itertools
import subprocess
import numpy as np

sys.path.append(".")

from pyutils.parallel import multi_process, for_each

parser = argparse.ArgumentParser()
parser.add_argument("--begin_id", type=int, default=0)
parser.add_argument("--end_id", type=int, default=None)
parser.add_argument("--split_num", type=int, default=1)
parser.add_argument("--node_num", type=int, default=None)

# experimental options
parser.add_argument("--exp_id", type=int, default=0)
parser.add_argument('--task_type', type=str, default='STS-B')
parser.add_argument('--layer_min', type=int, default=1)
parser.add_argument('--layer_max', type=int, default=100)
parser.add_argument('--fix_encoder', action="store_true")
parser.add_argument('--init_model', action="store_true")
parser.add_argument("--debug", action="store_true")
args = parser.parse_args()

epochs = {
    "CoLA": 3,
    "MNLI": 3,
    "MRPC": 3,
    "SST-2": 3,
    "MRPC": 3,
    "STS-B": 10,
    "QQP": 3,
    "QNLI": 3,
    "RTE": 3,
    "WNLI": 3
}


def lr_func():
    if args.fix_encoder:
        return 1e-3
    else:
        return 1e-6


if __name__ == '__main__':

    def _run(_id, num_layer, worker_id=None):
        output_dir = "../result/glue/{}/{}/{}/{}/{}".format(
            args.task_type, num_layer, ["pre_trained",
                                        "init"][args.init_model],
            ["train_encoder", "fix_encoder"][args.fix_encoder], args.exp_id)
        if os.path.exists(output_dir):
            return
        cmd = "python src/library/glue/glue.py "
        cmd += "--model_type albert "
        cmd += "--model_name_or_path albert-large-v2 "
        cmd += "--task_name {} ".format(args.task_type)
        cmd += "--do_train --do_eval --do_lower_case "
        cmd += "--data_dir ../data/glue_data/{} ".format(args.task_type)
        cmd += "--max_seq_length 64 "
        cmd += "--per_gpu_eval_batch_size 16 "
        cmd += "--per_gpu_train_batch_size 8 "
        cmd += "--num_train_epochs {} ".format(epochs[args.task_type])
        cmd += "--learning_rate {} ".format(lr_func())
        cmd += "--overwrite_output_dir "
        cmd += "--output_dir {} ".format(output_dir)
        cmd += "--num_layer {} ".format(num_layer)
        cmd += "--fix_encoder {} ".format(int(args.fix_encoder))
        cmd += "--init_model {} ".format(int(args.init_model))
        cmd += "--warmup_steps 0 "
        cmd += "--save_steps 1000000 "

        print("[{}] : {}".format(_id, cmd))
        if not args.debug:
            subprocess.call(cmd.split())
        # stderr=subprocess.DEVNULL)
        # , stdout=subprocess.DEVNULL,

    layers = np.arange(args.layer_min, args.layer_max + 1, 1)
    arg_list = list(itertools.product(layers))
    arg_list = [(_id, ) + _ for _id, _ in enumerate(arg_list)]
    arg_list = arg_list[args.begin_id:args.end_id:args.split_num]

    if args.node_num is None:
        for_each(_run, arg_list, expand=True, verbose=False)
    else:
        multi_process(_run,
                      arg_list,
                      verbose=False,
                      append_id=True,
                      expand=True,
                      nodes=args.node_num)
