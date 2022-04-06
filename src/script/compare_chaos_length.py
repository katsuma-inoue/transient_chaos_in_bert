"""Comparing chaos length between pre-trained
and randomly initialized ALBERT models.
"""

import os
import argparse
import pandas as pd
import numpy as np
import json
import gc
from tqdm import tqdm
from datetime import datetime
from collections import defaultdict
from typing import List, Union

import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    for k in range(len(physical_devices)):
        tf.config.experimental.set_memory_growth(physical_devices[k], True)
        print('memory growth:',
              tf.config.experimental.get_memory_growth(physical_devices[k]))
else:
    print("Not enough GPU hardware devices available")
from src.library.model.albert_system import AlbertSystem
from src.library.utils import wiki_random_extractor


def calculate_lyap_list(net: AlbertSystem,
                        sentences: List[str],
                        save_state: bool = False,
                        save_one_dim: bool = True,
                        save_one_token: bool = False):
    """Make the list of lyapunov exponents values of input sentences with net.
    The lyapunov exponents will be calculated every tau steps. Tau is defined when net is made.
    Parameters
    ----------
    net : AlbertSystem
        The ALBERT network to evaluate.
    sentences : List[str]
        List of input sentences.
    save_state: bool
        Whether or not to save the intermediate state of ALBERT encoder layer.
    save_one_dim : bool, optional
        Whether or not to save only one dimension of intermediate states, by default True
    save_one_token : bool, optional
        Whether or not to save vector of only first word, by default False
    Returns
    -------
    numpy.ndarray (batch_size, T//tau)
        The list of lyapunov exponents values at each step.
    """
    batch_inputs = defaultdict(list)
    for sentence in sentences:
        inputs, _ = net.tokenizer.tokenize_text([sentence],
                                                max_seq_len=args.max_seq_len,
                                                return_pieces=False)
        for k, v in inputs.items():
            inputs[k] = np.array(v, dtype=np.int64).reshape(1, -1)
            batch_inputs[k].append(inputs[k].copy())
    for k, v in batch_inputs.items():
        batch_inputs[k] = np.concatenate(batch_inputs[k], axis=0)
    net.set_input(**batch_inputs)
    return net.maximum_lyapunov(args.tau,
                                args.T,
                                epsilon=args.epsilon,
                                save_state=save_state,
                                save_one_dim=save_one_dim,
                                save_one_token=save_one_token,
                                callback_gpu=lambda x: x.cpu())


def calc_chaos_length(
        pre_trained: bool = True,
        num_iter: int = 5,
        max_seq_len: int = 32,
        dataset: List[str] = ["This is a sample script."],
        batch_size: int = 10,
        root_save_dir: Union[str, os.PathLike] = "../result/compare_chaos_length",
        save_state: bool = False,
        save_one_dim: bool = True,
        save_one_token: bool = False):
    """Calculate the chaos length for each network.
    Parameters
    ----------
    pre_trained : bool, optional
        Whether or not to use the pre-trained network, by default True
    num_iter : int, optional
        Number of networks to calculate, by default 5.
    max_seq_len: int, optional
        The maximum number of tokens for input.
    dataset : List[str], optional
        The input texts, by default ["This is a sample script."]
    batch_size : int, optional
        Batch size, by default batch_size
    root_save_dir : Union[str, os.PathLike], optional
        For the path for root save directory, by default "./out/compare_chaos_length"
    save_state : bool, optional
        Whether or not to save the intermediate state, by default False
    save_one_dim : bool, optional
        Whether or not to save only one dimension of intermediate states, by default True
    """
    for n in range(num_iter):
        net = AlbertSystem("albert_large",
                           "2",
                           max_seq_len,
                           pretrained_weights=pre_trained)
        results = []
        texts = []
        states = []
        for idx, text in tqdm(enumerate(dataset)):
            texts.append(text)
            if len(texts) == args.batch_size:
                lyap_list, _, state = calculate_lyap_list(
                    net,
                    texts,
                    save_state=save_state,
                    save_one_dim=save_one_dim,
                    save_one_token=save_one_token)
                texts = []
                results.append(lyap_list.T.copy())  # (batch_size, T//tau)
                if save_state:
                    states.append(state)

        save_dir_name = "pre_trained" if pre_trained else "initial"
        save_dir = os.path.join(root_save_dir, save_dir_name, f"{n:02d}")
        os.makedirs(save_dir, exist_ok=True)
        if save_state:
            states = np.concatenate(
                states, axis=1)  # (T, batch_size, max_seq_len, hidden_size)
            np.savez_compressed(os.path.join(save_dir, "intermediate_states"),
                                states)

        results = np.concatenate(results, axis=0)
        lyap_steps = args.T // args.tau
        col_list = [args.tau * i for i in range(lyap_steps)]
        df = pd.DataFrame(results, columns=col_list, index=dataset)
        df.index.name = "input_text"
        df = df.T
        df.index.name = "steps"
        df.to_csv(os.path.join(save_dir, "lyap_list.csv"))
        np.save(os.path.join(save_dir, "lyap_lists.npy"), results)

        del net
        gc.collect()


parser = argparse.ArgumentParser()

parser.add_argument("--tau", type=int, default=100)
parser.add_argument("--T", type=int, default=500)
parser.add_argument("--N",
                    type=int,
                    default=100,
                    help="Number of sentence samples.")
parser.add_argument("--max_seq_len", type=int, default=128)
parser.add_argument("--num_initial_nets", type=int, default=10)
parser.add_argument("--batch_size", type=int, default=10)
parser.add_argument("--epsilon", type=float, default=1e-2)
parser.add_argument("--test",
                    action='store_true',
                    help='Checking the code or not.')
parser.add_argument("--input_file",
                    default=None,
                    help='The path to the text file for input to the ALBERT.')
parser.add_argument(
    "--save_state",
    action='store_true',
    help='Whether or not to save the intermediate state of ALBERT encoder layer.')
parser.add_argument(
    "--save_one_dim",
    action='store_true',
    help='When save_state, whether or not to save only one dimension on each state.'
)
parser.add_argument(
    "--save_one_token",
    action='store_true',
    help='When save_state, whether or not to save vector of only first word. If --save_one_dim, this option will be ignored.'
)

args = parser.parse_args()

if __name__ == '__main__':
    now = datetime.now().strftime("%Y%m%d%H%M%S")
    data_dir = os.path.join("./out/compare_chaos_length", now)
    pre_trained_dir = os.path.join(data_dir, "pre_trained")
    initial_dir = os.path.join(data_dir, "initial")
    os.makedirs(pre_trained_dir, exist_ok=True)
    os.makedirs(initial_dir, exist_ok=True)
    with open(os.path.join(data_dir, "config.json"), "w") as f:
        string = json.dumps(args.__dict__, indent=4)
        f.write(string)
    dataset = wiki_random_extractor.extract_wikipedia_sentence(
        args.N, L=args.max_seq_len * 4)
    if type(args.input_file) == str and os.path.exists(args.input_file):
        print(f"Found specified input txt file: {args.input_file}")
        with open(args.input_file, "r") as f:
            lines = f.readlines()
        dataset = []
        for line in lines:
            text = ": ".join(line.split(": ")[1:])
            dataset.append(text)

    with open(os.path.join(data_dir, "input.txt"), "w") as f:
        for idx, text in enumerate(dataset):
            text = text.replace("\n", "")
            f.write(f"{idx:03d}: {text}\n")
    calc_chaos_length(pre_trained=True,
                      num_iter=1,
                      max_seq_len=args.max_seq_len,
                      dataset=dataset,
                      batch_size=args.batch_size,
                      root_save_dir=data_dir,
                      save_state=args.save_state,
                      save_one_dim=args.save_one_dim,
                      save_one_token=args.save_one_token)
    calc_chaos_length(pre_trained=False,
                      num_iter=args.num_initial_nets,
                      max_seq_len=args.max_seq_len,
                      dataset=dataset,
                      batch_size=args.batch_size,
                      root_save_dir=data_dir,
                      save_state=args.save_state,
                      save_one_dim=args.save_one_dim,
                      save_one_token=args.save_one_token)
