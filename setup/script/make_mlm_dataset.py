"""Preparing dataset for MLM tasks.
"""
import argparse
import os

from src.library.utils import wiki_random_extractor


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_seq_len", type=int, default=32)
    parser.add_argument("--N", type=int, default=10000)
    parser.add_argument("--test_ratio", type=float, default=0.05)
    return parser


if __name__ == "__main__":
    args = get_args().parse_args()
    save_dir = "../data/wiki"
    train_fname = "corpus_for_mlm_train.txt"
    test_fname = "corpus_for_mlm_test.txt"
    dataset = wiki_random_extractor.extract_wikipedia_sentence(
        args.N, L=args.max_seq_len * 4)
    num_test = int(args.N * args.test_ratio)
    num_train = int(args.N - num_test)
    train_path = os.path.join(save_dir, train_fname)
    test_path = os.path.join(save_dir, test_fname)
    with open(train_path, "w") as f:
        dataset_train = dataset[:num_train]
        for line in dataset_train:
            f.write(line + "\n")
    with open(test_path, "w") as f:
        dataset_test = dataset[num_train:]
        for line in dataset_test:
            f.write(line + "\n")
    print(f"Dataset has been created at {train_path} and {test_path}.")
