import os
import math
import numpy as np
import pandas as pd
from typing import List, Union
from collections import defaultdict
from sklearn.model_selection import train_test_split

from pyutils.tqdm import tqdm

from src.library.model.albert_tokenization import Tokenizer
from src.library.utils.wiki_random_extractor import extract_wikipedia_sentence


class DataLoader(object):
    def __init__(self,
                 df: pd.DataFrame,
                 tokenizer: Tokenizer,
                 batch_size: int = 32,
                 max_seq_len: int = 32,
                 shuffle: bool = False):
        """
        Args:
            df (pd.DataFrame): [description]
            tokenizer (Tokenizer): [description]
            batch_size (int, optional): [description]. Defaults to 32.
            max_seq_len (int, optional): [description]. Defaults to 32.
            shuffle (bool, optional): [description]. Defaults to False.
        """
        self.df = df
        self.size = int(df.shape[0])
        self.batch_size = batch_size
        self.tokenizer = tokenizer
        self.shuffle = shuffle
        self.max_seq_len = max_seq_len
        self._i = 0
        self.ids = self.df.index.to_numpy()
        self.input_keys = []
        self.label_key = []

    def __iter__(self):
        return self

    def __next__(self):
        if self._i == len(self):
            self.ids = self.df.index.to_numpy()
            if self.shuffle:
                self.ids = np.random.permutation(self.ids)
            self._i = 0
            raise StopIteration
        _start = self._i * self.batch_size
        _end = (self._i + 1) * self.batch_size
        self._i += 1
        return self.make_batch(self.ids[_start:_end])

    def __len__(self):
        return math.ceil(self.size / self.batch_size)

    def make_batch(self, ids: List[int]):
        """Make batch for AlbertSystem

        Parameters
        ----------
        ids : List[int]
            [description]

        Returns
        -------
        dict
            [description]
        """
        batch_inputs = defaultdict(list)
        for _id in ids:
            inputs, _ = self.tokenizer.tokenize_text(
                [self.df.loc[_id, _key] for _key in self.input_keys],
                max_seq_len=self.max_seq_len,
                return_pieces=False)
            for k, v in inputs.items():
                v_new = np.array(v, dtype=np.int64).reshape(1, -1)
                batch_inputs[k].append(v_new[:, :self.max_seq_len])
        for k, v in batch_inputs.items():
            batch_inputs[k] = np.concatenate(batch_inputs[k], axis=0)
        labels = self.df.loc[ids, self.label_key].to_numpy()
        return ids, batch_inputs, labels


class STSDataLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        super(STSDataLoader, self).__init__(*args, **kwargs)
        self.input_keys = ["sentence1", "sentence2"]
        self.label_key = "score"


class QQPDataLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        super(QQPDataLoader, self).__init__(*args, **kwargs)
        self.input_keys = ["question1", "question2"]
        self.label_key = "is_duplicate"


class WikipediaDataLoader(object):
    def __init__(self,
                 tokenizer: Tokenizer,
                 sample_size: int = 1000,
                 batch_size: int = 32,
                 max_seq_len: int = 32,
                 shuffle: bool = False,
                 seed: int = None,
                 data_dir: str = "./data",
                 sentence_length: int = -1):
        """
        Args:
            tokenizer (Tokenizer): [description]
            sample_size (int, optional): The number of samples to use.
            batch_size (int, optional): [description]. Defaults to 32.
            max_seq_len (int, optional): [description]. Defaults to 32.
            shuffle (bool, optional): [description]. Defaults to False.
            data_dir (str, optional): The directory path to wikipedia data. Defaults to "./data".
        """
        self.size = sample_size
        self.batch_size = batch_size
        self.tokenizer = tokenizer
        self.shuffle = shuffle
        self.seed = seed
        self.max_seq_len = max_seq_len
        self._i = 0
        self.ids = np.arange(self.size)
        self.sentences = extract_wikipedia_sentence(
            N=self.size,
            data_dir=data_dir,
            L=self.max_seq_len * 4,
            sentence_length=sentence_length)

    def __iter__(self):
        return self

    def __next__(self):
        if self._i == 0:
            self.reset()
        if self._i == len(self):
            self._i = 0
            raise StopIteration
        _start = self._i * self.batch_size
        _end = (self._i + 1) * self.batch_size
        self._i += 1
        return self.make_batch(self.ids[_start:_end])

    def __len__(self):
        return math.ceil(self.size / self.batch_size)

    def reset(self):
        self.ids = np.arange(self.size)
        if self.shuffle:
            self.ids = np.random.RandomState(seed=self.seed).permutation(self.ids)

    def make_batch(self, ids: List[int]):
        """Make batch for AlbertSystem

        Args:
            ids (List[int]): list of ids

        Returns:
            dict: [description]
        """
        batch_inputs = defaultdict(list)
        for _id in ids:
            if type(self.sentences[_id]) is str:
                texts = [self.sentences[_id]]
            else:
                texts = self.sentences[_id]
            # input(self.sentences[_id])
            inputs, _ = self.tokenizer.tokenize_text(
                texts, max_seq_len=self.max_seq_len,
                return_pieces=False)
            for k, v in inputs.items():
                v_new = np.array(v, dtype=np.int64).reshape(1, -1)
                batch_inputs[k].append(v_new[:, :self.max_seq_len])
                # inputs[k] = np.array(v, dtype=np.int64).reshape(1, -1)
                # batch_inputs[k].append(inputs[k].copy())
        for k, v in batch_inputs.items():
            batch_inputs[k] = np.concatenate(batch_inputs[k], axis=0)
        return ids, batch_inputs


def load_sts_data(tokenizer, batch_size, max_seq_len, **kwargs):
    """load dataloader for STS-B dataset."""
    sts_cols = [
        "index", "genre", "filename", "year", "old_index", "source1",
        "source2", "sentence1", "sentence2", "score"
    ]
    train_path = "../data/glue_data/STS-B/train.tsv"
    # test_path = "../data/glue_data/STS-B/dev.tsv"
    train_df = pd.read_csv(train_path, delimiter='\t', names=sts_cols)
    train_df["score"] = pd.to_numeric(train_df["score"], errors="coerce")
    train_df = train_df[train_df["score"].notna()]
    train_df = train_df[(train_df["score"] == 5.0) | (train_df["score"] == 0.0)]
    train_df, dev_df = train_test_split(train_df, **kwargs)
    train_loader = STSDataLoader(train_df,
                                 tokenizer,
                                 batch_size=batch_size,
                                 max_seq_len=max_seq_len,
                                 shuffle=True)
    dev_loader = STSDataLoader(dev_df,
                               tokenizer,
                               batch_size=batch_size,
                               max_seq_len=max_seq_len,
                               shuffle=False)
    return train_loader, dev_loader


def load_qqp_data(tokenizer, batch_size, max_seq_len, **kwargs):
    """load dataloader for QQP dataset."""
    sts_cols = ["id", "qid1", "qid2", "question1", "question2", "is_duplicate"]
    # loading dataset.
    train_path = "../data/glue_data/QQP/train.tsv"
    # test_path = "../data/glue_data/QQP/dev.tsv"
    train_df = pd.read_csv(train_path, delimiter="\t", names=sts_cols)
    train_df["is_duplicate"] = pd.to_numeric(train_df["is_duplicate"],
                                             errors="coerce")
    train_df = train_df[train_df["is_duplicate"].notna()]
    train_df["is_duplicate"] = train_df["is_duplicate"].astype(int)
    train_df, dev_df = train_test_split(train_df, **kwargs)
    train_loader = QQPDataLoader(train_df,
                                 tokenizer,
                                 batch_size=batch_size,
                                 max_seq_len=max_seq_len,
                                 shuffle=True)
    dev_loader = QQPDataLoader(dev_df,
                               tokenizer,
                               batch_size=batch_size,
                               max_seq_len=max_seq_len,
                               shuffle=False)
    return train_loader, dev_loader


def load_wiki_data(tokenizer,
                   batch_size,
                   max_seq_len,
                   train_size=9500,
                   eval_size=500,
                   sentence_length=-1,
                   **kwargs):
    train_loader = WikipediaDataLoader(tokenizer,
                                       train_size,
                                       batch_size,
                                       max_seq_len,
                                       shuffle=True,
                                       sentence_length=sentence_length)
    eval_loader = WikipediaDataLoader(
        tokenizer,
        eval_size,
        batch_size,
        max_seq_len,
        shuffle=False,
        sentence_length=sentence_length,
    )
    return train_loader, eval_loader


def get_dataloader(task: str, tokenizer: Tokenizer, batch_size: int,
                   max_seq_len: int, **kwargs):
    if task == "STS-B":
        return load_sts_data(tokenizer, batch_size, max_seq_len, **kwargs)
    elif task == "QQP":
        return load_qqp_data(tokenizer, batch_size, max_seq_len, **kwargs)
    elif task == "Wikipedia":
        return load_wiki_data(tokenizer, batch_size, max_seq_len, **kwargs)
    else:
        raise NotImplementedError
