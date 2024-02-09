from typing import Optional, Callable, List, Tuple
import numpy as np
import re
from collections import defaultdict

import torch
from torch import nn
from torch.utils.data.dataset import Dataset
from torch.utils.data import Sampler
from torchtext.vocab import build_vocab_from_iterator, Vocab
import torchtext
import torch.nn.functional as F
import os


MAX_LENGTH = 640


def preprocess_data(data_path: str, max_tokens: int=8192, max_rows: int=8192) -> Tuple[List[List[int]], Vocab]:
    """
    Read, filter and tokenize data
    """
    with open(os.path.join(data_path)) as fin:
        raw_eval_data = fin.read()
        lines = raw_eval_data.split(" \n \n ")
    if len(lines) > max_rows:
        inds = np.random.choice(len(lines), max_rows, replace=False)
        lines = np.array(lines)[inds]

    tokenizer = torchtext.data.utils.get_tokenizer("basic_english")
    data = []
    for line in lines:
        forbidden = [" ", "\n", "="]
        if any([line.startswith(symb) for symb in forbidden]):
            continue
        data.append(tokenizer(re.sub(r'[^\w\s]','',line)))

    vocab = build_vocab_from_iterator(iter(data), specials=["<unk>"], max_tokens=max_tokens)
    vocab.set_default_index(vocab["<unk>"])

    for ind, row in enumerate(data):
        data[ind] = vocab(row)

    return data, vocab


class BrainDataset(Dataset):
    def __init__(self, data: List[List[int]], max_length: int = MAX_LENGTH):
        self.max_length = max_length
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        return self.data[idx][:self.max_length]


class BigBrainDataset(Dataset):
    def __init__(self, data: List[List[int]], max_length: int = MAX_LENGTH):
        self.max_length = max_length
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        return self.data[idx][:self.max_length]


class UltraDuperBigBrainDataset(Dataset):
    def __init__(self, data: List[List[int]], max_length: int = MAX_LENGTH, n_bins: int = 1):
        self.max_length = max_length
        self.data = data
        self.n_bins = n_bins

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        return self.data[idx][:self.max_length]


def collate_fn(
    batch: list[tuple[str, torch.Tensor]], 
    max_length: Optional[int] = MAX_LENGTH,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Pad each sequence of the incoming sequences list
    :param batch: a list of the objects received from the dataset by __getitem__
    :param max_length: maximum sequence length to pad to (for "Brain" approach only)
    :return: tuple of padded sequences and corresponding training targets
    """
    text_list = []
    current_max_len = 0
    for _text in batch:
        processed_text = torch.tensor(_text, dtype=torch.int64)
        text_list.append(processed_text)
        current_max_len = max(current_max_len, len(text_list[-1]))
    
    if max_length is None:
        max_length = current_max_len

    for ind, _text in enumerate(text_list):
        text_list[ind] = F.pad(input=_text, pad=(0, max_length - len(_text)), value=0)

    text_list = torch.vstack(text_list)
    return text_list


class UltraDuperBigBrainBatchSampler(Sampler):

    def __init__(self, dataset: UltraDuperBigBrainDataset, batch_size: int, 
                 max_length: Optional[int] = MAX_LENGTH):
        self.dataset = dataset
        self.batch_size = batch_size
        self.max_length = max_length

        self.len_to_inds = defaultdict(list)
        for ind, data in enumerate(dataset.data):
            self.len_to_inds[len(data)].append(ind)

        self.count_to_len = []
        for key in self.len_to_inds:
            self.count_to_len.append((len(self.len_to_inds), key))
        self.count_to_len = np.array(self.count_to_len)

    def __len__(self):
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        probs = self.count_to_len[:, 0] / sum(self.count_to_len[:, 0])
        inds = np.random.choice(len(self.count_to_len), self.__len__(), p=probs)
        for ind in inds:
            main_len = self.count_to_len[ind, 1]
            left_len = max(main_len - self.dataset.n_bins // 2, 1)
            right_len = left_len + self.dataset.n_bins
            nums = []
            lens = []
            for cur_len in range(left_len, right_len + 1):
                if cur_len in self.len_to_inds:
                    nums.append(len(self.len_to_inds[cur_len]))
                    lens.append(cur_len)
            if sum(nums) <= self.batch_size:
                yield [yind for cur_len in lens for yind in self.len_to_inds[cur_len]]
            else:
                cum_nums = np.cumsum(nums)
                batch_inds = np.random.choice(sum(nums), self.batch_size)
                yield_inds = []
                for ind in batch_inds:
                    len_ind = np.argwhere(ind + 0.5 < cum_nums)[0][0]
                    cur_len = lens[len_ind]
                    inner_ind = ind
                    if len_ind > 0:
                        inner_ind -= cum_nums[len_ind - 1]
                    yield_inds.append(self.len_to_inds[cur_len][inner_ind])
                yield yield_inds
