from enum import Enum
import tqdm
from dataset import BrainDataset, collate_fn, preprocess_data, BigBrainDataset, \
    UltraDuperBigBrainDataset, UltraDuperBigBrainBatchSampler
from transformer import GPT2likeModel
import typing as tp
import numpy as np
from time import perf_counter
from typing import List


import torchtext
from torchtext.vocab import build_vocab_from_iterator, Vocab
import torch
from torch.utils.data import DataLoader
from torch import nn


class DataMode(Enum):
    BRAIN = 1
    BIG_BRAIN = 2
    ULTRA_DUPER_BIG_BRAIN = 3


def get_gpt2_model(n_tokens: int, hidden_size: int=1024, heads: int=8) -> torch.nn.Module:
    model = GPT2likeModel(n_tokens, hidden_size, heads)
    return model


def run_epoch(data: List[List[int]], vocab: Vocab, data_mode: DataMode,
              batch_size=8, n_bins: int | None=None) -> None:
    if data_mode == DataMode.BRAIN:
        dataset = BrainDataset(data)
        lambda_collate = lambda x: collate_fn(x)
    elif data_mode == DataMode.BIG_BRAIN:
        dataset = BigBrainDataset(data)
        lambda_collate = lambda x: collate_fn(x, None)
    
    if data_mode == DataMode.ULTRA_DUPER_BIG_BRAIN:
        dataset = UltraDuperBigBrainDataset(data, n_bins=n_bins)
        sampler = UltraDuperBigBrainBatchSampler(dataset, batch_size=batch_size)
        lambda_collate = lambda x: collate_fn(x, None)
        dataloader = DataLoader(dataset, batch_sampler=sampler, num_workers=0, 
                                collate_fn=lambda_collate, pin_memory=True)
    else:
        dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=lambda_collate,
                                num_workers=8, shuffle=True, pin_memory=True)

    model = get_gpt2_model(len(vocab))
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    device = torch.device('cuda')
    model.to(device)

    model.train()
    print("Warmup:")
    iters = 0
    for data in tqdm.tqdm(dataloader):
        start = perf_counter()
        data = data.to(device)
        output = model(data)
        loss = criterion(output.transpose(1, 2), data) 
        iters += data.shape[0]
        if iters > 1000:
            break
    
    batch_times = []
    print("Train:")
    for data in tqdm.tqdm(dataloader):
        start = perf_counter()
        data = data.to(device)
        output = model(data)
          
        loss = criterion(output.transpose(1, 2), data) 

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        torch.cuda.synchronize()
        batch_times.append((perf_counter() - start) * batch_size / data.shape[0])
    
    
    batch_processing_times = {
        "minimum": min(batch_times),
        "maximum": max(batch_times),
        "mean": np.mean(batch_times).item(),
        "median": np.median(batch_times).item(),
    }
    print(batch_processing_times)
    return batch_processing_times


if __name__ == "__main__":
    data_path = "wikitext-103-raw/wiki.train.raw"
    # data_path = "/ssd/research/efficient-dl-systems/week03_fast_pipelines/homework/task2/" + data_path

    data, vocab = preprocess_data(data_path)

    run_epoch(data, vocab, DataMode.BRAIN)
    run_epoch(data, vocab, DataMode.BIG_BRAIN)
    run_epoch(data, vocab, DataMode.ULTRA_DUPER_BIG_BRAIN, n_bins=1)
    run_epoch(data, vocab, DataMode.ULTRA_DUPER_BIG_BRAIN, n_bins=5)
    run_epoch(data, vocab, DataMode.ULTRA_DUPER_BIG_BRAIN, n_bins=10)
    run_epoch(data, vocab, DataMode.ULTRA_DUPER_BIG_BRAIN, n_bins=20)
    run_epoch(data, vocab, DataMode.ULTRA_DUPER_BIG_BRAIN, n_bins=50)
