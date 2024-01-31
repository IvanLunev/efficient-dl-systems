import pytest
import os
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Normalize
from torchvision.datasets import CIFAR10
from torchvision.utils import make_grid, save_image

from modeling.diffusion import DiffusionModel
from modeling.training import train_step, train_epoch, generate_samples
from modeling.unet import UnetModel


@pytest.fixture
def train_dataset():
    transforms = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    dataset = CIFAR10(
        "./data",
        train=True,
        download=True,
        transform=transforms,
    )
    return dataset


@pytest.mark.parametrize(["device"], [["cpu"], ["cuda"]])
def test_train_on_one_batch(device, train_dataset):
    # note: you should not need to increase the threshold or change the hyperparameters
    ddpm = DiffusionModel(
        eps_model=UnetModel(3, 3, hidden_size=32),
        betas=(1e-4, 0.02),
        num_timesteps=1000,
        device=device
    )
    ddpm.to(device)

    optim = torch.optim.Adam(ddpm.parameters(), lr=5e-4)
    dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)

    x, _ = next(iter(dataloader))
    assert x.shape == (4, 3, 32, 32)

    loss = None
    for i in range(50):
        loss = train_step(ddpm, x, optim, device)
    assert loss < 0.5


@pytest.mark.parametrize(["device"], [["cpu"], ["cuda"]])
def test_training(device, train_dataset):
    # note: implement and test a complete training procedure (including sampling)
    ddpm = DiffusionModel(
        eps_model=UnetModel(3, 3, hidden_size=128),
        betas=(1e-4, 0.02),
        num_timesteps=1000,
        device=device
    )
    ddpm.to(device)
    
    optim = torch.optim.Adam(ddpm.parameters(), lr=5e-4)
    subdata = torch.utils.data.Subset(train_dataset, range(8))
    dataloader = DataLoader(subdata, batch_size=8, shuffle=True)

    for epoch in range(100):
        train_epoch(ddpm, dataloader, optim, device)

    os.makedirs("tests/test_out", exist_ok=True)

    x, _ = next(iter(dataloader))
    grid = make_grid(x, nrow=4)
    save_image(grid, f"tests/test_out/{device}_in.png")
    generate_samples(ddpm, device, path=f"tests/test_out/{device}_out.png")
