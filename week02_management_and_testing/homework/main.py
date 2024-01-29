import torch
import os
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10

from modeling.diffusion import DiffusionModel
from modeling.training import generate_samples, train_epoch
from modeling.unet import UnetModel


def main(device: str, num_epochs: int = 100):
    ddpm = DiffusionModel(
        eps_model=UnetModel(3, 3, hidden_size=128),
        betas=(1e-4, 0.02),
        num_timesteps=1000,
        device=device
    )
    ddpm.to(device)

    train_transforms = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    dataset = CIFAR10(
        "cifar10",
        train=True,
        download=True,
        transform=train_transforms,
    )

    # subdata = torch.utils.data.Subset(dataset, range(8))
    dataloader = DataLoader(dataset, batch_size=128, num_workers=8, shuffle=True)
    optim = torch.optim.Adam(ddpm.parameters(), lr=1e-4)

    os.makedirs("samples", exist_ok=True)
    for i in range(num_epochs):
        train_epoch(ddpm, dataloader, optim, device)
        generate_samples(ddpm, device, f"samples/{i:02d}.png")

    torch.save(ddpm.state_dict(), "ddpm.pt")


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    main(device=device)
