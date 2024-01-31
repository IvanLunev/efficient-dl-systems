import torch
import os
import wandb
import math
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torchvision.utils import make_grid

from modeling.diffusion import DiffusionModel
from modeling.training import generate_samples, train_epoch
from modeling.unet import UnetModel

from omegaconf import DictConfig, OmegaConf
import hydra


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(config: DictConfig):

    wandb.init(
        project=config.wandb.project, name=config.wandb.experiment,
        config = OmegaConf.to_container(config, resolve=True, throw_on_missing=True)
    )

    if config.solver.device == "cuda":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = "cpu"
    print(f"Selected device: {device}")


    ddpm = DiffusionModel(
        eps_model=UnetModel(3, 3, hidden_size=config.model.hidden_size),
        betas=config.model.betas,
        num_timesteps=config.model.num_timesteps,
        device=device
    )
    ddpm.to(device)
    wandb.watch(ddpm)

    transforms_list = [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    if config.data.flip_aug:
        transforms_list.append(transforms.RandomHorizontalFlip())

    train_transforms = transforms.Compose(transforms_list)

    dataset = CIFAR10(
        "cifar10",
        train=True,
        download=True,
        transform=train_transforms,
    )

    dataloader = DataLoader(dataset, batch_size=config.solver.batch_size, 
                            num_workers=config.solver.num_workers, shuffle=True)
    if config.solver.optimazer == "adam":
        optim = torch.optim.Adam(ddpm.parameters(), lr=config.solver.lr)
    elif config.solver.optimazer == "sgd":
        optim = torch.optim.SGD(ddpm.parameters(), lr=config.solver.lr, momentum=config.solver.momentum)
    else:
        print("Wrong option for optimazer!")
        return


    for epoch in range(config.solver.num_epochs):
        train_loss = train_epoch(ddpm, dataloader, optim, device)
        current_step = (epoch + 1) * len(dataset)

        metrics = {'train_loss': train_loss}
        wandb.log(metrics, step=current_step)

        # make and log images
        images_count = 16
        images, _ = next(iter(dataloader))
        images = make_grid(images[:images_count], nrow=math.ceil(images_count ** (1/2)))
        gen_images = generate_samples(ddpm, device, count=images_count)
        assert images.shape == gen_images.shape

        images = images.permute(1, 2, 0).cpu().numpy()
        gen_images = gen_images.permute(1, 2, 0).cpu().numpy()

        images = wandb.Image(images, caption=f"Input images")
        gen_images = wandb.Image(gen_images, caption=f"Generated images")

        wandb.log({"images": [images, gen_images]}, step=current_step)


    torch.save(ddpm.state_dict(), "ddpm.pt")
    wandb.finish()


if __name__ == "__main__":
    main()
