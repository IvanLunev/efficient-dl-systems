import torch
from torch import nn
from tqdm.auto import tqdm

from unet import Unet

from dataset import get_train_data


class StaticScaler():
    def __init__(self):
        self._scale: float = 2.0**16
        self.inv_scale = 1.0 / self._scale
    
    def scale(self, loss: torch.nn.Module) -> torch.nn.Module:
        return loss * self._scale

    def unscale_(self, optimizer: torch.optim.Optimizer):
        for params in optimizer.param_groups[0]['params']:
            params.grad *= self.inv_scale

    def step(self, optimizer: torch.optim.Optimizer):
        optimizer.step()

    def update(self):
        pass


class DynamicScaler(torch.cuda.amp.GradScaler):
    def __init__(self):
        self._scale: float = 2.0**8
        self.grad_check = True
        self.good_iter_num = 0
        self.good_iter_min = 10
    
    def scale(self, loss: torch.nn.Module) -> torch.nn.Module:
        return loss * self._scale

    def unscale_(self, optimizer: torch.optim.Optimizer):
        for params in optimizer.param_groups[0]['params']:
            is_nan_num = torch.isnan(params.grad).sum()
            is_inf_num = torch.isinf(params.grad).sum()
            grad_check = is_nan_num + is_inf_num == 0
            self.grad_check = grad_check.cpu().numpy()
            if not self.grad_check:
                return
        for params in optimizer.param_groups[0]['params']:
            inv_scale = 1.0 / self._scale
            params.grad *= inv_scale

    def step(self, optimizer: torch.optim.Optimizer):
        if self.grad_check:
            optimizer.step()

    def update(self):
        if not self.grad_check:
            self.good_iter_num = 0
            self._scale *= 0.5
        elif self.good_iter_num < self.good_iter_min:
            self.good_iter_num += 1
        else:
            self._scale *= 2.0


def train_epoch(
    train_loader: torch.utils.data.DataLoader,
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler
) -> None:
    model.train()

    pbar = tqdm(enumerate(train_loader), total=len(train_loader))
    for i, (images, labels) in pbar:
        images = images.to(device)
        labels = labels.to(device)

        with torch.cuda.amp.autocast():
            outputs = model(images)
            loss = criterion(outputs, labels)
        # TODO: your code for loss scaling here
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        scaler.step(optimizer)
        scaler.update()

        optimizer.zero_grad()

        accuracy = ((outputs > 0.5) == labels).float().mean()

        pbar.set_description(f"Loss: {round(loss.item(), 4)} " f"Accuracy: {round(accuracy.item() * 100, 4)}")


def train(scaler=torch.cuda.amp.GradScaler(enabled=True)):
    device = torch.device("cuda:0")
    model = Unet().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    train_loader = get_train_data()

    num_epochs = 5
    for epoch in range(0, num_epochs):
        train_epoch(train_loader, model, criterion, optimizer, device=device, scaler=scaler)

if __name__ == "__main__":
    train()