import pytest
import torch

from modeling.diffusion import DiffusionModel
from modeling.unet import UnetModel, ConvBlock, DownBlock, UpBlock, TimestepEmbedding


@pytest.mark.parametrize(
        "input_tensor",
    [
        torch.randn(2, 64, 32, 32),
        torch.randn(2, 128, 64, 64),
        torch.randn(2, 64, 128, 128),
        torch.randn(2, 128, 256, 256),
    ]
)
def test_conv_block(input_tensor):
    B, C, H, W = input_tensor.shape

    conv_block = ConvBlock(C, C)
    conv_block_out = conv_block(input_tensor)
    assert conv_block_out.shape == (B, C, H, W)

    conv_block_res = ConvBlock(C, C // 2, True)
    conv_block_res_out = conv_block_res(input_tensor)
    assert conv_block_res_out.shape == (B, C // 2, H, W)


@pytest.mark.parametrize(
        "input_tensor",
    [
        torch.randn(2, 64, 32, 32),
        torch.randn(2, 128, 64, 64),
        torch.randn(2, 64, 128, 128),
        torch.randn(2, 128, 256, 256),
    ]
)
def test_down_block(input_tensor):
    B, C, H, W = input_tensor.shape

    down_block = DownBlock(C, C * 2)
    down_block_out = down_block(input_tensor)
    assert down_block_out.shape == (B, C * 2, H // 2, W // 2)


@pytest.mark.parametrize(
        "input_tensor",
    [
        torch.randn(2, 64, 32, 32),
        torch.randn(2, 128, 64, 64),
        torch.randn(2, 64, 128, 128),
        torch.randn(2, 128, 256, 256),
    ]
)
def test_up_block(input_tensor):
    B, C, H, W = input_tensor.shape

    up_block = UpBlock(C * 2, C)
    up_block_out = up_block(input_tensor, input_tensor)
    assert up_block_out.shape == (B, C, H * 2, W * 2)


@pytest.mark.parametrize("num_timesteps", [2, 4, 8])
def test_time_emb(num_timesteps):
    B, C = 64, 128
    time_emb = TimestepEmbedding(C)
    timestep = torch.randint(1, num_timesteps + 1, (B,)) / num_timesteps
    time_emb_out = time_emb(timestep)
    assert time_emb_out.shape == (B, C)


@pytest.mark.parametrize(
    [
        "input_tensor",
        "num_timesteps",
    ],
    [
        (
            torch.randn(2, 3, 32, 32),
            10,
        ),
        (
            torch.randn(2, 3, 64, 64),
            20,
        ),
        (
            torch.randn(2, 3, 128, 128),
            30,
        ),
        (
            torch.randn(2, 3, 256, 256),
            40,
        ),
    ],
)
def test_unet(input_tensor, num_timesteps):
    B, C, H, W = input_tensor.shape
    net = UnetModel(C, C, hidden_size=128)
    timestep = torch.randint(1, num_timesteps + 1, (B,)) / num_timesteps
    out = net(input_tensor, timestep)
    assert out.shape == input_tensor.shape


def test_diffusion(num_channels=3, batch_size=4):
    torch.manual_seed(0)
    # note: you should not need to change the thresholds or the hyperparameters
    net = UnetModel(num_channels, num_channels, hidden_size=128)
    model = DiffusionModel(eps_model=net, betas=(1e-4, 0.02), num_timesteps=20)

    input_data = torch.randn((batch_size, num_channels, 32, 32))

    output = model(input_data)
    assert output.ndim == 0
    assert 1.0 <= output <= 1.2

    output = model.sample(4, (3, 32, 32), 'cpu')
    assert output.shape == (4, 3, 32, 32)
