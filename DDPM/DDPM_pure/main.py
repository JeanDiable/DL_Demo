import os
import time

import cv2
import einops
import numpy as np
import torch
import torch.nn as nn
from dataset import get_dataloader, get_img_shape
from ddpm import DDPM
from model import build_network


def train(
    ddpm: DDPM, net, device='cuda', ckpt_path='./ckpt.pth', batch_size=512, n_epochs=100
):
    """training the whole DDPM algo

    Args:
        ddpm (DDPM): the DDPM class that applies the DDPM algo
        net (nn.Module): the network using to optimize the noise
        device (cuda device or CPU):
        ckpt_path (path): where to store the ckpt_path
    """
    n_steps = ddpm.n_steps
    dataloader = get_dataloader(batch_size)
    net = net.to(device)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)

    tic = time.time()
    for epoch in range(n_epochs):
        total_loss = 0

        for x, _ in dataloader:
            batch_size = x.shape[0]
            t = torch.randint(0, n_steps, (batch_size,)).to(device)
            x = x.to(device)

            eps = torch.randn_like(x).to(device)
            x_t = ddpm.sample_forward(x, t, eps)
            eps_theta = net(x_t, t.reshape(-1, 1))
            loss = loss_fn(eps_theta, eps)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch_size
        total_loss /= len(dataloader.dataset)
        toc = time.time()
        torch.save(net.state_dict(), ckpt_path)
        print(f'epoch {epoch} loss: {total_loss} elapsed {(toc - tic):.2f}s')
    print('Training Done')


def sample_imgs(
    ddpm,
    net,
    output_path='./diffusion.jpg',
    n_sample=81,
    device='cuda',
    simple_var=False,
):
    net = net.to(device)
    net = net.eval()
    with torch.no_grad():
        shape = (n_sample, *get_img_shape())  # 1, 3, 28, 28
        imgs = (
            ddpm.sample_backward(shape, net, device=device, simple_var=simple_var)
            .detach()
            .cpu()
        )
        imgs = (imgs + 1) / 2 * 255
        imgs = imgs.clamp(0, 255)
        imgs = einops.rearrange(
            imgs, '(b1 b2) c h w -> (b1 h) (b2 w) c', b1=int(n_sample**0.5)
        )

        imgs = imgs.numpy().astype(np.uint8)
        cv2.imwrite(output_path, imgs)


if __name__ == "__main__":

    batch_size = 512
    n_epochs = 40
    n_steps = 1000
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    unet_res_cfg = {
        'channels': [10, 20, 40, 80],
        'pe_dim': 128,
        'residual': True,
    }
    net = build_network(unet_res_cfg, n_steps)
    ddpm = DDPM(device, n_steps)

    train(ddpm, net, device, batch_size=batch_size, n_epochs=n_epochs)

    net.load_state_dict(torch.load('./ckpt.pth'))
    sample_imgs(ddpm, net)
