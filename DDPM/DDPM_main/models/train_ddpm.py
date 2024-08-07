import os
import sys
from pathlib import Path

top_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(top_dir))

import torch
import torch.nn as nn
import torch.nn.functional as F
from config.config import Config
from data.mnist_data import MNISTData
from diffuser_unet import DFUNet
from inference import inference
from scheduler.ddpm_scheduler import DDPMScheduler
from torch.utils.data import DataLoader
# 用于显示进度条，不需要可以去掉
from tqdm.auto import tqdm
from visualize.plot import *

config = Config(r"/sharedata/usr/huangsuizhi/DL_Demo/DDPM/DDPM_main/config/config.yaml")
model = DFUNet(config).to(config.device)
scheduler = DDPMScheduler(config)

# diffusers 里面用的是 AdamW,
# lr 不能设置的太大
optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

training_data = MNISTData(config, r"/sharedata/datasets/MNIST", return_label=True)
train_dataloader = DataLoader(training_data, batch_size=config.batch, shuffle=True)

# 训练模型
for ep in range(config.epochs):
    progress_bar = tqdm(total=len(train_dataloader))
    model.train()
    for image, _ in train_dataloader:
        batch = image.shape[0]
        timesteps = scheduler.sample_timesteps(batch)
        noise = torch.randn(image.shape).to(config.device)
        noisy_image = scheduler.add_noise(image=image, noise=noise, timesteps=timesteps)

        # pred = model(noisy_image, timesteps)[0]
        pred = model(noisy_image, timesteps)
        loss = F.mse_loss(pred, noise)
        optimizer.zero_grad()
        loss.backward()

        # gradient clipping, 用来防止 exploding gradients
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        progress_bar.update(1)
        logs = {"loss": loss.detach().item(), "ep": ep + 1}
        progress_bar.set_postfix(**logs)

    # 保存模型
    if (ep + 1) % config.save_period == 0 or (ep + 1) == config.epochs:
        torch.save(
            {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
            },
            r"/sharedata/usr/huangsuizhi/DL_Demo/DDPM/DDPM_main/models/checkpoints/model_ep"
            + str(ep + 1),
        )

    # 采样一些图片
    if (ep + 1) % config.sample_period == 0:
        model.eval()
        labels = (
            torch.randint(0, 9, (config.num_inference_images, 1))
            .to(config.device)
            .float()
        )
        image = inference(model, scheduler, config.num_inference_images, config)
        image = (image / 2 + 0.5).clamp(0, 1)
        plot_images(image, save_dir=config.proj_name, titles=labels.detach().tolist())
        model.train()
