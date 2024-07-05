import os
import sys
from pathlib import Path

top_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(top_dir))

import torch
from config.config import Config
from diffuser_unet import DFUNet
from inference import inference
from scheduler.ddpm_scheduler import DDPMScheduler
from unet import UNet
from visualize.plot import *

config = Config(r"../config/config.yaml")
model = DFUNet(config).to(config.device)
model.eval()

scheduler = DDPMScheduler(config)

# 读取模型
checkpoint = torch.load(r"checkpoints\model_ep125")
model.load_state_dict(checkpoint['model_state_dict'])

image = inference(model, scheduler, config.num_inference_images, config)
image = (image / 2 + 0.5).clamp(0, 1)
plot_images(image, save_dir="test")
