# See: https://github.com/deepmind/sonnet/blob/v2/examples/vqvae_example.ipynb.

import cv2
import numpy as np
import torch
import kagglehub
import dlib
import os
import shutil

from pathlib import Path
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10
from vqvae import VQVAE
from dataset import CelebaDataset
from utils import *

torch.set_printoptions(linewidth=160)

# Initialize model.
device = torch.device("cuda:0")
use_ema = True
model_args = {
    "in_channels": 3,
    "num_hiddens": 128,
    "num_downsampling_layers": 2,
    "num_residual_layers": 2,
    "num_residual_hiddens": 32,
    "embedding_dim": 64,
    "num_embeddings": 512,
    "use_ema": use_ema,
    "decay": 0.99,
    "epsilon": 1e-5,
}
model = VQVAE(**model_args).to(device)

# Initialize dataset.
batch_size = 32
workers = 10
normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[1.0, 1.0, 1.0])
transform = transforms.Compose(
    [
        transforms.ToTensor(),
        normalize,
    ]
)
data_root = Path("../data")
CelebA_HQ_path = os.path.join(data_root, "CelebA-HQ-img")
if not os.path.exists(CelebA_HQ_path) or len(os.listdir(CelebA_HQ_path)) == 0:
    print("Downloading CelebA-HQ data...")
    celeba_path = kagglehub.dataset_download("ipythonx/celebamaskhq")
    shutil.move(os.path.join(celeba_path, "CelebAMask-HQ/CelebA-HQ-img"), data_root)
    shutil.rmtree(celeba_path)
    print("Data sorted.")
Cropped_CelebA_HQ_path = os.path.join(data_root, "Cropped_CelebA-HQ-img")
if not os.path.exists(Cropped_CelebA_HQ_path) or len(os.listdir(Cropped_CelebA_HQ_path)) == 0:
    print("Cropping CelebA-HQ data...")
    detector = dlib.get_frontal_face_detector()
    for im_name in tqdm(os.listdir(CelebA_HQ_path)):
        im_path = os.path.join(CelebA_HQ_path, im_name)
        im = cv2.imread(im_path)
        cropped, _, _, _, _, h = preprocess_face(im, detector)
        if h == None:
            continue
        writting_path = os.path.join(Cropped_CelebA_HQ_path, im_name)
        cv2.imwrite(writting_path, cropped)
    print("All images are cropped.")
    
# train_dataset = CIFAR10(data_root, True, transform, download=True)
merging = False
cropped = True
train_dataset = CelebaDataset(data_root, merging=merging, cropped=cropped)
train_data_variance = train_dataset.calculate_mean_variance()[1]
train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=workers,
)

# Multiplier for commitment loss. See Equation (3) in "Neural Discrete Representation
# Learning".
beta = 0.25

# Initialize optimizer.
train_params = [params for params in model.parameters()]
lr = 3e-4
optimizer = optim.Adam(train_params, lr=lr)
criterion = nn.MSELoss()

# Train model.
epochs = 2
eval_every = 100
best_train_loss = float("inf")
model.train()
for epoch in range(epochs):
    total_train_loss = 0
    total_recon_error = 0
    n_train = 0
    for (batch_idx, train_tensors) in enumerate(train_loader):
        optimizer.zero_grad()
        imgs, targets = train_tensors
        imgs = imgs.to(device)
        targets = targets.to(device)
        out = model(imgs)
        recon_error = criterion(out["x_recon"], targets) / train_data_variance
        total_recon_error += recon_error.item()
        loss = recon_error + beta * out["commitment_loss"]
        if not use_ema:
            loss += out["dictionary_loss"]

        total_train_loss += loss.item()
        loss.backward()
        optimizer.step()
        n_train += 1

        if ((batch_idx + 1) % eval_every) == 0:
            print(f"epoch: {epoch}\nbatch_idx: {batch_idx + 1}", flush=True)
            total_train_loss /= n_train
            if total_train_loss < best_train_loss:
                best_train_loss = total_train_loss
                if merging:
                    path_to_save = os.path.join("..", "checkpoints", "merging")
                else:
                    path_to_save = os.path.join("..", "checkpoints", "vqvae")
                save_checkpoint(path_to_save, model, optimizer, epoch, loss)

            print(f"total_train_loss: {total_train_loss}")
            print(f"best_train_loss: {best_train_loss}")
            print(f"recon_error: {total_recon_error / n_train}\n")

            total_train_loss = 0
            total_recon_error = 0
            n_train = 0

# Generate and save reconstructions.
model.eval()

valid_dataset = CelebaDataset(data_root, merging=merging, cropped=cropped)
valid_loader = DataLoader(
    dataset=valid_dataset,
    batch_size=batch_size,
    num_workers=workers,
)

with torch.no_grad():
    for valid_tensors in valid_loader:
        imgs, targets = valid_tensors
        break
    save_img_tensors_as_grid(imgs, 4, "true")
    save_img_tensors_as_grid(targets, 4, "targets")
    save_img_tensors_as_grid(model(imgs.to(device))["x_recon"], 4, "recon")
