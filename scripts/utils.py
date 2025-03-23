
import torch
import os
import cv2
import numpy as np
from dataset import CelebaDataset
from tqdm import tqdm
from sklearn.decomposition import PCA
from PIL import Image
from vqvae import VQVAE
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


def save_checkpoint(
    save_path,
    model,
    optimizer,
    epoch,
    loss,
):

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }

    torch.save(checkpoint, os.path.join(save_path, f"best_loss.pkl"))


def save_img_tensors_as_grid(img_tensors, nrows, f):
    img_tensors = img_tensors.permute(0, 2, 3, 1)
    imgs_array = img_tensors.detach().cpu().numpy()
    imgs_array = (imgs_array - imgs_array.min()) * (1/(imgs_array.max() - imgs_array.min()) * 255)
    (batch_size, img_size) = img_tensors.shape[:2]
    ncols = batch_size // nrows
    img_arr = np.zeros((nrows * img_size, ncols * img_size, 3))
    for idx in range(batch_size):
        row_idx = idx // ncols
        col_idx = idx % ncols
        row_start = row_idx * img_size
        row_end = row_start + img_size
        col_start = col_idx * img_size
        col_end = col_start + img_size
        img_arr[row_start:row_end, col_start:col_end] = imgs_array[idx]

    Image.fromarray(img_arr.astype(np.uint8), "RGB").save(f"{f}.jpg")
    return 0

def PCA_analysis(dataloader, weight_path="../checkpoints/vqvae/full_images.pkl", device = torch.device("cuda:0")):
    model_args = {
    "in_channels": 3,
    "num_hiddens": 128,
    "num_downsampling_layers": 2,
    "num_residual_layers": 2,
    "num_residual_hiddens": 32,
    "embedding_dim": 64,
    "num_embeddings": 512,
    "use_ema": True,
    "decay": 0.99,
    "epsilon": 1e-5,
    }
    model = VQVAE(**model_args).to(device)
    model.load_state_dict(torch.load(weight_path, weights_only=True)["model_state_dict"])

    encoder = model.encoder  # Your trained encoder
    pre_vq = model.pre_vq_conv
    vq = model.vq
    decoder = model.decoder  # Your trained decoder

    data_loader = DataLoader(dataset) # Your dataset

    # 1. Collect latent vectors from real images
    latents = []
    images = []
    for i, batch in tqdm(enumerate(data_loader), desc="Generating latent spaces"):
        inputs, targets = batch
        z = encoder(inputs.to(device))  # Get the mean latent vector
        latents.append(z.detach().cpu().numpy())
        images.append(inputs)
        if i == 1000:
            break
        
    original_shape = z.shape
    image_shape = inputs.shape

    latents = np.vstack(latents)  # Shape: (num_samples, latent_dim)
    images = np.vstack(images)
    images = images.reshape(images.shape[0], -1)
    latents = latents.reshape(latents.shape[0], -1)
    n_components = 4
    # 2. Apply PCA
    pca = PCA(n_components=n_components)
    pca_latents = pca.fit_transform(latents)

    pca_i = PCA(n_components=n_components)
    pca_im = pca_i.fit_transform(images)
    # 3. Modify a latent vector along a principal component
    idx = 10 # Choose a sample
    original_z = latents[idx]
    original_im = images[idx]
    # max_value = original_z.max()
    for i in tqdm(range(n_components), desc="Plotting components"):
        component = pca.components_[i][:128*256*256] #[:128*32*32]
        component = component.reshape(128, 256, 256) #(128, 32,32)
        avg_component = np.mean(component, axis=0)
        
        plt.imshow(avg_component, cmap="coolwarm")
        plt.colorbar()
        plt.title(f"PCA component {i} (AVG)")
        plt.savefig(f"../PCA/PCA_AVG_component_{i}.jpg")
        plt.close()

    original_max = original_z.max()
    im_max = original_im.max()
    for i in range(n_components):
        for alpha in [-50, 0, 50]:  # Vary along PC direction
            modified_z =  original_z + alpha * original_max * pca.components_[i]  # Modify along PC_i, original_z + alpha * np.random.random_sample(original_z.shape)
            modified_z = torch.tensor(modified_z).float().unsqueeze(0)

            # 4. Decode and visualize
            pre_vq_modified_z = pre_vq(modified_z.view(original_shape).to(device))
            vq_modified_z = vq(pre_vq_modified_z)[0]
            recon = decoder(vq_modified_z).detach().cpu()
            modified_im = torch.from_numpy((original_im + alpha * im_max * pca_i.components_[i]).reshape(image_shape))
            title = f"../results/PC_{i}_alpha_{alpha}"
            save_img_tensors_as_grid(recon, 1, title)
            title_im = f"../results/IM_PC_{i}_alpha_{alpha}"
            save_img_tensors_as_grid(modified_im, 1, title_im)

def preprocess_face(image, detector):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    if len(faces) > 0:
        x, y, w, h = faces[0].left(), faces[0].top(), faces[0].width(), faces[0].height()
        face = image[y:y+h, x:x+w]
        face = cv2.resize(face, (128, 128)) # / 255.0
        return face, image.shape, x, y, w, h
    return None, None, None, None, None, None

if __name__=='__main__':
    data_root = "../data"
    dataset = CelebaDataset(data_root, cropped=False)
    loader = DataLoader(
        dataset=dataset,
        batch_size=32,
        shuffle=False,
        num_workers=10,
    )
    PCA_analysis(loader) #, weight_path="../checkpoints/vqvae/best_loss.pkl")
