The goal of this project is to evaluate to what extent it is possible to do face swapping using Vector Quantised Variational AutoEncoder (VQ-VAE)

To use the scripts as they are, your directories should be organised as follow:

main
|- scripts
    |- vqvae.py
    |- train_vqvae.py
    |- dataset.py
    |- utils.py
|- data
    |- Celeba-HQ-img
          |- .jpg
|- checkpoints
    |- vqvae
        |- .pkl
    |- merging
        |- .pkl
