# CelebA Face Reconstruction Autoencoder

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

A convolutional autoencoder that learns to compress and reconstruct human face
images from the CelebA dataset, implemented end-to-end in PyTorch.

## Overview

This project trains an unsupervised convolutional autoencoder on a subset of
the [CelebA](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) face dataset.
The model learns a compact latent representation of a face image and
reconstructs it back to full resolution, using pixel-wise mean squared error
as the training objective. The notebook covers the full pipeline: data
loading, model architecture, training with validation tracking, and
qualitative visualization of reconstruction quality.

## Features

- Clean, modular PyTorch implementation with a configurable `Config` dataclass
- Reproducible experiments via consistent seeding
- Automatic CelebA download and reproducible subset sampling
- Convolutional encoder-decoder architecture with strided/transposed convolutions
- Per-epoch training **and** validation loss tracking
- Loss curve visualization and side-by-side original vs. reconstructed image comparisons
- GPU-accelerated when CUDA is available, with automatic CPU fallback

## Project Structure

```
celeba-autoencoder/
│
├── celeba_autoencoder.ipynb   # Main notebook: data, model, training, evaluation
├── README.md                  # Project documentation (this file)
├── requirements.txt           # Python dependencies
├── LICENSE                    # MIT License
├── .gitignore                 # Ignored files (data, checkpoints, caches)
├── data/                      # CelebA dataset (downloaded automatically, gitignored)
├── assets/                    # Images used in this README
└── output/                    # Saved plots / sample reconstructions (optional)
```

## Requirements

- Python 3.9+
- PyTorch 2.0+
- Torchvision 0.15+
- Matplotlib
- NumPy

See [`requirements.txt`](requirements.txt) for pinned minimum versions.

## Installation

```bash
git clone https://github.com/<your-username>/celeba-autoencoder.git
cd celeba-autoencoder
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

1. Download the [CelebA dataset](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)
   (the aligned & cropped `img_align_celeba` image folder) and set
   `Config.data_root` in the notebook to point to that folder.
2. Launch Jupyter:
   ```bash
   jupyter notebook celeba_autoencoder.ipynb
   ```
3. Run all cells in order. Images are loaded directly from your local
   `data_root` folder — no download step runs automatically, avoiding
   torchvision's Google Drive quota/permission issues.
4. Training progress (train/validation loss per epoch) prints to the notebook
   output, followed by a loss curve and reconstruction comparison plot.

> **Note:** the notebook uses a small, seeded subset (5,000 training / 500
> validation images) drawn from your local folder by default — adjust
> `train_subset_size` / `val_subset_size` in `Config` for a more thorough run.

## Example Output

The notebook produces two key visual outputs:

- **Loss curve** — training vs. validation MSE loss across epochs
- **Reconstruction grid** — original CelebA faces next to their autoencoder reconstructions

## Configuration

All hyperparameters live in a single `Config` dataclass at the top of the
notebook:

| Parameter           | Default | Description                              |
|---------------------|---------|--------------------------------------------|
| `batch_size`         | 64      | Samples per batch                          |
| `epochs`             | 10     | Training epochs                            |
| `learning_rate`      | 1e-4    | Adam optimizer learning rate               |
| `image_size`         | 64      | Resize/crop resolution (pixels)            |
| `train_subset_size`  | 5000    | Number of training images sampled          |
| `val_subset_size`    | 500     | Number of validation images sampled        |
| `data_root`          | *(local path)* | Folder containing pre-downloaded CelebA `.jpg` images |
| `seed`               | 42      | Random seed for reproducibility            |

## Future Work

- Train on the full CelebA dataset given sufficient compute
- Add a learning-rate scheduler and early stopping
- Extend to a Variational Autoencoder (VAE) for latent-space sampling
- Add perceptual loss for sharper reconstructions
- Integrate experiment tracking (TensorBoard / Weights & Biases)

## License

This project is licensed under the [MIT License](LICENSE).

## Author

Developed by **Mehran**.

## Acknowledgements

- Liu, Z., Luo, P., Wang, X., & Tang, X. (2015). *Deep Learning Face Attributes
  in the Wild.* ICCV.
- The [PyTorch](https://pytorch.org/) and [Torchvision](https://pytorch.org/vision/stable/index.html) teams.
