# GANime: Generating Anime and Manga Character Drawings from Sketches with Deep Learning

<p align="center">
  <a href="https://arxiv.org/abs/2508.09207"><img src="https://img.shields.io/badge/arXiv-2508.09207-b31b1b.svg" alt="arXiv"></a>
  <a href="https://doi.org/10.48550/arXiv.2508.09207"><img src="https://img.shields.io/badge/DOI-10.48550%2FarXiv.2508.09207-blue" alt="DOI"></a>
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License: MIT"></a>
  <img src="https://img.shields.io/badge/TensorFlow-2.1.0-orange?logo=tensorflow" alt="TensorFlow">
  <img src="https://img.shields.io/badge/Python-3.7+-3776AB?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/AWS-EC2-FF9900?logo=amazonwebservices&logoColor=white" alt="AWS">
</p>

> **Tai Vu**, **Robert Yang** -- Stanford University
>
> [[Paper]](https://arxiv.org/abs/2508.09207) [[Poster]](https://drive.google.com/file/d/1EpBrz6kuhmAfMsv6CDTD0tgZkV2ah-60/view?usp=sharing)

A comparative study of deep generative models for automatic anime sketch colorization. We implement and benchmark **Neural Style Transfer**, **Conditional GAN (Pix2Pix)**, and **CycleGAN** on 17,769 sketch-color pairs, achieving state-of-the-art results of **220.5 FID** and **0.76 SSIM** with our modified C-GAN incorporating total variation regularization.

## Results

### Quantitative Evaluation

All models were evaluated on 100 held-out images using FID (lower is better) and SSIM (higher is better):

| Model | FID | SSIM (mean) | SSIM (std) |
|:---|:---:|:---:|:---:|
| Neural Style Transfer | 345.506 | 0.6547 | 0.0989 |
| CycleGAN | 272.619 | 0.7238 | 0.0824 |
| C-GAN (Pix2Pix) | 227.948 | 0.7469 | 0.0741 |
| **C-GAN + TV Loss (Ours)** | **220.499** | **0.7559** | **0.0738** |

Our modified C-GAN reduces FID by **36.2%** and improves SSIM by **15.5%** over Neural Style Transfer, producing high-quality, high-resolution colorizations visually close to human-drawn artwork.

### Qualitative Samples

Each triplet: input sketch (left), ground truth (middle), generated output (right).

<p align="center">
  <img src="https://user-images.githubusercontent.com/46636857/77137430-806a8e80-6aa0-11ea-8cd8-56d17de21835.png">
</p>

<p align="center">
  <img src="https://user-images.githubusercontent.com/46636857/77137442-92e4c800-6aa0-11ea-8fb3-133146c1b32a.png">
</p>

## Architecture

### Pix2Pix (Conditional GAN)

The best-performing model uses a **U-Net generator** with an **N=70 PatchGAN discriminator**:

**Generator (U-Net):** 8-block encoder-decoder with skip connections.
- Encoder: `Conv2D(k=4, s=2)` &rarr; `BatchNorm` &rarr; `LeakyReLU(0.2)` per block. Filter progression: 64 &rarr; 128 &rarr; 256 &rarr; 512 &rarr; 512 &rarr; 512 &rarr; 512 &rarr; 512, producing a 1&times;1 bottleneck.
- Decoder: `Conv2DTranspose(k=4, s=2)` &rarr; `BatchNorm` &rarr; `ReLU` with skip connections concatenating encoder features. Dropout(0.5) on the first 3 decoder blocks for regularization.
- Output: `Conv2DTranspose(3, tanh)` producing 256&times;256&times;3 RGB images in [-1, 1].

**Discriminator (PatchGAN):** Classifies overlapping 70&times;70 patches rather than the full image, enforcing high-frequency structural correctness.
- Input: 6-channel concatenation of `[sketch, target/generated]`.
- Architecture: 3 downsampling blocks (64 &rarr; 128 &rarr; 256) &rarr; `Conv2D(512, s=1)` &rarr; `BatchNorm` &rarr; `Conv2D(1)` outputting a 30&times;30 patch-level classification map.

**Composite Loss:**

```
L(G, D) = L_cGAN(G, D) + lambda_L1 * L_L1(G) + lambda_tv * L_tv(G)
```

| Loss Term | Formulation | Weight |
|:---|:---|:---:|
| Adversarial | `E[log D(x,y)] + E[log(1 - D(x, G(x,z)))]` | 1.0 |
| L1 Reconstruction | `E[‖y - G(x,z)‖₁]` | 100.0 |
| Total Variation | `sum(\|y_{i+1,j} - y_{i,j}\| + \|y_{i,j+1} - y_{i,j}\|)` | 1e-4 |

### CycleGAN

Enables **unpaired** sketch-to-color translation with two generator-discriminator pairs:

- **Generators G, F:** U-Net with **Instance Normalization** (replacing BatchNorm for style-invariant feature normalization). G: sketch &rarr; color, F: color &rarr; sketch.
- **Discriminators D_X, D_Y:** PatchGAN with Instance Normalization.

**Loss:** Adversarial + cycle consistency (`lambda_cyc = 10`) + identity loss (`0.5 * lambda_cyc`):

```
L(G, F, D_X, D_Y) = L_GAN(G, D_Y) + L_GAN(F, D_X) + lambda_cyc * L_cyc(G, F)
L_cyc(G, F)        = E[‖F(G(x)) - x‖₁] + E[‖G(F(y)) - y‖₁]
```

### Neural Style Transfer

- **Optimization-based NST:** Iteratively optimizes pixel values of the stylized image using a frozen VGG19 backbone. Content features from `block5_conv2`; style features via Gram matrices from `block{1..5}_conv1`.
- **Fast NST:** Single forward pass through Google Magenta's Arbitrary Image Stylization v1-256 network via TensorFlow Hub.

## Dataset

**Anime Sketch Colorization Pair** ([Kaggle](https://www.kaggle.com/ktaebum/anime-sketch-colorization-pair))

| Split | Images | Resolution | Format |
|:---|:---:|:---:|:---|
| Train | 14,224 | 256 &times; 256 | Paired sketch-color PNG |
| Test | 3,545 | 256 &times; 256 | Paired sketch-color PNG |
| **Total** | **17,769** | | |

**Preprocessing pipeline:**
1. Original 512&times;1024 images split at midpoint into sketch and color channels
2. Both resized to 256&times;256 (nearest-neighbor interpolation)
3. Pixel values normalized to [-1, 1]: `(pixel / 127.5) - 1`
4. **Augmentation:** Resize to 286&times;286 &rarr; random crop to 256&times;256 + random horizontal flip
5. Shuffled per epoch with buffer size 400

## Training Configuration

| Hyperparameter | Pix2Pix | CycleGAN | NST |
|:---|:---:|:---:|:---:|
| Optimizer | Adam | Adam | Adam |
| Learning rate | 2e-4 | 2e-4 | 0.02 |
| beta_1, beta_2 | 0.5, 0.999 | 0.5, 0.999 | 0.99, 0.999 |
| epsilon | 1e-7 | 1e-7 | 0.1 |
| Batch size | 32 | 8 | 1 |
| Epochs | 150 | 150 | 1,000 |
| Checkpoint freq | 5 epochs | 5 epochs | -- |
| Infrastructure | AWS EC2 + Colab GPU | AWS EC2 + Colab GPU | Colab GPU |

## Project Structure

```
GANime/
├── models/
│   ├── networks.py                     # U-Net generator & PatchGAN discriminator
│   ├── Pix2PixModel.py                 # C-GAN training loop with L1 + TV loss
│   ├── CycleGANModel.py                # Dual-generator cycle-consistent training
│   ├── NeuralStyleTransferModel.py     # VGG19 Gram-matrix optimization
│   └── FastNeuralStyleTransferModel.py # TF Hub single-pass stylization
├── dataloaders/
│   ├── Pix2PixDataLoader.py            # Paired image loading & augmentation
│   ├── CycleGANDataLoader.py           # Unpaired domain loading & augmentation
│   └── NeuralStyleTransferDataLoader.py
├── utils/
│   ├── evaluation_metrics.py           # FID (InceptionV3) & SSIM computation
│   ├── preprocess_data.py              # Dataset splitting & format conversion
│   └── download_data.py                # Kaggle API dataset fetcher
├── options/
│   ├── TrainOptions.py                 # Training CLI arguments
│   ├── TestOptions.py                  # Inference CLI arguments
│   └── EvaluateOptions.py              # Evaluation CLI arguments
├── train.py                            # Unified training entrypoint
├── test.py                             # Inference & output generation
└── evaluate.py                         # Metric computation (FID / SSIM)
```

## Getting Started

### Prerequisites

```
TensorFlow >= 2.1.0
TensorFlow Hub
NumPy
Matplotlib
SciPy
Kaggle API
```

### Download & Preprocess Data

```bash
# Download via Kaggle API
python utils/download_data.py

# Preprocess for target model
python utils/preprocess_data.py --model pix2pix      # paired format
python utils/preprocess_data.py --model cyclegan      # unpaired format
```

### Training

```bash
# Pix2Pix (recommended -- best results)
python train.py --model pix2pix --epochs 150 --lr 2e-4 --batch-size 32 \
    --use-tv-loss --lambda-tv-loss 1e-4 \
    --data-path <data_dir> --output-path outputs/ --checkpoint-path checkpoints/

# CycleGAN
python train.py --model cyclegan --epochs 150 --lr 2e-4 --batch-size 8 \
    --data-path <data_dir> --output-path outputs/ --checkpoint-path checkpoints/

# Neural Style Transfer
python train.py --model neural_style_transfer --epochs 1000 \
    --content-path <content_img> --style-path <style_img> --output-path outputs/

# Fast Neural Style Transfer (single-pass, no training required)
python train.py --model fast_neural_style_transfer \
    --content-path <content_img> --style-path <style_img> --output-path outputs/
```

Add `--resume` to continue training from the latest checkpoint.

### Inference

```bash
python test.py --model pix2pix --data-path <data_dir> \
    --output-path outputs/ --checkpoint-path checkpoints/
```

### Evaluation

```bash
# Frechet Inception Distance
python evaluate.py --model pix2pix --metric fid --output-path outputs/

# Structural Similarity Index
python evaluate.py --model pix2pix --metric ssim --output-path outputs/
```

FID uses InceptionV3 (ImageNet-pretrained) activations to measure distributional similarity between generated and real images. SSIM measures per-pixel structural fidelity (luminance, contrast, structure) with `filter_size=11`, `k1=0.01`, `k2=0.03`.

## Key Findings

- **C-GAN dominates** across both metrics due to paired supervision and the L1 reconstruction objective, which provides strong pixel-level gradients that stabilize adversarial training.
- **Total variation regularization** yields a further 3.3% FID improvement by suppressing high-frequency artifacts and color bleeding at region boundaries.
- **CycleGAN** produces reasonable colorizations despite unpaired training, but cycle consistency alone is insufficient to match paired supervision quality.
- **SSIM plateaus early (~epoch 10)** while **FID continues improving until ~epoch 35**, with qualitative improvements persisting through epoch 100 -- indicating that perceptual quality diverges from pixel-level metrics at later stages of training.
- The PatchGAN discriminator's patch-level classification enforces high-frequency detail accuracy, particularly for hair strands, eye highlights, and clothing folds.

## Citation

```bibtex
@article{vu2025ganime,
  title={GANime: Generating Anime and Manga Character Drawings from Sketches with Deep Learning},
  author={Vu, Tai and Yang, Robert},
  journal={arXiv preprint arXiv:2508.09207},
  year={2025}
}
```

## References

1. Gatys, L. A., Ecker, A. S., & Bethge, M. (2015). A Neural Algorithm of Artistic Style. *arXiv:1508.06576*.
2. Ghiasi, G., et al. (2017). Exploring the Structure of a Real-Time, Arbitrary Neural Artistic Stylization Network. *arXiv:1705.06830*.
3. Isola, P., et al. (2017). Image-to-Image Translation with Conditional Adversarial Networks. *CVPR 2017*.
4. Zhu, J.-Y., et al. (2017). Unpaired Image-to-Image Translation Using Cycle-Consistent Adversarial Networks. *ICCV 2017*.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
