# GANime

This program implements deep generative models for generating colorized anime characters based on sketch drawings. There are three main models used in the program: Neural Style Transfer, Conditional GAN (Pix2Pix), and CycleGAN.

## Installation

The project requires the following frameworks:

- TensorFlow 2.1.0: https://www.tensorflow.org

- TensorBoard: https://www.tensorflow.org/tensorboard

- NumPy: https://numpy.org

## Download the dataset

## Train the models

- Neural Style Transfer

```bash
python train.py --model neural_style_transfer --epochs 1000 --content-path /path/to/content/image/  --style-path /path/to/style/image/ --output-path /path/to/output/image/
```

- Fast Neural Style Transfer

```bash
python train.py --model fast_neural_style_transfer --content-path /path/to/content/image/  --style-path /path/to/style/image/ --output-path /path/to/output/image/
```

- Pix2Pix

```bash
python train.py --model pix2pix --epochs 150 --lr 2e-4 --batch-size 32 --data-path /path/to/dataset/ --resume --output-path /path/to/outputs/ --checkpoint-path /path/to/checkpoints/ 
```

- CycleGAN

```bash
python train.py --model cyclegan --epochs 150 --lr 2e-4 --batch-size 8 --data-path /path/to/dataset/ --resume --output-path /path/to/outputs/ --checkpoint-path /path/to/checkpoints/ 
```
