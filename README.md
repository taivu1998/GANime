# GANime

This program implements deep generative models for generating colorized anime characters based on sketch drawings. There are three main models used in this project: Neural Style Transfer, Conditional GAN (Pix2Pix), and CycleGAN.

## Installation

The project requires the following frameworks:

- TensorFlow 2.1.0: https://www.tensorflow.org

- TensorBoard: https://www.tensorflow.org/tensorboard

- NumPy: https://numpy.org

- Kaggle API: https://github.com/Kaggle/kaggle-api

## Download the dataset

This project uses Anime Sketch Colorization Pair dataset from Kaggle. To download the dataset, go to https://www.kaggle.com/ktaebum/anime-sketch-colorization-pair or run the following command (with Kaggle API installed):

```bash
python utils/download_data.py
```

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

## Test the models

- Neural Style Transfer

```bash
python test.py --model neural_style_transfer --data-path /path/to/dataset/ --output-path /path/to/outputs/ 
```

- Fast Neural Style Transfer

```bash
python test.py --model fast_neural_style_transfer --data-path /path/to/dataset/ --output-path /path/to/outputs/ 
```

- Pix2Pix

```bash
python test.py --model pix2pix --data-path /path/to/dataset/ --output-path /path/to/outputs/ --checkpoint-path /path/to/checkpoints/ 
```

- CycleGAN

```bash
python test.py --model cyclegan --data-path /path/to/dataset/ --output-path /path/to/outputs/ --checkpoint-path /path/to/checkpoints/ 
```

## Evaluate the models

The program implements two evaluation metrics, including FID and SSIM. To evaluate the models, run the following command:

```bash
python evaluate.py --model [model] --metric [metric]
```
