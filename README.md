# GANime

This program implements several deep generative models for generating colorized anime characters based on sketch drawings. There are three main models used in this project, including Neural Style Transfer, Conditional GAN (Pix2Pix), and CycleGAN. [[Paper]](https://drive.google.com/file/d/1lubQpV6RLaykZOroFpk6AMrSW5djzTIO/view?usp=sharing) [[Poster]](https://drive.google.com/file/d/1EpBrz6kuhmAfMsv6CDTD0tgZkV2ah-60/view?usp=sharing)

## Demo

This section displays some outputs of the program. Each example contains an input image (left), a ground truth image (middle), and a generated image (right).

<p align="center">
  <img src="https://user-images.githubusercontent.com/46636857/77137430-806a8e80-6aa0-11ea-8cd8-56d17de21835.png">
</p>

<p align="center">
  <img src="https://user-images.githubusercontent.com/46636857/77137442-92e4c800-6aa0-11ea-8fb3-133146c1b32a.png">
</p>

## Installation

The project requires the following frameworks:

- TensorFlow 2.1.0: https://www.tensorflow.org

- TensorBoard: https://www.tensorflow.org/tensorboard

- TensorFlow Hub: https://www.tensorflow.org/hub

- NumPy: https://numpy.org

- Matplotlib: https://matplotlib.org

- Kaggle API: https://github.com/Kaggle/kaggle-api

## Download and preprocess data

This project uses the Anime Sketch Colorization Pair dataset from Kaggle. To get the data, go to https://www.kaggle.com/ktaebum/anime-sketch-colorization-pair, download the dataset, and unzip it in the ```data``` folder. Another option is to run the following command (with Kaggle API installed):

```bash
python utils/download_data.py
```

After that, run the following command to preprocess the data:

```bash
python utils/preprocess_data.py --model <model>
```

## Train the models

- Neural Style Transfer

```bash
python train.py --model neural_style_transfer --epochs 1000 --content-path </path/to/content/image/>  --style-path </path/to/style/image/> --output-path </path/to/output/image/>
```

- Fast Neural Style Transfer

```bash
python train.py --model fast_neural_style_transfer --content-path </path/to/content/image/>  --style-path </path/to/style/image/> --output-path </path/to/output/image/>
```

- Pix2Pix

```bash
python train.py --model pix2pix --epochs 150 --lr 2e-4 --batch-size 32 --data-path </path/to/dataset/> --resume --output-path </path/to/outputs/> --checkpoint-path </path/to/checkpoints/>
```

- CycleGAN

```bash
python train.py --model cyclegan --epochs 150 --lr 2e-4 --batch-size 8 --data-path </path/to/dataset/> --resume --output-path </path/to/outputs/> --checkpoint-path </path/to/checkpoints/>
```

## Test the models

- Neural Style Transfer

```bash
python test.py --model neural_style_transfer --data-path </path/to/dataset/> --output-path </path/to/outputs/>
```

- Fast Neural Style Transfer

```bash
python test.py --model fast_neural_style_transfer --data-path </path/to/dataset/> --output-path </path/to/outputs/>
```

- Pix2Pix

```bash
python test.py --model pix2pix --data-path </path/to/dataset/> --output-path </path/to/outputs/> --checkpoint-path </path/to/checkpoints/>
```

- CycleGAN

```bash
python test.py --model cyclegan --data-path </path/to/dataset/> --output-path </path/to/outputs/> --checkpoint-path </path/to/checkpoints/>
```

## Evaluate the models

The program implements two evaluation metrics, including the Structural Similarity (SSIM) Index and the Frechet Inception Distance (FID). To evaluate the models, run the following command:

```bash
python evaluate.py --model <model> --metric <metric> --output-path </path/to/outputs/>
```

## Authors

* **Tai Vu** - Stanford University

* **Robert Yang** - Stanford University


## References

- Gatys, Leon A., et al. “A Neural Algorithm of Artistic Style.” ArXiv:1508.06576 [Cs, q-Bio], Sept. 2015. arXiv.org, http://arxiv.org/abs/1508.06576.

- Ghiasi, Golnaz, et al. “Exploring the Structure of a Real-Time, Arbitrary Neural Artistic Stylization Network.” ArXiv:1705.06830 [Cs], Aug. 2017. arXiv.org, http://arxiv.org/abs/1705.06830.

- Isola, Phillip, et al. “Image-to-Image Translation with Conditional Adversarial Networks.” ArXiv:1611.07004 [Cs], Nov. 2018. arXiv.org, http://arxiv.org/abs/1611.07004.

- Zhu, Jun-Yan, et al. “Unpaired Image-to-Image Translation Using Cycle-Consistent Adversarial Networks.” ArXiv:1703.10593 [Cs], Nov. 2018. arXiv.org, http://arxiv.org/abs/1703.10593.

- “Tutorials | TensorFlow Core.” TensorFlow, https://www.tensorflow.org/tutorials.

- Tensorflow/Examples. 2018. tensorflow, 2020. GitHub, https://github.com/tensorflow/examples.

- “Machine Learning Mastery.” Machine Learning Mastery, https://machinelearningmastery.com/.




