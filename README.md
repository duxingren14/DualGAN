
 # ICCV paper of DualGAN
<a href="https://arxiv.org/abs/1704.02510">DualGAN: unsupervised dual learning for image-to-image translation</a>

please cite the paper, if the codes has been used for your research.

# architecture of DualGAN

![architecture](https://github.com/duxingren14/DualGAN/blob/master/0.png)

# How to setup

## Prerequisites

* Linux

* Python (2.7 or later)

* numpy

* scipy

* NVIDIA GPU + CUDA 8.0 + CuDNN v5.1

* TensorFlow 1.0 or later


# Getting Started
## steps

* clone this repo:

```
git clone https://github.com/duxingren14/DualGAN.git

cd DualGAN
```

* download datasets (e.g., sketch-photo), run:

```
bash ./datasets/download_dataset.sh sketch-photo
```

* download pre-trained model (e.g., sketch-photo), run:

```
bash ./checkpoint/download_ckpt.sh sketch-photo
```

* train the model:

```
python main.py --phase train --dataset_name sketch-photo --image_size 256 --lambda_A 1000.0 --lambda_B 1000.0 --epoch 100
```

* test the model:

```
python main.py --phase test --dataset_name sketch-photo --image_size 256 --lambda_A 1000.0 --lambda_B 1000.0 --epoch 100
```

## optional

Similarly, run experiments on facades dataset with the following commands:

```
bash ./datasets/download_dataset.sh facades

python main.py --phase train --dataset_name facades --lambda_A 1000.0 --lambda_B 1000.0 --epoch 100

python main.py --phase test --dataset_name facades --lambda_A 1000.0 --lambda_B 1000.0 --epoch 100
```

for thoese who cannot download datasets from using the scripts

<a href="https://drive.google.com/drive/folders/1i7hvUocQ5-u9K1QcD_NjIEKgkTWB7QMh?usp=sharing">all datasets from google drive</a>


# Experimental results:

![day2night](https://github.com/duxingren14/DualGAN/blob/master/6.PNG)
![da2ni](https://github.com/duxingren14/DualGAN/blob/master/da2ni.png)
![la2ph](https://github.com/duxingren14/DualGAN/blob/master/la2ph.png)
![ph2la](https://github.com/duxingren14/DualGAN/blob/master/ph2la.png)
![sk2ph](https://github.com/duxingren14/DualGAN/blob/master/sk2ph.png)
![ph2sk](https://github.com/duxingren14/DualGAN/blob/master/ph2sk.png)
![ch2oi](https://github.com/duxingren14/DualGAN/blob/master/ch2oi.png)
![oi2ch](https://github.com/duxingren14/DualGAN/blob/master/oi2ch.png)

# Acknowledgments

Codes are built on the top of pix2pix-tensorflow and DCGAN-tensorflow. Thanks for their precedent contributions!
