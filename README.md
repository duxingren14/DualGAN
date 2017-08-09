
 # DualGAN
DualGAN: unsupervised dual learning for image-to-image translation

# architecture of DualGAN

![architecture](https://github.com/duxingren14/DualGAN/blob/master/0.png)



# How to setup

## Prerequisites

Linux

Python 

numpy

scipy

NVIDIA GPU + CUDA 8.0 + CuDNNv5.1

TensorFlow 1.0



# Getting Started

Clone this repo:

git clone https://github.com/duxingren14/DualGAN.git

cd DualGAN

To download datasets (e.g., sketch-photo), run:

bash ./datasets/download_dataset.sh sketch-photo

Train the model:

python main.py --phase train --dataset_name sketch-photo --image_size 256 --epoch 45 --lambda_A 20.0 --lambda_B 20.0 --A_channels 1 --B_channels 1


Test the model:

python main.py --phase test --dataset_name sketch-photo --image_size 256 --epoch 45 --lambda_A 20.0 --lambda_B 20.0 --A_channels 1 --B_channels 1



Similarly, run experiments on facades dataset with the following commands:

bash ./datasets/download_dataset.sh facades

python main.py --phase train --dataset_name facades --image_size 256 --epoch 45 --lambda_A 20.0 --lambda_B 20.0 --A_channels 3 --B_channels 3

python main.py --phase test --dataset_name facades --image_size 256 --epoch 45 --lambda_A 20.0 --lambda_B 20.0 --A_channels 3 --B_channels 3


# Some of Datasets are available from:

facades: http://cmp.felk.cvut.cz/~tylecr1/facade/

sketch: http://mmlab.ie.cuhk.edu.hk/archive/cufsf/

maps: https://mega.nz/#!r8xwCBCD!lNBrY_2QO6pyUJziGj7ikPheUL_yXA8xGXFlM3GPL3c

oil-chinese:  http://www.cs.mun.ca/~yz7241/, jump to http://www.cs.mun.ca/~yz7241/dataset/

day-night: http://www.cs.mun.ca/~yz7241/dataset/


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
