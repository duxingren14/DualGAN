
 # DualGAN
DualGAN: unsupervised dual learning for image-to-image translation

# architecture of DualGAN

![day2night](https://github.com/duxingren14/DualGAN/blob/master/0.png)



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

Train the model:

python main.py --phase train --dataset_name dataset_name

Test the model:

python main.py --phase test --dataset_name dataset_name




# A portion of Datasets are available from:

facades: http://cmp.felk.cvut.cz/~tylecr1/facade/

sketch: http://mmlab.ie.cuhk.edu.hk/archive/cufsf/

maps: https://mega.nz/#!r8xwCBCD!lNBrY_2QO6pyUJziGj7ikPheUL_yXA8xGXFlM3GPL3c


# Experimental results:

![day2night](https://github.com/duxingren14/DualGAN/blob/master/1.PNG)
![day2night](https://github.com/duxingren14/DualGAN/blob/master/2.PNG)


![day2night](https://github.com/duxingren14/DualGAN/blob/master/4.PNG)

![day2night](https://github.com/duxingren14/DualGAN/blob/master/5.PNG)

![day2night](https://github.com/duxingren14/DualGAN/blob/master/3.PNG)

![day2night](https://github.com/duxingren14/DualGAN/blob/master/6.PNG)




# Acknowledgments

Codes are built on the top of pix2pix-tensorflow and DCGAN-tensorflow. Thanks for their precedent contributions!
