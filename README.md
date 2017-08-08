
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

To download datasets, run:

bash ./datasets/download_dataset.sh

Train the model:

python main.py --phase train --dataset_name dataset_name 

Test the model:

python main.py --phase test --dataset_name dataset_name



# A portion of Datasets are available from:

facades: http://cmp.felk.cvut.cz/~tylecr1/facade/

sketch: http://mmlab.ie.cuhk.edu.hk/archive/cufsf/

maps: https://mega.nz/#!r8xwCBCD!lNBrY_2QO6pyUJziGj7ikPheUL_yXA8xGXFlM3GPL3c

oil-chinese:  http://www.cs.mun.ca/~yz7241/, jump to http://www.cs.mun.ca/~yz7241/dataset/

day-night: http://www.cs.mun.ca/~yz7241/dataset/


# Experimental results:

![day2night](https://github.com/duxingren14/DualGAN/blob/master/1.PNG)
![day2night](https://github.com/duxingren14/DualGAN/blob/master/5.PNG)
![day2night](https://github.com/duxingren14/DualGAN/blob/master/6.PNG)



![oil2chinese](https://github.com/duxingren14/DualGAN/blob/master/A_1_realA.PNG)
![oil2chinese](https://github.com/duxingren14/DualGAN/blob/master/A_1_A2B.PNG)
![oil2chinese](https://github.com/duxingren14/DualGAN/blob/master/A_1_A2B2A.PNG)

![oil2chinese](https://github.com/duxingren14/DualGAN/blob/master/A_3_realA.PNG)
![oil2chinese](https://github.com/duxingren14/DualGAN/blob/master/A_3_A2B.PNG)
![oil2chinese](https://github.com/duxingren14/DualGAN/blob/master/A_3_A2B2A.PNG)

![chinese2oil](https://github.com/duxingren14/DualGAN/blob/master/B_1161_realB.PNG)
![chinese2oil](https://github.com/duxingren14/DualGAN/blob/master/B_1161_B2A.PNG)
![chinese2oil](https://github.com/duxingren14/DualGAN/blob/master/B_1161_B2A2B.PNG)

![chinese2oil](https://github.com/duxingren14/DualGAN/blob/master/B_1143_realB.PNG)
![chinese2oil](https://github.com/duxingren14/DualGAN/blob/master/B_1143_B2A.PNG)
![chinese2oil](https://github.com/duxingren14/DualGAN/blob/master/B_1143_B2A2B.PNG)

# Acknowledgments

Codes are built on the top of pix2pix-tensorflow and DCGAN-tensorflow. Thanks for their precedent contributions!
