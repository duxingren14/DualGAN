"""
Some codes from https://github.com/Newmu/dcgan_code
"""
from __future__ import division
import random
import cv2
import numpy as np
import os
from time import gmtime, strftime



    
def load_data(image_path, flip=False, is_test=False, image_size = 128):
    img = load_image(image_path)
    img = preprocess_img(img, img_size=image_size, flip=flip, is_test=is_test)

    img = img/127.5 - 1.
    if len(img.shape)<3:
        img = np.expand_dims(img, axis=2)
    return img

def load_image(image_path):
    img = imread(image_path)
    return img

def preprocess_img(img, img_size=128, flip=False, is_test=False):
    img = cv2.resize(img, (img_size, img_size))
    if (not is_test) and flip and np.random.random() > 0.5:
        img = np.fliplr(img)
    return img

def get_image(image_path, image_size, is_crop=True, resize_w=64, is_grayscale = False):
    return transform(imread(image_path, is_grayscale), image_size, is_crop, resize_w)

def save_images(images, size, image_path):
    dir = os.path.dirname(image_path)
    if not os.path.exists(dir):
        os.makedirs(dir)
    return imsave(inverse_transform(images), size, image_path)

def imread(path, is_grayscale = False):
    if (is_grayscale):
        return cv2.imread(path, flatten = True)#.astype(np.float)
    else:
        return cv2.imread(path)#.astype(np.float)

def merge_images(images, size):
    return inverse_transform(images)

def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    if len(images.shape) < 4:
        img = np.zeros((h * size[0], w * size[1], 1))
        images = np.expand_dims(images, axis = 3)
    else:
        img = np.zeros((h * size[0], w * size[1], images.shape[3]))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        img[j*h:j*h+h, i*w:i*w+w, :] = image
    if images.shape[3] ==1:
        return np.concatenate([img,img,img],axis=2)
    else:
        return img.astype(np.uint8)

def imsave(images, size, path):
    return cv2.imwrite(path, merge(images, size))

def transform(image, npx=64, is_crop=True, resize_w=64):
    # npx : # of pixels width/height of image
    if is_crop:
        cropped_image = center_crop(image, npx, resize_w=resize_w)
    else:
        cropped_image = image
    return np.array(cropped_image)/127.5 - 1.

def inverse_transform(images):
    return ((images+1.)*127.5)