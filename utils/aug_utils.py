import tensorflow as tf
import numpy as np
import cv2

@tf.function
def tf_augmentation(image, label):
    color_methods = [random_brigthness, random_hue, random_saturation]
    geometric_methods = [random_flip_horizontally, random_flip_vertically]
    for augmentation_method in geometric_methods + color_methods:
        image, label = tf_randomly_apply(augmentation_method, image, label)
    return image, label

@tf.function
def tf_resize(image, label, size):
    image_resized = tf.image.resize(image, size)
    label_resized = tf.image.resize(label, size)
    return image_resized, label_resized

@tf.function
def tf_randomly_apply(method, image, label):
    if tf.random.uniform(())>0.5:
        return method(image, label)
    return image, label

@tf.function
def random_flip_horizontally(image, label):    
    return tf.image.flip_left_right(image), tf.image.flip_left_right(label)

@tf.function
def random_flip_vertically(image, label):
    return tf.image.flip_up_down(image), tf.image.flip_up_down(label)

@tf.function
def random_saturation(image, label, lower=0.5, upper=1.5):
    return tf.image.random_saturation(image, lower, upper), label

@tf.function
def random_hue(image, label, max_delta=0.08):
    return tf.image.random_hue(image, max_delta), label

@tf.function
def random_contrast(image, label, lower=0.5, upper=1.5):
    return tf.image.random_contrast(image, lower, upper), label

@tf.function
def random_brigthness(image, label, max_delta=0.12):
    return tf.image.random_brightness(image, max_delta), label
