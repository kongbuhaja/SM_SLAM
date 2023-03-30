import tensorflow as tf
import numpy as np
import os, shutil, sys, cv2, glob
import xml.etree.ElementTree as ET
import tqdm
from config import *
from PIL import Image
from utils import aug_utils

class Dataset():
    def __init__(self, dtype=DTYPE, input_size=INPUT_SIZE, batch_size=BATCH_SIZE):
        self.dtype = dtype
        self.input_size = input_size
        self.batch_size = batch_size
        
    def __call__(self, split):
        tfrecord_path = f'./data/{self.dtype}/{split}.tfrecord'
        info_path = f'./data/{self.dtype}/{split}.info'
        
        if not self.tfrecord_exists(tfrecord_path):
            self.make_tfrecord(split, tfrecord_path, info_path)
            
        data = self.read_tfrecord(tfrecord_path)
        length = self.read_info(info_path)
        
        data = data.map(self.resize, num_parallel_calls=-1)
        data = data.batch(self.batch_size, drop_remainder=True, num_parallel_calls=-1)
        data = data.cache()
        
        if split == 'train':
            data = data.shuffle(buffer_size = length*10)
            data = data.map(self.augmentation, num_parallel_calls=-1)
            
        
        data = data.map(self.normalize, num_parallel_calls=-1).prefetch(1)
        
        return data, length
    
    @tf.function
    def resize(self, image, label):
        return aug_utils.tf_resize(image, label, self.input_size)
    
    @tf.function
    def augmentation(self, image, label):
        return aug_utils.tf_augmentation(image, label)
    
    @tf.function
    def normalize(self, image, label):
        return image/255., tf.round(label)
        
    def tfrecord_exists(self, tfrecord_path):        
        if os.path.exists(tfrecord_path):
            print(f'{tfrecord_path} is exist')
            return True
        return False
        
    def make_tfrecord(self, split, tfrecord_path, info_path):        
        raw_file_paths = self.get_file_paths(split, False)
        label_file_paths = self.get_file_paths(split, True)
        
        length = len(raw_file_paths)
        
        with open(info_path, 'w') as f:
            f.write(str(length))
        
        with tf.io.TFRecordWriter(tfrecord_path) as writer:
            for i in tqdm.tqdm(range(len(raw_file_paths))):
                raw_image = self.read_image(raw_file_paths[i])
                label_image = self.read_image(label_file_paths[i], True)
                writer.write(_data_features(raw_image, label_image))  
    
    def read_tfrecord(self, tfrecord_path):
        dataset = tf.data.TFRecordDataset(tfrecord_path, num_parallel_reads=-1) \
                        .map(self.parse_tfrecord_fn)
        return dataset
    
    def read_info(self, info_path):
        with open(info_path, 'r') as f:
            lines = f.readlines()
        return int(lines[0])
    
    def parse_tfrecord_fn(self, example):
        feature_description = {
            'raw_image': tf.io.FixedLenFeature([], tf.string),
            'label_image': tf.io.FixedLenFeature([], tf.string)
        }
        example = tf.io.parse_single_example(example, feature_description)
        example['raw_image'] = tf.io.decode_jpeg(example['raw_image'], channels=3)
        example['label_image'] = tf.io.decode_jpeg(example['label_image'], channels=1)
        
        return example['raw_image'], example['label_image']
    
    def get_file_paths(self, split, gt):
        path = f'./data/{self.dtype}/'
        
        if self.dtype=='cityscapes':
            if gt==True:
                path += 'gtFine_trainvaltest/gtFine/'
                filename = '/*labelIds.png'
            else:
                path += 'leftImg8bit_trainvaltest/leftImg8bit/'
                filename = '/*.png'
            path += split + '/'
            dirs = os.listdir(path)
            file_paths = []
            for d in dirs:
                file_paths += glob.glob(path + d + filename)
            return file_paths
            
        elif self.dtype=='carla':
            path += split +'/'
            if gt==True:
                path += 'semantic/'
            else:
                path += 'front/'
            filename = '*.png'
            
            file_paths = []
            file_paths += glob.glob(path + filename)
            return file_paths       
    
    def read_image(self, path, gray=False):
        if gray:
            color = cv2.IMREAD_GRAYSCALE
        else:
            color = cv2.IMREAD_COLOR
            
        img = cv2.imread(path, color)
        if gray:
            img = img[..., None]
        if not gray:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
        return img

def _data_features(raw_image, label_image):
    raw_image_feature = _image_feature(raw_image)
    label_image_feature = _image_feature(label_image)
    
    object_features = {
        'raw_image': raw_image_feature,
        'label_image': label_image_feature
    }
    example = tf.train.Example(features=tf.train.Features(feature=object_features))
    return example.SerializeToString()    
def _image_feature(value):
    if value.shape[-1] == 1:
        return _bytes_feature(tf.io.encode_jpeg(value, format='grayscale').numpy())
    return _bytes_feature(tf.io.encode_jpeg(value).numpy())
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
