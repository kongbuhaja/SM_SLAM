import tensorflow as tf
from config import *

class seg_loss():
    def __init__(self, norm_type=NORM, loss_type=LOSS):
        if norm_type=='softmax':
            self.norm_fn = tf.nn.softmax
        elif norm_type=='sigmoid':
            self.norm_fn = tf.nn.sigmoid
        if loss_type=='ce':
            self.loss_fn = self.CrossEntropyLoss
        elif loss_type=='bce':
            self.loss_fn = self.binaryCrossEntropy
        elif loss_type=='mse':
            self.loss_fn = self.MSE
            
        
    def CrossEntropyLoss(self, target, pred):
        pred = tf.maximum(pred, 1e-7)       
        return -(target*tf.math.log(pred))
    
    def binaryCrossEntropy(self, target, pred, eps=1e-7):
        pred = tf.minimum(tf.maximum(pred, eps), 1-eps)        
        return -(target*tf.math.log(pred) + (1.-target)*tf.math.log(1.-pred))
        
    def MSE(self, target, pred, eps=1e-7):        
        target = tf.cast(target, tf.int32)[..., 0]
        target = tf.one_hot(target, pred.shape[-1], dtype=tf.float32)
        
        loss = tf.math.sqrt(tf.minimum(tf.maximum((target - pred)**2, eps), 1e+30))
        return loss
    
    def loss(self, target, pred):
        pred = self.norm_fn(pred)
        
        target = tf.cast(target[..., 0], tf.int32)        
        target = tf.one_hot(target, pred.shape[-1], dtype=tf.float32)
        
        loss = self.loss_fn(target, pred)
    
        # return tf.reduce_mean(tf.reduce_sum(loss, [1,2,3]))
        loss = tf.reduce_sum(loss, -1)
        return tf.reduce_mean(loss)
    
    def loss2(self, target, pred):
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        
        pred = self.norm_fn(pred)
        target = tf.minimum(tf.maximum(target[..., 0], 1e-7), 1 - 1e-7)
        loss = loss_fn(target, pred)
        
        return loss
    
    
    def __call__(self, target, pred):
        return self.loss(target, pred)