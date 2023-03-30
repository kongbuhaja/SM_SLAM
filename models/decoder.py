import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Layer
from tensorflow.keras.initializers import GlorotUniform as glorot
from models.common import *
    
class Decoder(Layer):
    def __init__(self, num_classes, kernel_initializer=glorot, **kwargs):
        super().__init__(**kwargs)
        self.kernel_initializer = kernel_initializer

        self.branch1_conv = Conv(64, 1, kernel_initializer=self.kernel_initializer)
        self.branch2_conv = Conv(32, 1, kernel_initializer=self.kernel_initializer)
        
        self.transpose_conv1 = TransposeConv(256, 5, 2, kernel_initializer=self.kernel_initializer)

        self.conv1 = Conv(256, 5, kernel_initializer=self.kernel_initializer)
        self.transpose_conv2 = TransposeConv(256, 5, 2, kernel_initializer=self.kernel_initializer)
        
        self.conv2 = Conv(256, 5, kernel_initializer=self.kernel_initializer)
        
        self.conv3 = Conv(256, 5, kernel_initializer=self.kernel_initializer)
        
        self.conv4 = Conv(256, 5, kernel_initializer=self.kernel_initializer)
        
        self.conv5 = Conv(num_classes, 1, kernel_initializer=self.kernel_initializer, activate=False, bn=False)

    def call(self, input, branch1, branch2, training=False):
        branch2 = self.branch2_conv(branch2, training)
        x = Upsample(2)(input)
        # x = self.transpose_conv1(input, training)
        x = tf.concat([x, branch2], -1)
        x = self.conv1(x, training)
        
        x = Upsample(2)(x)
        # x = self.transpose_conv2(x)
        branch1 = self.branch1_conv(branch1, training)
        x = tf.concat([x, branch1], -1)
        x = self.conv2(x, training)

        x = Upsample(2)(x)
        x = self.conv3(x, training)
        
        x = Upsample(2)(x)
        x = self.conv4(x, training)
        x = self.conv5(x, training)
        # x = Upsample(4)(x)
        
        return x