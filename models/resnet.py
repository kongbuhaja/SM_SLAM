from models.common import *
from tensorflow.keras.layers import Layer, MaxPool2D
from tensorflow.keras.initializers import GlorotUniform as glorot

class Resnet50(Layer):
    def __init__(self, kernel_initializer=glorot, **kwargs):
        super().__init__(**kwargs)
        self.kernel_initializer = kernel_initializer

        self.conv1 = Conv(64, 7, 2, kernel_initializer=self.kernel_initializer)
        self.maxpool = MaxPool2D(3, 2, padding='same')
        self.resblock1 = [Resblock(64, i, kernel_initializer=self.kernel_initializer) for i in range(2)]
        self.resblock1 += [Resblock(64, 2, out=2, kernel_initializer=self.kernel_initializer),
                           Resblock(64, 2, kernel_initializer=self.kernel_initializer)]
        self.resblock2 = [Resblock(128, 0, out=2, kernel_initializer=self.kernel_initializer)]
        self.resblock2 += [Resblock(128, i, kernel_initializer=self.kernel_initializer) for i in range(1,4)]

        self.resblock3 = [Resblock(256, i, kernel_initializer=self.kernel_initializer) for i in range(5)]
        self.resblock3 += [Conv(256, 1, 1, kernel_initializer=self.kernel_initializer),
                           Conv(256, 1, 1, kernel_initializer=self.kernel_initializer)]
        

    def call(self, input, training=False):
        x = self.conv1(input, training)
        x = self.maxpool(x)

        for i in range(len(self.resblock1)):
            if i==2:
                x, branch1 = self.resblock1[i](x, training)
            else:
                x = self.resblock1[i](x, training)
            
        
        for i in range(len(self.resblock2)):
            if i==0:
                x, branch2 = self.resblock2[i](x, training)
            else:
                x = self.resblock2[i](x, training)
            
        for i in range(len(self.resblock3)):
            x = self.resblock3[i](x, training)
        
        return x, branch1, branch2