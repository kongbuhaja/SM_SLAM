import tensorflow as tf
from tensorflow.keras import Model
from models.xception import *
from models.resnet import *
from models.decoder import *
from models.common import *
from losses.seg_loss import seg_loss
from config import *
from tensorflow.keras.initializers import GlorotUniform as glorot


class Model_(Model):
    def __init__(self, num_classes=NUM_CLASSES, backbone=BACKBONE_TYPE, kernel_initializer=glorot, **kwargs):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.loss = seg_loss()
        self.kernel_initializer = kernel_initializer

        if backbone=='xception':
            self.encoder = Xception(self.kernel_initializer)
        elif backbone=='resnet':
            self.encoder = Resnet50(self.kernel_initializer)
            
        self.aspp = ASPP(kernel_initializer=self.kernel_initializer)
        
        self.decoder = Decoder(num_classes, self.kernel_initializer)

    def call(self, input, training=False):
        x, branch1, branch2 = self.encoder(input, training)
        x = self.aspp(x, training)
        x = self.decoder(x, branch1, branch2, training)
        
        return x   