from tensorflow.keras.layers import Conv2D, BatchNormalization, Layer, LeakyReLU, SeparableConv2D, Add, UpSampling2D, AveragePooling2D, Conv2DTranspose
from tensorflow.keras.initializers import GlorotUniform as glorot
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Model
import tensorflow as tf

class Conv(Layer):
    def __init__(self, units, kernel_size, strides=1, padding='same', dilation_rate=1, kernel_initializer=glorot, activate=True, bn=True, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.kernel_initializer = kernel_initializer
        self.activate = activate
        self.bn = bn
        self.padding = padding
        self.strides = strides

        self.conv = Conv2D(self.units, self.kernel_size, strides=self.strides, padding=self.padding, dilation_rate=self.dilation_rate,
                           use_bias=not self.bn, kernel_initializer=self.kernel_initializer)
        self.batchN = BatchNormalization()

    def call(self, input, training=False):
        x = self.conv(input)
        if self.bn:
            x = self.batchN(x, training)
        if self.activate:
            x = LeakyReLU(alpha=0.1)(x)
        
        return x    

class Resblock(Layer):
    def __init__(self, units, idx=True, out=1, kernel_initializer=glorot, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.kernel_initializer = kernel_initializer
        self.idx = idx
        self.out = out

        if not self.idx:
            if self.units==64:
                self.first_strides = 1
            else:
                self.first_strides = 2
            self.branch_conv = Conv(self.units*4, 1, self.first_strides, kernel_initializer=self.kernel_initializer, activate=False)
        else:
            self.first_strides = 1
    
        self.conv1 = Conv(self.units, 1, self.first_strides, kernel_initializer=self.kernel_initializer)
        self.conv2 = Conv(self.units, 3, 1, kernel_initializer=self.kernel_initializer)
        self.conv3 = Conv(self.units*4, 1, 1, kernel_initializer=self.kernel_initializer, activate=False)
        
        
    def call(self, input, training=False):
        branch = input

        x = self.conv1(input, training)
        x = self.conv2(x, training)
        if self.out==2:
            x2 = x
        x = self.conv3(x, training)

        if not self.idx:
            branch = self.branch_conv(branch, training)

        x = Add()([x, branch])
        x = LeakyReLU(alpha=0.1)(x)
        if self.out==2:
            return x, x2
        return x

    
class Upsample(Layer):
    def __init__(self, out_size, method='bilinear', **kwargs):
        super().__init__(**kwargs)
        self.upsample_layer = UpSampling2D(out_size, interpolation=method)
        
    def call(self, input):
        x = self.upsample_layer(input)
        return x
    
class TransposeConv(Layer):
    def __init__(self, units, kernel_size, strides, padding='same', dilation_rate=1, kernel_initializer='glorot', activate=True, bn=True, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.kernel_size = kernel_size
        self.kernel_initializer = kernel_initializer
        self.padding = padding
        self.dilation_rate = dilation_rate
        self.strides = strides
        self.activate = activate
        self.bn = bn
        
        self.transpose_conv = Conv2DTranspose(self.units, self.kernel_size, self.strides, self.padding, dilation_rate=self.dilation_rate,
                                              use_bias=not self.bn, kernel_initializer=self.kernel_initializer)
        self.batchN = BatchNormalization()
    def call(self, input, training=False):
        x = self.transpose_conv(input)
        if self.bn:
            x = self.batchN(x, training)
        if self.activate:
            x = LeakyReLU(alpha=0.1)(x)
        return x

class SConv(Layer):
    def __init__(self, units, kernel_size, strides=1, padding='same', dilation_rate=1, activate=True, bn=True, kernel_initializer=glorot, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.dilation_rate = dilation_rate
        self.activate = activate
        self.bn = bn
        self.kernel_initializer = kernel_initializer

        self.sep_conv = SeparableConv2D(self.units, self.kernel_size, self.strides, self.padding, dilation_rate=self.dilation_rate, use_bias=not bn,
                                        depthwise_initializer=self.kernel_initializer, pointwise_initializer=self.kernel_initializer)
        self.batchN = BatchNormalization()
    
    def call(self, input, training=False):
        x = self.sep_conv(input)
        if self.bn:
            x = self.batchN(x, training)
        if self.activate:
            x = LeakyReLU(alpha=0.1)(x)
        
        return x
    
class ASPP(Layer):
    def __init__(self, kernel_initializer=glorot, **kwargs):
        super().__init__(**kwargs)
        self.kernel_initializer = kernel_initializer

        self.conv1 = Conv(256, 1)

        self.out1 = Conv(256, 1, dilation_rate=1)
        self.out2 = Conv(256, 3, dilation_rate=6)
        self.out3 = Conv(256, 3, dilation_rate=12)
        self.out4 = Conv(256, 3, dilation_rate=18)

        self.conv2 = Conv(256, 1)

    def call(self, input, training=False):
        h, w = input.shape[1:3]
        out_pool = AveragePooling2D((h, w))(input)
        out_pool = self.conv1(out_pool, training)
        out_pool = Upsample([h, w])(out_pool)

        out1 = self.out1(input, training)
        out2 = self.out2(input, training)
        out3 = self.out3(input, training)
        out4 = self.out4(input, training)

        x = tf.concat([out1, out2, out3, out4, out_pool], -1)
        x = self.conv2(x, training)

        return x