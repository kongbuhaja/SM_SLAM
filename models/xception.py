from tensorflow.keras.layers import Layer, Add
from tensorflow.keras.initializers import GlorotUniform as glorot
from models.common import *

class Xception(Layer):
    def __init__(self, kernel_initializer=glorot, **kwargs):
        super().__init__(**kwargs)
        self.kernel_initializer = kernel_initializer

        self.entry_flow = EntryFlow(self.kernel_initializer)
        self.middle_flow = MiddleFlow(self.kernel_initializer)
        self.exit_flow = ExitFlow(self.kernel_initializer)
    
    def call(self, input, training=False):
        x, branch1, branch2 = self.entry_flow(input, training)
        x = self.middle_flow(x, training)
        x = self.exit_flow(x, training)

        return x, branch1, branch2

class EntryFlow(Layer):
    def __init__(self, kernel_initializer=glorot, **kwargs):
        super().__init__(**kwargs)
        self.kernel_initializer = kernel_initializer

        self.conv1 = Conv(32, 3, 2, kernel_initializer=self.kernel_initializer)
        self.conv2 = Conv(64, 3, kernel_initializer=self.kernel_initializer)
        
        self.sep_conv1 = SConv(128, 3, kernel_initializer=self.kernel_initializer)
        self.sep_conv2 = SConv(128, 3, kernel_initializer=self.kernel_initializer)
        self.sep_conv3 = SConv(128, 3, 2, dilation_rate=2, kernel_initializer=self.kernel_initializer)
        self.branch_conv1 = Conv(128, 1, 2, kernel_initializer=self.kernel_initializer)

        self.sep_conv4 = SConv(256, 3, kernel_initializer=self.kernel_initializer)
        self.sep_conv5 = SConv(256, 3, kernel_initializer=self.kernel_initializer)
        self.sep_conv6 = SConv(256, 3, 2, dilation_rate=2, kernel_initializer=self.kernel_initializer)
        self.branch_conv2 = Conv(256, 1, 2, kernel_initializer=self.kernel_initializer)

        self.sep_conv7 = SConv(728, 3, kernel_initializer=self.kernel_initializer)
        self.sep_conv8 = SConv(728, 3, kernel_initializer=self.kernel_initializer)
        self.sep_conv9 = SConv(728, 3, 2, dilation_rate=2, kernel_initializer=self.kernel_initializer)
        self.branch_conv3 = Conv(728, 1, 2, kernel_initializer=self.kernel_initializer)
    
    def call(self, input, training=False):
        x = self.conv1(input, training) # 512, 256
        x = self.conv2(x, training) 
        branch_ = x

        x = self.sep_conv1(x, training)
        x = self.sep_conv2(x, training)
        x = self.sep_conv3(x, training) # 256, 128
        branch_ = self.branch_conv1(branch_, training)
        x = Add()([x, branch_])
        branch_ = x
        branch1 = x

        x = self.sep_conv4(x, training)
        x = self.sep_conv5(x, training)
        x = self.sep_conv6(x, training)
        branch_ = self.branch_conv2(branch_, training)
        x = Add()([x, branch_])
        branch_ = x
        branch2 = x

        x = self.sep_conv7(x, training)
        x = self.sep_conv8(x, training)
        x = self.sep_conv9(x, training)
        branch_ = self.branch_conv3(branch_, training)
        x = Add()([x, branch_])

        return x, branch1, branch2
    
class MiddleFlow(Layer):
    def __init__(self, kernel_initializer=glorot, **kwargs):
        super().__init__(**kwargs)
        self.kernel_initializer = kernel_initializer

        self.sep_conv = [[SConv(728, 3, kernel_initializer=self.kernel_initializer) for j in range(3)] for i in range(16)]
    
    def call(self, input, training=False):
        x= input
        for i in range(len(self.sep_conv)):
            branch = x
            for j in range(len(self.sep_conv[i])):
                x = self.sep_conv[i][j](x, training)
            x = Add()([x, branch])
        return x
    
class ExitFlow(Layer):
    def __init__(self, kernel_initializer=glorot, **kwargs):
        super().__init__(**kwargs)
        self.kernel_initializer = kernel_initializer

        self.sep_conv1 = SConv(728, 3, kernel_initializer=self.kernel_initializer)
        self.sep_conv2 = SConv(1024, 3, kernel_initializer=self.kernel_initializer)
        self.sep_conv3 = SConv(1024, 3, 2, dilation_rate=2, kernel_initializer=self.kernel_initializer)
        self.branch_conv = Conv(1024, 1, 2, kernel_initializer=self.kernel_initializer)

        self.sep_conv4 = SConv(1536, 3, kernel_initializer=self.kernel_initializer)
        self.sep_conv5 = SConv(1536, 3, dilation_rate=2, kernel_initializer=self.kernel_initializer)
        self.sep_conv6 = SConv(2048, 3, kernel_initializer=self.kernel_initializer)
    
    def call(self, input, training=False):
        branch = input
        
        x = self.sep_conv1(input, training)
        x = self.sep_conv2(x, training)
        x = self.sep_conv3(x, training)
        branch = self.branch_conv(branch, training)
        x = Add()([x, branch])

        x = self.sep_conv4(x, training)
        x = self.sep_conv5(x, training)
        x = Upsample(2)(x)
        x = self.sep_conv6(x, training)
        
        return x
