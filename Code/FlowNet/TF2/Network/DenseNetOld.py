#!/usr/bin/env python

import tensorflow as tf
import sys
import numpy as np
import inspect
from functools import wraps
from tensorflow.contrib.framework import add_arg_scope
from tensorflow.contrib.framework import arg_scope
from ..Misc.Decorators import *

# TODO: Add training flag
    
class BaseLayers(object):
    def __init__(self):
        self.CurrBlock = 0
    # Decorator to count number of functions have been called
    # Ideas from
    # https://stackoverflow.com/questions/13852138/how-can-i-define-decorator-method-inside-class
    # https://stackoverflow.com/questions/41678265/how-to-increase-a-number-every-time-a-function-is-run

    @CountAndScope
    @add_arg_scope
    def ConvBNReLUBlock(self, inputs = None, filters = None, kernel_size = None, strides = None, padding = None):
        conv =  self.Conv(inputs, filters, kernel_size, strides, padding)
        bn = self.BN(conv)
        Output = self.ReLU(bn)
        return Output

    @CountAndScope
    @add_arg_scope
    def Conv(self, inputs = None, filters = None, kernel_size = None, strides = None, padding = None, activation=None):
        Output = tf.layers.conv2d(inputs = inputs, filters = filters, kernel_size = kernel_size,\
                                  strides = strides, padding = padding, activation=activation) 
        return Output

    @CountAndScope
    @add_arg_scope
    def BN(self, inputs = None):
        Output = tf.layers.batch_normalization(inputs = inputs) 
        return Output
    
    @CountAndScope
    @add_arg_scope
    def ReLU(self, inputs = None):
        Output = tf.nn.relu(inputs)
        return Output

    @CountAndScope
    @add_arg_scope
    def Concat(self, inputs = None, axis=0):
        Output = tf.concat(values = inputs, axis = axis)
        return Output

    @CountAndScope
    @add_arg_scope
    def Dense(self, inputs = None, filters = None, activation=None):
        Output = tf.layers.dense(inputs, units = filters, activation=activation)
        return Output

class DenseNet(BaseLayers):
    def __init__(self, ImageSize = None, NumOut = None, InputPH = None, Training = False):
        super(DenseNet, self).__init__()
        self.ImageSize = ImageSize
        self.NumOut = NumOut
        if(InputPH is None):
            print('ERROR: Input PlaceHolder cannot be empty!')
            sys.exit(0)
        self.InputPH = InputPH
        self.ExpansionFactor = 2.0 # Factor by which number of output neurons grow at every stage
        self.InitNeurons = 32
        self.NDenseBlockLayers = 6
        self.Net = self.InputPH
        self.Training = Training

    @CountAndScope
    @add_arg_scope
    def DenseBlock(self, inputs = None, filters = None, kernel_size = None, strides = None, padding = None, Bypass = False, NDenseBlockLayers = None):
        # expandfilter = int(4.0*filters)
        for count in range(NDenseBlockLayers):
        	if count == 0:
        		bottle = self.Bottleneck(inputs = inputs, filters = filters, kernel_size = (1,1), padding = padding, strides=(1,1), activation=tf.nn.relu)
        		concat = inputs
        	concat = self.Concat(inputs = [concat, bottle])
        	bottle = self.Bottleneck(inputs = concat, filters = filters, kernel_size = (1,1), padding = padding, strides=(1,1), activation=tf.nn.relu)
        # squeeze = self.Conv(inputs = inputs, filters = filters, kernel_size = (1,1), padding = padding, strides=(1,1), activation=tf.nn.relu, name='squeeze')
        # expand1x1 = self.Conv(inputs = squeeze, filters = expandfilter, kernel_size = (1,1), padding = padding, strides=(1,1), activation=tf.nn.relu, name='expand1x1')
        # expand3x3 = self.Conv(inputs = squeeze, filters = expandfilter, kernel_size = (3,3), padding = padding, strides=(1,1), activation=tf.nn.relu, name='expand3x3')
        # concat = self.Concat(inputs = [expand1x1, expand3x3])
        # if(Bypass):
            # concat = tf.math.add(inputs, concat, name='add')
        return concat


    @CountAndScope
    @add_arg_scope
    def TransitionLayer(self, inputs = None, filters = None, kernel_size = None, strides = None, padding = None, Bypass = False, NDenseBlockLayers = None):
        # expandfilter = int(4.0*filters)
        output = self.BN(inputs = inputs)
        output = self.ReLU(inputs = output)
        output = self.Conv(inputs = output)
        # TODO: Add Dropout here
        # TODO: Add Dropout here
        return concat

    @CountAndScope
    @add_arg_scope
    def Bottleneck(self, inputs = None, padding = None):

		output = self.BN(inputs = inputs)
		output = self.ReLU(inputs = output)
		output = self.Conv(inputs = output)
		# TODO: Add Dropout here
		output = self.BN(inputs = output)
		output = self.ReLU(inputs = output)
		output = self.Conv(inputs = output)
		# TODO: Add Dropout here
		return output

	@CountAndScope
    @add_arg_scope
    def OutputLayer(self, inputs = None, padding = None):
        conv = self.Conv(inputs = inputs, filters = self.NumOut, kernel_size = (1,1), strides = (1,1), padding = padding)
        dense = self.Dense(inputs = conv, filters = self.NumOut, activation=None)
        return dense

    def _arg_scope(self):
        with arg_scope([self.ConvBNReLUBlock, self.Conv, self.BN, self.ReLU, self.FireModule], kernel_size = (3,3), strides = (2,2), padding = 'same') as sc: 
            return sc
        
    def MakeNet(self):
        with arg_scope(self._arg_scope()):
            # Conv
            self.Net = self.Conv(inputs = self.Net, filters = self.InitNeurons, kernel_size = (7,7))
            # DenseBlock
            self.Net = self.DenseBlock(inputs = self.Net, filters = self.InitNeurons, kernel_size = (5,5))
            
            # Transition
            for count in range(3):
            	self.Net = self.Transition(inputs = self.Net, filters = self.InitNeurons, kernel_size = (3,3))
            	self.Net = self.DenseBlock(inputs = self.Net, filters = self.InitNeurons, kernel_size = (3,3))

            self.Net = self.BN(inputs = self.Net)
            self.Net = self.ReLU(inputs = self.Net)

            # TODO: Global Average Pooling
            # Flatten
            # FC
			self.Net = self.OutputLayer(inputs = self.Net, padding = 'same')
            
def main():
     # Test functionality of code
     InputPH = tf.placeholder(tf.float32, shape=(32, 100, 100, 3), name='Input')
     DN = DenseNet(InputPH = InputPH, NumOut = 10)
     print(SN.CurrBlock)
     Z = SN.MakeNet()
     print(SN.CurrBlock)
     Writer = tf.summary.FileWriter('/home/nitin/PRGEye/Logs3/', graph=tf.get_default_graph())        

if __name__=="__main__":
    main()
