#!/usr/bin/env python

import tensorflow as tf
import sys
import numpy as np
import inspect
from functools import wraps
from tensorflow.contrib.framework import add_arg_scope
from tensorflow.contrib.framework import arg_scope
import Misc.TFUtils as tu
from Misc.Decorators import *
import Misc.warpICSTN2 as warp2
from Network.BaseLayers import *
import Misc.MiscUtils as mu

# TODO: Add training flag

class ShuffleNet(BaseLayers):
    def __init__(self, InputPH = None, Training = False,  Padding = None,\
                 Opt = None, InitNeurons = None, ExpansionFactor = None, NumBlocks = None, Suffix = None):
        super(ShuffleNet, self).__init__()
        if(InputPH is None):
            print('ERROR: Input PlaceHolder cannot be empty!')
            sys.exit(0)
        if(Opt is None):
            print('ERROR: Options cannot be empty!')
            sys.exit(0)
        self.InputPH = InputPH
        self.Training = Training
        if(InitNeurons is None):
            InitNeurons = 16
        if(ExpansionFactor is None):
            ExpansionFactor =  1.6
        if(NumBlocks is None):
            NumBlocks = 2
        self.InitNeurons = InitNeurons
        self.ExpansionFactor = ExpansionFactor
        self.DropOutRate = 0.7
        self.NumBlocks = NumBlocks
        if(Padding is None):
            Padding = 'same'
        self.Padding = Padding
        self.Opt = Opt
        if(Suffix is None):
            Suffix = ''
        self.Suffix = Suffix


    @CountAndScope
    @add_arg_scope
    def DepthwiseConvBN(self, inputs = None, filters = None, kernel_size = None, strides = None, padding = None):
        conv = tf.layers.separable_conv2d(inputs = inputs, filters = filters, kernel_size = kernel_size, strides = (1,1), padding = padding, dilation_rate  = (1,1), activation=None)
        bn = self.BN(conv)
        return bn

    @CountAndScope
    @add_arg_scope
    def Shuffle(self, inputs = None, filters = None, kernel_size = None, strides = None, padding = None, groups = 2):
        # Taken from https://github.com/timctho/shufflenet-v2-tensorflow/blob/ae091dfbf10e5bf0fb723e00ebbf5410b550f4f8/module.py
        n, h, w, c = inputs.get_shape().as_list()
        Output = tf.reshape(inputs, shape=tf.convert_to_tensor([tf.shape(inputs)[0], h, w, groups, c // groups]))
        Output = tf.transpose(Output, tf.convert_to_tensor([0, 1, 2, 4, 3]))
        Output = tf.reshape(Output, shape=tf.convert_to_tensor([tf.shape(Output)[0], h, w, c]))
        return Output

    @CountAndScope
    @add_arg_scope
    def OutputLayer(self, inputs = None, padding = None, rate=None, NumOut=None):
        if(rate is None):
            rate = 0.5
        if(NumOut is None):
           NumOut = self.NumOut     
        flat = self.Flatten(inputs = inputs)
        drop = self.Dropout(inputs = flat, rate=rate)
        dense = self.Dense(inputs = drop, filters = NumOut, activation=None)
        return dense

    @CountAndScope
    @add_arg_scope
    def ShuffleNetv2Block(self, inputs = None, filters = None, NumOut = None, ExpansionFactor = None):
        if(ExpansionFactor is None):
            ExpansionFactor = self.ExpansionFactor
        # Conv
        Net = self.ConvBNReLUBlock(inputs = inputs, filters = filters, kernel_size = (7,7))
        # Conv
        NumFilters = int(filters*ExpansionFactor)
        Net = self.ConvBNReLUBlock(inputs = Net, filters = NumFilters, kernel_size = (5,5))
        # Conv
        for count in range(self.NumBlocks):
            NumFilters = int(NumFilters*ExpansionFactor)
            HalfNumFilters = NumFilters // 2 # Will conact two HalfNumFilters twice to get NumFilters
            # Right Branch
            OutRight = self.ConvBNReLUBlock(inputs = Net, filters = HalfNumFilters, kernel_size = (1,1), strides = (1,1))
            OutRight = self.DepthwiseConvBN(inputs = OutRight, filters = HalfNumFilters)
            OutRight = self.ConvBNReLUBlock(inputs = OutRight, filters = HalfNumFilters, kernel_size = (1,1), strides = (1,1))
            # Left Branch
            OutLeft = self.DepthwiseConvBN(inputs = Net, filters = HalfNumFilters)
            OutLeft = self.ConvBNReLUBlock(inputs = OutLeft, filters = HalfNumFilters, kernel_size = (1,1), strides = (1,1))
            # Conact
            Out = self.Concat([OutRight, OutLeft], axis=3)
            # Channel Shuffle
            Net = self.Shuffle(Out)
      
        # Output
        Net = self.OutputLayer(inputs = Net, rate=self.DropOutRate, NumOut = NumOut)
        return Net
        
    def _arg_scope(self):
        with arg_scope([self.DepthwiseConvBN, self.ConvBNReLUBlock, self.Conv], kernel_size = (3,3), strides = (2,2), padding = self.Padding) as sc: 
            return sc
        
    def Network(self):
        with arg_scope(self._arg_scope()):
            for count in range(self.Opt.NumBlocks):
                if(count == 0):
                    pNow = self.Opt.pInit
                    pMtrxNow = warp2.vec2mtrx(self.Opt, pNow)
                with tf.variable_scope('ICTSNBlock' + str(count) + self.Suffix):
                    # Warp Original Image based on previous composite warp parameters
                    if(self.Training):
                        ImgWarpNow = warp2.transformImage(self.Opt, self.InputPH, pMtrxNow)

                    # Compute current warp parameters
                    dpNow = self.ShuffleNetv2Block(self.InputPH,  filters = self.InitNeurons, NumOut = self.Opt.warpDim[count]) 
                    dpMtrxNow = warp2.vec2mtrx(self.Opt, dpNow)    
                    pMtrxNow = warp2.compose(self.Opt, pMtrxNow, dpMtrxNow) 

                    # Update counter used for looping over warpType
                    self.Opt.currBlock += 1

                    if(self.Opt.currBlock == self.Opt.NumBlocks):
                        # Decrement counter so you use last warp Type
                        self.Opt.currBlock -= 1
                        pNow = warp2.mtrx2vec(self.Opt, pMtrxNow) 
                        if(self.Training):
                            ImgWarp = warp2.transformImage(self.Opt, self.InputPH, pMtrxNow) # Final Image Warp
                        else:
                            ImgWarp = None
            
        return pMtrxNow, pNow, ImgWarp
