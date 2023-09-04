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

class MobileNet(BaseLayers):
    def __init__(self, InputPH = None, Training = False,  Padding = None,\
                 Opt = None, InitNeurons = None, ExpansionFactor = None, NumBlocks = None, Suffix = None):
        super(MobileNet, self).__init__()
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
            ExpansionFactor =  2.0
        if(NumBlocks is None):
            NumBlocks = 3
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
        if(Suffix is None):
            Suffix = ''
        self.Suffix = Suffix

    @CountAndScope
    @add_arg_scope
    def DepthwiseConvBNReLU(self, inputs = None, filters = None, kernel_size = None, strides = None, padding = None):
        conv = tf.layers.separable_conv2d(inputs = inputs, filters = filters, kernel_size = kernel_size, strides = (1,1), padding = padding, dilation_rate  = (1,1), activation=None)
        bn = self.BN(conv)
        Output = self.ReLU(bn)
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
    def MobileNetv1Block(self, inputs = None, filters = None, NumOut = None, ExpansionFactor = None):
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
            Net = self.DepthwiseConvBNReLU(inputs = Net, filters = NumFilters)
            Net = self.ConvBNReLUBlock(inputs = Net, filters = NumFilters)
             
        # Output
        Net = self.OutputLayer(inputs = Net, rate=self.DropOutRate, NumOut = NumOut)
        return Net
        
    def _arg_scope(self):
        with arg_scope([self.DepthwiseConvBNReLU, self.ConvBNReLUBlock, self.Conv], kernel_size = (3,3), strides = (2,2), padding = self.Padding) as sc: 
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
                    dpNow = self.MobileNetv1Block(self.InputPH,  filters = self.InitNeurons, NumOut = self.Opt.warpDim[count]) 
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


