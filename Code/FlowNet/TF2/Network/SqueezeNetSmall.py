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

class SqueezeNet(BaseLayers):
    def __init__(self, InputPH = None, Training = False,  Padding = None, Opt = None, InitNeurons = None, NumFire = None, NumBlocks = None, Suffix = None):
        super(SqueezeNet, self).__init__()
        if(InputPH is None):
            print('ERROR: Input PlaceHolder cannot be empty!')
            sys.exit(0)
        if( Opt is None):
            print('ERROR: Options cannot be empty!')
            sys.exit(0)
        self.InputPH = InputPH
        if(InitNeurons is None):
          InitNeurons = 4
        self.InitNeurons = InitNeurons
        self.Training = Training
        self.ExpansionFactor = 1.15
        self.DropOutRate = 0.7
        if(Padding is None):
            Padding = 'same'
        self.Padding = Padding
        self.Opt = Opt
        if(NumFire is None):
            NumFire =  1
        if(NumBlocks is None):
            NumBlocks = 1
        self.NumBlocks = NumBlocks
        self.NumFire = NumFire
        if(Suffix is None):
            Suffix = ''
        self.Suffix = Suffix


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
    def FireModule(self, inputs = None, filters = None, kernel_size = None, strides = None, padding = None, Bypass = False):
        expandfilter = int(4.0*filters)
        squeeze = self.Conv(inputs = inputs, filters = filters, kernel_size = (1,1), padding = padding, strides=(1,1), activation=tf.nn.relu, name='squeeze')
        expand1x1 = self.Conv(inputs = squeeze, filters = expandfilter, kernel_size = (1,1), padding = padding, strides=(1,1), activation=tf.nn.relu, name='expand1x1')
        expand3x3 = self.Conv(inputs = squeeze, filters = expandfilter, kernel_size = (3,3), padding = padding, strides=(1,1), activation=tf.nn.relu, name='expand3x3')
        concat = self.Concat(inputs = [expand1x1, expand3x3], axis=1)
        if(Bypass):
            concat = tf.math.add(inputs, concat, name='add')
        return concat

    @CountAndScope
    @add_arg_scope
    def FireConvBlock(self, inputs = None, filters = None, kernel_size = None, strides = None, padding = None, Bypass = False, NumFire = None):
        Net = inputs
        for count in range(NumFire): 
            Net = self.FireModule(inputs = Net, filters = filters, kernel_size = kernel_size, strides = strides, padding = padding, Bypass = Bypass)
        Net = self.Conv(inputs = Net, filters = filters, kernel_size = (1,1), padding = padding, strides=(1,1), activation=tf.nn.relu)
        return Net

    @CountAndScope
    @add_arg_scope
    def SqueezeNetBlock(self, inputs = None, filters = None, NumOut = None):
        # Conv
        Net = self.ConvBNReLUBlock(inputs = inputs, padding = self.Padding, filters = filters, kernel_size = (7,7))
        
        # Conv
        NumFilters = int(filters*self.ExpansionFactor)
        Net = self.ConvBNReLUBlock(inputs = Net, padding = self.Padding, filters = NumFilters, kernel_size = (5,5))

        # 3 x FireConv blocks
        for count in range(self.NumBlocks):
            NumFilters = int(NumFilters*self.ExpansionFactor)
            Net = self.FireConvBlock(inputs = Net, filters = NumFilters, Bypass = False, NumFire = self.NumFire)

        # TODO: Global Avg. Pool
        # Output
        Net = self.OutputLayer(inputs = Net, padding = self.Padding, rate=self.DropOutRate, NumOut = NumOut)
        return Net
        
    def _arg_scope(self):
        with arg_scope([self.Conv, self.ConvBNReLUBlock, self.FireConvBlock], kernel_size = (3,3), strides = (2,2), padding = self.Padding) as sc: 
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
                    dpNow = self.SqueezeNetBlock(self.InputPH,  filters = self.InitNeurons, NumOut = self.Opt.warpDim[count])

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
