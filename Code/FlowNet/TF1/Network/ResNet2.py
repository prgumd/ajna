#!/usr/bin/env python

import tensorflow as tf
import sys
import numpy as np
import inspect
from functools import wraps
from tensorflow.contrib.framework import add_arg_scope
from tensorflow.contrib.framework import arg_scope
# Required to import ..Misc so you don't have to run as package with -m flag
import Misc.TFUtils as tu
from Misc.Decorators import *
import Misc.warpICSTN2 as warp2
from Network.BaseLayers import *
import Misc.MiscUtils as mu

# TODO: Add training flag

class ResNet(BaseLayers):
    # http://torch.ch/blog/2016/02/04/resnets.html
    def __init__(self, InputPH = None, Padding = None,\
                 NumOut = None, InitNeurons = None, ExpansionFactor = None, NumSubBlocks = None, NumBlocks = None, Suffix = None, UncType = None):
        super(ResNet, self).__init__()
        if(InputPH is None):
            print('ERROR: Input PlaceHolder cannot be empty!')
            sys.exit(0)
        self.InputPH = InputPH
        if(InitNeurons is None):
            InitNeurons = 37
        if(ExpansionFactor is None):
            ExpansionFactor =  2.0
        if(NumSubBlocks is None):
            NumSubBlocks = 2
        if(NumBlocks is None):
            NumBlocks = 1
        self.InitNeurons = InitNeurons
        self.ExpansionFactor = ExpansionFactor
        self.DropOutRate = 0.7
        self.NumSubBlocks = NumSubBlocks
        self.NumBlocks = NumBlocks
        if(Padding is None):
            Padding = 'same'
        self.Padding = Padding
        if(Suffix is None):
            Suffix = ''
        self.Suffix = Suffix
        if(NumOut is None):
            NumOut = 1
        self.NumOut = 1
        self.currBlock = 0
        self.UncType = UncType
        if(UncType == 'Aleatoric'):
            # Each channel with also have a variance associated with it
            self.NumOut *= 2

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
    def ResBlock(self, inputs = None, filters = None, kernel_size = None, strides = None, padding = None):
        Net = self.ConvBNReLUBlock(inputs = inputs, filters = filters, padding = padding, strides=(1,1))
        Net = self.Conv(inputs = Net, filters = filters, padding = padding, strides=(1,1), activation=None)
        Net = self.BN(inputs = Net)
        Net = tf.add(Net, inputs)
        Net = self.ReLU(inputs = Net)
        return Net

    @CountAndScope
    @add_arg_scope
    def ResBlockTranspose(self, inputs = None, filters = None, kernel_size = None, strides = None, padding = None):
        Net = self.ConvTransposeBNReLUBlock(inputs = inputs, filters = filters, padding = padding, strides=(1,1))
        Net = self.ConvTranspose(inputs = Net, filters = filters, padding = padding, strides=(1,1), activation=None)
        Net = self.BN(inputs = Net)
        Net = tf.add(Net, inputs)
        Net = self.ReLU(inputs = Net)
        return Net

    @CountAndScope
    @add_arg_scope
    def ResNetBlock(self, inputs):
        # Encoder has (NumSubBlocks + 2) x (Conv + BN + ReLU) Layers
        # Conv
        NumFilters = self.InitNeurons
        Net = self.ConvBNReLUBlock(inputs = inputs, filters = NumFilters, kernel_size = (7,7))
        # Conv
        NumFilters = int(NumFilters*self.ExpansionFactor)
        Net = self.ConvBNReLUBlock(inputs = Net, filters = NumFilters, kernel_size = (5,5))
        # Conv
        for count in range(self.NumSubBlocks):
            Net = self.ResBlock(inputs = Net, filters = NumFilters)
            NumFilters = int(NumFilters*self.ExpansionFactor)
            # Extra Conv for downscaling
            Net = self.Conv(inputs = Net, filters = NumFilters, padding = self.Padding)


        # 2 Decoder Branches, 1 predicts Disparity and another predicts log(variance)
        # Decoder has (NumSubBlocks + 2 or 3) x (ConvTranspose + BN + ReLU) Layers
        # ConvTranspose
        # Branch 1 for Disparity
        Net1 = Net
        NumFilters1 = NumFilters
        for count in range(self.NumSubBlocks):
            Net1 = self.ResBlockTranspose(inputs = Net1, filters = NumFilters1)
            NumFilters1 = int(NumFilters1/self.ExpansionFactor)
            # Extra ConvTranspose for upscaling
            Net1 = self.ConvTranspose(inputs = Net1, filters = NumFilters1, padding = self.Padding)
        
        # ConvTranspose
        NumFilters1 = int(NumFilters1/self.ExpansionFactor)
        Net1 = self.ConvTransposeBNReLUBlock(inputs = Net1, filters = NumFilters1, kernel_size = (5,5))
        # ConvTranspose
        NumFilters1 = int(NumFilters1/self.ExpansionFactor)
        Net1 = self.ConvTransposeBNReLUBlock(inputs = Net1, filters = NumFilters1, kernel_size = (7,7))
        # ConvTranspose
        Net1 = self.ConvTranspose(inputs = Net1, filters = 1, kernel_size = (7,7), strides = (1,1), activation=None)


        # Branch 2 for log(variance)
        Net2 = Net
        NumFilters2 = NumFilters
        for count in range(self.NumSubBlocks):
            Net2 = self.ResBlockTranspose(inputs = Net2, filters = NumFilters2)    
            NumFilters2 = int(NumFilters2/self.ExpansionFactor)
            # Extra ConvTranspose for upscaling
            Net2 = self.ConvTranspose(inputs = Net2, filters = NumFilters2, padding = self.Padding)
        
        # ConvTranspose
        NumFilters2 = int(NumFilters2/self.ExpansionFactor)
        Net2 = self.ConvTransposeBNReLUBlock(inputs = Net2, filters = NumFilters2, kernel_size = (5,5))
        # ConvTranspose
        NumFilters2 = int(NumFilters1/self.ExpansionFactor)
        Net2 = self.ConvTransposeBNReLUBlock(inputs = Net2, filters = NumFilters2, kernel_size = (7,7))        
        # ConvTranspose
        Net2 = self.ConvTranspose(inputs = Net2, filters = 1, kernel_size = (7,7), strides = (1,1), activation=None)

        Net = tf.concat([Net1, Net2], axis=3)        
        return Net


    def _arg_scope(self):
        with arg_scope([self.ConvBNReLUBlock, self.ConvTransposeBNReLUBlock, self.Conv, self.ConvTranspose, self.ResBlock, self.ResBlockTranspose], kernel_size = (3,3), strides = (2,2), padding = self.Padding) as sc:
            return sc
        
    def Network(self):
        with arg_scope(self._arg_scope()):
            OutNow = self.InputPH
            for count in range(self.NumBlocks):
                with tf.variable_scope('EncoderDecoderBlock' + str(count) + self.Suffix):
                    OutNow = self.ResNetBlock(OutNow) 

                    # Update counter used for looping over warpType
                    self.currBlock += 1
        return OutNow
