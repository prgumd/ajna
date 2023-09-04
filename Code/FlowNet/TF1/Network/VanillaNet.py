#!/usr/bin/env python

# Use when all warps are of same type

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

class VanillaNet(BaseLayers):
    def __init__(self, InputPH = None, Padding = None,\
                 NumOut = None, InitNeurons = None, ExpansionFactor = None, NumSubBlocks = None, NumBlocks = None, Suffix = None, UncType = None):
        super(VanillaNet, self).__init__()
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
        self.NumOut = NumOut
        self.currBlock = 0


    @CountAndScope
    @add_arg_scope
    def VanillaNetBlock(self, inputs):
        # Encoder has (NumSubBlocks + 2) x (Conv + BN + ReLU) Layers
        # Conv
        NumFilters = self.InitNeurons
        Net = self.ConvBNReLUBlock(inputs = inputs, filters = NumFilters, kernel_size = (7,7))
        # Conv
        NumFilters = int(NumFilters*self.ExpansionFactor)
        Net = self.ConvBNReLUBlock(inputs = Net, filters = NumFilters, kernel_size = (5,5))
        # Conv
        for count in range(self.NumSubBlocks):
            NumFilters = int(NumFilters*self.ExpansionFactor)
            Net = self.ConvBNReLUBlock(inputs = Net, filters = NumFilters, kernel_size = (3,3))

        # Decoder has (NumSubBlocks + 2 or 3) x (ConvTranspose + BN + ReLU) Layers
        # ConvTranspose
        for count in range(self.NumSubBlocks):
            NumFilters = int(NumFilters/self.ExpansionFactor)
            Net = self.ConvTransposeBNReLUBlock(inputs = Net, filters = NumFilters, kernel_size = (3,3))
        
        # ConvTranspose
        NumFilters = int(NumFilters/self.ExpansionFactor)
        Net = self.ConvTransposeBNReLUBlock(inputs = Net, filters = NumFilters, kernel_size = (5,5))
        # ConvTranspose
        NumFilters = int(NumFilters/self.ExpansionFactor)
        Net = self.ConvTransposeBNReLUBlock(inputs = Net, filters = NumFilters, kernel_size = (7,7))        
        # ConvTranspose
        Net = self.ConvTransposeBNReLUBlock(inputs = Net, filters = self.NumOut, kernel_size = (7,7), strides = (1,1))
        return Net
        
    def _arg_scope(self):
        with arg_scope([self.ConvBNReLUBlock, self.ConvTransposeBNReLUBlock, self.Conv, self.ConvTranspose], kernel_size = (3,3), strides = (2,2), padding = self.Padding) as sc: 
            return sc
        
    def Network(self):
        with arg_scope(self._arg_scope()):
            OutNow = self.InputPH
            for count in range(self.NumBlocks):
                with tf.variable_scope('EncoderDecoderBlock' + str(count) + self.Suffix):
                    OutNow = self.VanillaNetBlock(OutNow) 

                    # Update counter used for looping over warpType
                    self.currBlock += 1
            
        return OutNow
