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
                with tf.variable_scope('ICTSNBlock' + str(count)):
                    # Warp Original Image based on previous composite warp parameters
                    # Compute current warp parameters
                    dpNow = self.MobileNetv1Block(self.InputPH,  filters = self.InitNeurons, NumOut = self.Opt.warpDim[count]) 
                    dpMtrxNow = warp2.vec2mtrx(self.Opt, dpNow)    
                    # pMtrxNow = warp2.compose(self.Opt, pMtrxNow, dpMtrxNow)
                    with tf.name_scope("compose"):
                        pMtrxNow = tf.matmul(dpMtrxNow,pMtrxNow)
                        # pMtrxNew = tf.divide(pMtrxNew, pMtrxNew[:,2:3,2:3])           

                    # Update counter used for looping over warpType
                    self.Opt.currBlock += 1

                    if(self.Opt.currBlock == self.Opt.NumBlocks):
                        # Decrement counter so you use last warp Type
                        self.Opt.currBlock -= 1
                        # pNow = dpNow
                        pNow = warp2.mtrx2vec(self.Opt, pMtrxNow) 

        return pNow

# def main():
#    tu.SetGPU(0)
#    # Test functionality of code
#    PatchSize = np.array([100, 100, 3])
#    MiniBatchSize = 1
#    InputPH = tf.placeholder(tf.float32, shape=(MiniBatchSize, PatchSize[0], PatchSize[1], PatchSize[2]), name='Input')
#    # Create network class variable
#    Opt =  warp2.Options(PatchSize= PatchSize, MiniBatchSize=MiniBatchSize, warpType = ['scale', 'scale', 'translation', 'translation']) # ICSTN Options
#    VN = VanillaNet(InputPH = InputPH, Training = True, Opt = Opt)
#    # Build the atual network
#    pMtrxNow, pNow, ImgWarp = VN.Network()
#    # Setup Saver
#    Saver = tf.train.Saver()
#    # This runs on 1 thread of CPU when tu.SetGPU(-1) is set
#    # config=tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1, allow_soft_placement=True)
#    # tf.Session(config=config)
#    with tf.Session() as sess:
#        # Initialize Weights
#        sess.run(tf.global_variables_initializer())
#        tu.FindNumFlops(sess, 1)
#        tu.FindNumParams(1)
#        tu.CalculateModelSize(1)
#        # Save model every epoch
#        SaveName = '/home/nitin/PRGEye/CheckPoints/SpeedTests/TestVanillaNet/model.ckpt'
#        Saver.save(sess, save_path=SaveName)
#        print(SaveName + ' Model Saved...') 
#        FeedDict = {VN.InputPH: np.random.rand(MiniBatchSize,PatchSize[0],PatchSize[1],PatchSize[2])}
#        for count in range(10):
#            Timer1 = mu.tic()
#            pMtrxNowVal, pNowVal, ImgWarpVal = sess.run([pMtrxNow, pNow, ImgWarp], feed_dict=FeedDict)
#            print(1/mu.toc(Timer1))
#    Writer = tf.summary.FileWriter('/home/nitin/PRGEye/Logs3/', graph=tf.get_default_graph())

# if __name__=="__main__":
#     main()
