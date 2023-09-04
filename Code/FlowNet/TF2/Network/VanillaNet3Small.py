#!/usr/bin/env python

# For [Translation, Translation, Scale, Scale]

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
    def __init__(self, InputPH = None, Training = False,  Padding = None,\
                 Opt = None, InitNeurons = None, ExpansionFactor = None, NumBlocks = None):
        super(VanillaNet, self).__init__()
        if(InputPH is None):
            print('ERROR: Input PlaceHolder cannot be empty!')
            sys.exit(0)
        if(Opt is None):
            print('ERROR: Options cannot be empty!')
            sys.exit(0)
        self.InputPH = InputPH
        self.Training = Training
        if(InitNeurons is None):
            InitNeurons = 18
        if(ExpansionFactor is None):
            ExpansionFactor =  2.0
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
    def VanillaNetBlock(self, inputs = None, filters = None, NumOut = None, ExpansionFactor = None):
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
            Net = self.ConvBNReLUBlock(inputs = Net, filters = NumFilters, kernel_size = (3,3))
      
        # Output
        Net = self.OutputLayer(inputs = Net, rate=self.DropOutRate, NumOut = NumOut)
        return Net
        
    def _arg_scope(self):
        with arg_scope([self.ConvBNReLUBlock, self.Conv], kernel_size = (3,3), strides = (2,2), padding = self.Padding) as sc: 
            return sc
        
    def Network(self):
        with arg_scope(self._arg_scope()):
            # MODIFY THIS DEPENDING ON ARCH!
            # pRet = tf.zeros([32, 3])
            for count in range(self.Opt.NumBlocks):
                if(count == 0):
                    pNow = self.Opt.pInit
                    pMtrxNow = warp2.vec2mtrx(self.Opt, pNow)
                with tf.variable_scope('ICTSNBlock' + str(count)):
                    # Warp Original Image based on previous composite warp parameters
                    if(self.Training):
                        ImgWarpNow = warp2.transformImage(self.Opt, self.InputPH, pMtrxNow)

                    # Compute current warp parameters
                    dpNow = self.VanillaNetBlock(self.InputPH,  filters = self.InitNeurons, NumOut = self.Opt.warpDim[count]) 
                    dpMtrxNow = warp2.vec2mtrx(self.Opt, dpNow)    
                    pMtrxNow = warp2.compose(self.Opt, pMtrxNow, dpMtrxNow)

                    # MODIFY THIS DEPENDING ON ARCH!
                    if(count == 1):
                        # Numpy like indexing directly doesn't work on Tensors
                        # https://stackoverflow.com/questions/37670886/how-do-i-select-certain-columns-of-a-2d-tensor-in-tensorflow
                        # print(self.Opt.warpDim[self.Opt.currBlock])
                        pReta =  tf.transpose(tf.nn.embedding_lookup(tf.transpose(warp2.mtrx2vec(self.Opt, pMtrxNow)), [0,1]))
                    
                    # Update counter used for looping over warpType
                    self.Opt.currBlock += 1


                    if(self.Opt.currBlock == self.Opt.NumBlocks):
                        # Decrement counter so you use last warp Type
                        self.Opt.currBlock -= 1
                        pNow = warp2.mtrx2vec(self.Opt, pMtrxNow)
                        # MODIFY THIS DEPENDING ON ARCH!
                        pRetb = tf.expand_dims(tf.transpose(tf.nn.embedding_lookup(tf.transpose(warp2.mtrx2vec(self.Opt, pMtrxNow)), 0)), axis=1)
                        pRet = tf.concat([pRetb, pReta], axis=1)
                        if(self.Training):
                            ImgWarp = warp2.transformImage(self.Opt, self.InputPH, pMtrxNow) # Final Image Warp
                        else:
                            ImgWarp = None
            
        return pMtrxNow, pRet, ImgWarp

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
