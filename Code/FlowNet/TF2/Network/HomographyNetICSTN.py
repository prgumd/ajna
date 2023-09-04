import tensorflow as tf
import sys
import numpy as np
import Misc.warpICSTN as warp

# Don't generate pyc codes
sys.dont_write_bytecode = True

def ConvBlock(Input, Filters, KerSize, Strides, Padding, AppendNum):
    with tf.variable_scope(AppendNum):
        conv = tf.layers.conv2d(inputs = Input, filters = Filters, kernel_size = KerSize,\
                                strides = Strides, padding = Padding, activation=None, name='conv'+AppendNum)
        bn = tf.layers.batch_normalization(conv, name='bn'+AppendNum)
        bn = tf.nn.relu(bn, name='relu'+AppendNum)

    return bn

def ICSTNBlock(Img, ImageSize, MiniBatchSize, AppendNum=''):
    """
    Inputs: 
    Img is a MiniBatch of the current image
    ImageSize - Size of the Image
    Outputs:
    pr1 is the predicted output of homing vector for a MiniBatch
    """
    with tf.variable_scope(AppendNum):
        # Img is of size MxNx3
        conv1 = ConvBlock(Img, 8,  [7, 7], (2, 2), 'same', '1')
        conv2 = ConvBlock(conv1, 16, [5, 5], (2, 2), 'same', '2')
        conv3 = ConvBlock(conv2, 32, [3, 3], (2, 2), 'same', '3')
        conv4 = ConvBlock(conv3, 64, [3, 3], (2, 2), 'same', '4')

        # flat is of size BatchSize x M/16*N/16*64
        flat = tf.reshape(conv4, [-1, ImageSize[0]*ImageSize[1]*64/(16*16)], name='flat1'+AppendNum)

        # flatdrop is a dropout layer
        flatdrop = tf.layers.dropout(flat, rate=0.75, name='dropout1'+AppendNum)

        # fc1 
        fc1 = tf.layers.dense(flatdrop, units=128, activation=None, name='fc1'+AppendNum)
        
        fc1 = tf.nn.relu(flatdrop, name='relu'+AppendNum)

        # fc2
        fc2 = tf.layers.dense(fc1, units=8, activation=None, name='fc2'+AppendNum)

    return fc2

def ICSTN(Img, ImageSize, MiniBatchSize, opt):
    # ImgWarpAll = []

    for count in range(opt.NumBlocks):
        # print(count)
        if(count == 0):
            pNow = opt.pInit
            
        # Warp Image based on previous composite warp parameters
        pMtrxNow = warp.vec2mtrx(opt, pNow)
        ImgWarpNow = warp.transformImage(opt, Img, pMtrxNow)
        # ImgWarpAll.append(ImgWarpNow)

        # Compute current warp parameters
        dpNow = ICSTNBlock(Img, ImageSize, MiniBatchSize, AppendNum=str(count+1))
        pNow = warp.compose(opt, pNow, dpNow)

    pMtrx = warp.vec2mtrx(opt, pNow) # Final pMtrx
    ImgWarp = warp.transformImage(opt, Img, pMtrx) # Final Image Warp
    # ImgWarpAll.append(ImgWarp)

    return pMtrx, pNow, ImgWarp

        
        










