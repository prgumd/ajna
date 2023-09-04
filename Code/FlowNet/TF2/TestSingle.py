#!/usr/bin/env python3

# Dependencies:
# opencv, do (pip install opencv-python)
# skimage, do (sudo)
# termcolor, do (pip install termcolor)
# tqdm, do (pip install tqdm)


# TODO: Adapt more augmentation from: https://github.com/sthalles/deeplab_v3/blob/master/preprocessing/inception_preprocessing.py

import tensorflow as tf
import cv2
import sys
import os
import glob
import re
import Misc.ImageUtils as iu
import random
from skimage import data, exposure, img_as_float
import matplotlib.pyplot as plt
from Misc.MiscUtils import *
import numpy as np
import time
import argparse
import shutil
import string
from termcolor import colored, cprint
import math as m
from tqdm import tqdm
import Misc.STNUtils as stn
import Misc.TFUtils as tu
from Misc.DataHandling import *
from Misc.BatchCreationTestSingleTF import *
from Misc.Decorators import *
from Misc.FlowVisUtilsNP import *
# Import of network is done in main code
import importlib
from datetime import datetime
import getpass
import copy
import platform

# Don't generate pyc codes
sys.dont_write_bytecode = True

# Disable Eager Execution to use TF1 code in TF2
tf.compat.v1.disable_eager_execution()

def PrettyPrint(Args, NumParams, NumFlops, ModelSize, VN):
    # TODO: Write to file?
    Username = getpass.getuser()
    cprint('Running on {}'.format(Username), 'yellow')
    cprint('Network Statistics', 'yellow')
    cprint('Network Used: {}'.format(Args.NetworkName), 'yellow')
    cprint('GPU Used: {}'.format(Args.GPUDevice), 'yellow')
    cprint('Init Neurons {}, Expansion Factor {}, NumBlocks {}, NumSubBlocks {}, DropOutFactor {}'.format(VN.InitNeurons, VN.ExpansionFactor,\
                                                                                                          VN.NumBlocks, VN.NumSubBlocks, VN.DropOutRate), 'yellow')
    cprint('Num Params: {}'.format(NumParams), 'green')
    cprint('Num FLOPs: {}'.format(NumFlops), 'green')
    cprint('Estimated Model Size (MB): {}'.format(ModelSize), 'green')
    cprint('Augmentations Used: {}'.format(Args.Augmentations), 'green')
    cprint('Model loaded from: {}'.format(Args.CheckPointPath), 'red')
        
def TestOperation(InputPH, I1PH, I2PH, Label1PH, Label2PH, Args):
    """
    Inputs: 
    ImgPH is the Input Image placeholder
    HomingVecPH is the ground truth  homing vector placeholder
    DirNames - Full path to all image files without extension
    Train/Val - Idxs of all the images to be used for training/validation (held-out testing in this case)
    Train/ValLabels - Labels corresponding to Train/Val
    NumTrain/ValSamples - length(Train/Val)
    ImageSize - Size of the image
    NumEpochs - Number of passes through the Train data
    MiniBatchSize is the size of the MiniBatch
    OptimizerParams - List of all OptimizerParams: depends on Optimizer
    SaveCheckPoint - Save checkpoint every SaveCheckPoint iteration in every epoch, checkpoint saved automatically after every epoch
    CheckPointPath - Path to save checkpoints/model
    NumTestRunsPerEpoch - Number of passes of Val data with MiniBatchSize 
    DivTrain - Divide the data by this number for Epoch calculation, use if you have a lot of data
    LatestFile - Latest checkpointfile to continue training
    Outputs:
    Saves Trained network in CheckPointPath
    """
    # Create Network Object with required parameters
    ClassName = Args.NetworkName.replace('Network.', '').split('Net')[0]+'Net'
    Network = getattr(Args.Net, ClassName)
    VN = Network(InputPH = InputPH, InitNeurons = Args.InitNeurons, Suffix = Args.Suffix, NumOut = Args.NumOut, UncType = Args.UncType)

    # Predict output with forward pass
    # WarpI1Patch contains warp of both I1 and I2, extract first three channels for useful data
    prVal = VN.Network()
 
    # Setup Saver
    Saver = tf.compat.v1.train.Saver()

    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        Saver.restore(sess, Args.CheckPointPath)
        # Extract only numbers from the name
        print('Loaded checkpoints ....')

        # Create Batch Generator Object
        bg = BatchGeneration(sess, I1PH)

        # Create Data Augmentation Object
        if(Args.DataAug):
            Args.Augmentations =  ['Brightness', 'Contrast', 'Hue', 'Saturation', 'Gamma', 'Gaussian']
            da = iu.DataAugmentationTF(sess, I1PH, Augmentations = Args.Augmentations)
            DataAugGen = da.RandPerturbBatch()
        else:
            Args.Augmentations = 'None'
            DataAugGen = None
            da = None

        # Print out Number of parameters
        NumParams = tu.FindNumParams(1)
        # Print out Number of Flops
        NumFlops = tu.FindNumFlops(sess, 1)
        # Print out Expected Model Size
        ModelSize = tu.CalculateModelSize(1)

        # Pretty Print Stats
        PrettyPrint(Args, NumParams, NumFlops, ModelSize, VN)

        # Predictions File
        ModelName = Args.CheckPointPath.split('/')[-1]

        IBatch, I1Batch, I2Batch, P1Batch, P2Batch, Label1Batch = bg.GenerateBatchTF(Args.Img1, Args.Img2, Args.GT1, Args, da, DataAugGen)

        FeedDict = {VN.InputPH: IBatch, I1PH: P1Batch, I2PH: P2Batch, Label1PH: Label1Batch}
        prValRet = sess.run([prVal], feed_dict=FeedDict)

        A = np.squeeze(Label1Batch[0])
        if(Args.UncType == 'Aleatoric' or Args.UncType == 'Inlier' or Args.UncType == 'LinearSoftplus'):
            B = np.squeeze(prValRet[0])[:,:,0:2]
            BUnc = np.squeeze(prValRet[0])[:,:,2:]
        else:
            B = np.squeeze(prValRet[0])

        try:
            Error = np.abs(A-B)
            MeanErrorPx = np.mean(Error)
            print('Mean Error (Px.): {}'.format(MeanErrorPx))
        except:
            pass
        
        ADisp = flow_viz_np(A[:,:,0], A[:,:,1])
        BDisp = flow_viz_np(B[:,:,0], B[:,:,1])

        P1Disp = np.uint8(iu.remap(P1Batch[0], 0., 255.))
        P2Disp = np.uint8(iu.remap(P2Batch[0], 0., 255.))
        cv2.imshow('Img1, Img2', np.hstack((P1Disp, P2Disp)))

        cv2.imshow('GT, Pred', np.hstack((ADisp, BDisp)))

        ColorWheel = draw_color_wheel_np(Args.PatchSize[0], Args.PatchSize[1])
        cv2.imshow('Color Wheel', ColorWheel)
        
        cv2.waitKey(0)

        if(Args.UncType == 'Aleatoric' or Args.UncType == 'Inlier' or Args.UncType == 'LinearSoftplus'):
            BUncDisp = np.uint8(iu.remap(BUnc, 0., 255.))
            FlowMag = iu.remap(np.sqrt(B[:,:,0]**2 + B[:,:,1]**2), 0., 255.)

            UncNormFlow = np.divide(np.float32(BUncDisp), np.tile(FlowMag[:,:,np.newaxis], (1,1,2)))
            UncNormFlow = np.uint8(np.floor(UncNormFlow*255.))
            
            # FlowMagQuant =  iu.remap(np.floor(iu.remap(np.sqrt(B[:,:,0]**2 + B[:,:,1]**2), 0., Args.NumBins)), 0., 255.)
            
            cv2.imshow('UncX, UncY', np.hstack((BUncDisp[:,:,0], BUncDisp[:,:,1])))
            cv2.imshow('NormUncX, NormUncY', np.hstack((UncNormFlow[:,:,0], UncNormFlow[:,:,1])))
            # cv2.imshow('FlowMag, FlowMagQuant', np.hstack((np.uint8(FlowMag), np.uint8(FlowMagQuant))))
            cv2.imwrite('Unc.png', BUncDisp[:,:,0])
            cv2.imwrite('FlowVis.png', BDisp)
            cv2.waitKey(0)

        # Pretty Print Stats before exiting
        PrettyPrint(Args, NumParams, NumFlops, ModelSize, VN)
    
        
        
def main():
    """
    Inputs: 
    None
    Outputs:
    Runs the Training and testing code based on the Flag
    """
    # Parse Command Line arguments
    Parser = argparse.ArgumentParser()
    Parser.add_argument('--NetworkType', default='Large', help='Choice of Network type, choose from Small, Large, Default:Large')
    Parser.add_argument('--NetworkName', default='Network.ResNet', help='Name of network file, Default: Network.ResNet')
    Parser.add_argument('--CheckPointPath', default='/media/nitin/Education/PRGEyeOmni/Trained/SL1ResNetLR1e-4/199model.ckpt', \
        help='Path to save checkpoints, Default:/media/nitin/Education/PRGEyeOmni/Trained/SL1ResNetLR1e-4/199model.ckpt')
    Parser.add_argument('--GPUDevice', type=int, default=0, help='What GPU do you want to use? -1 for CPU, Default:0')
    Parser.add_argument('--DataAug', type=int, default=0, help='Do you want to do Data augmentation?, Default:0')
    Parser.add_argument('--InitNeurons', type=float, default=32, help='Number of starting neurons, Default: 32')
    Parser.add_argument('--Suffix', default='', help='Suffix for Naming Network, Default: ''')
    Parser.add_argument('--ImgSize', default='[240,320,3]', help='Image Size as list, Default: [240,320,3]')
    Parser.add_argument('--UncType', default='None', help='What type of uncertainity do you want? Choose from None or Aleatoric, Default: None')
    Parser.add_argument('--NumBins', default=0, type=int, help='Number of Bins, Default: 0 (No discretization)')
    Parser.add_argument('--Img1', default='None', help='First Image Path, Default: None')
    Parser.add_argument('--Img2', default='None', help='Second Image Path, Default: None')
    Parser.add_argument('--GT1', default='None', help='Ground Truth 1 to 2, Default: None')
    Parser.add_argument('--Attack', action='store_true', help='Enable Attacking of Flow using Patch in --AttackPath and --AttackSize')
    Parser.add_argument('--AttackPath', default='/home/nitin/GIT/flowattack/patches/Upatch1.png', help='Path to the attacking patch, Default: /home/nitin/GIT/flowattack/patches/Upatch2.png')
    Parser.add_argument('--AttackSize', default='51', type=int,  help='Size of the attacking patch in pixels, Default: 64 (Square Patch)')    
    
    
    Args = Parser.parse_args()
    
    # Import Network Module
    Args.Net = importlib.import_module(Args.NetworkName)

    # Set GPUDevice
    tu.SetGPU(Args.GPUDevice)

    # Setup all needed parameters including file reading
    Args.PatchSize = Args.ImgSize.replace('[', ' ').replace(']', ' ').replace(',', ' ').split()
    Args.PatchSize = np.array([int(i) for i in Args.PatchSize])

    # np.array([480, 320, 3]) # 240, 320 # 256, 256 for Omni
    Args.NumOut = 2

    # Define PlaceHolder variables for Input and Predicted output
    InputPH = tf.compat.v1.placeholder(tf.float32, shape=(1, Args.PatchSize[0], Args.PatchSize[1], 2*Args.PatchSize[2]), name='Input')

    # PH for losses
    I1PH = tf.compat.v1.placeholder(tf.float32, shape=(1, Args.PatchSize[0], Args.PatchSize[1], Args.PatchSize[2]), name='I1')
    I2PH = tf.compat.v1.placeholder(tf.float32, shape=(1, Args.PatchSize[0], Args.PatchSize[1], Args.PatchSize[2]), name='I2')
    
    Label1PH =  tf.compat.v1.placeholder(tf.float32, shape=(1, Args.PatchSize[0], Args.PatchSize[1], Args.NumOut), name='Label1')
    Label2PH =  tf.compat.v1.placeholder(tf.float32, shape=(1, Args.PatchSize[0], Args.PatchSize[1], Args.NumOut), name='Label2')
   
    TestOperation(InputPH, I1PH, I2PH, Label1PH, Label2PH, Args)
    
    
if __name__ == '__main__':
    main()

