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
from Misc.BatchCreationTestTF import *
from Misc.Decorators import *
# Import of network is done in main code
import importlib
from datetime import datetime
import getpass
import copy
import platform

# Don't generate pyc codes
sys.dont_write_bytecode = True


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
    cprint('Images used for Testing are in: {}'.format(Args.BasePath), 'red')

def WriteHeader(PredOuts, Args, NumParams, NumFlops, ModelSize, VN):
    PredOuts.write('Network Used: {}\n'.format(Args.NetworkName))
    PredOuts.write('Init Neurons {}, Expansion Factor {}, NumBlocks {}, NumSubBlocks {}, DropOutFactor {}\n'.format(VN.InitNeurons, VN.ExpansionFactor,\
                                                                                                                      VN.NumBlocks, VN.NumSubBlocks,  VN.DropOutRate))
    PredOuts.write('Num Params: {}\n'.format(NumParams))
    PredOuts.write('Num FLOPs: {}\n'.format(NumFlops))
    PredOuts.write('Estimated Model Size (MB): {}\n'.format(ModelSize))
    PredOuts.write('CheckPoints are saved in: {}\n'.format(Args.CheckPointPath))
    PredOuts.write('Images used for Testing are in: {}\n'.format(Args.BasePath))

        
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
    Saver = tf.train.Saver()

    with tf.Session() as sess:       
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
        FileName = Args.CheckPointPath.replace('/'+ModelName,'') + os.sep + 'PredOuts.txt'
        PredOuts = open(FileName, 'w')

        WriteHeader(PredOuts, Args, NumParams, NumFlops, ModelSize, VN)
        cprint('Header written to {}'.format(FileName), 'yellow')
        PredOuts.write('FileName, ErrorL1Px., ErrorL2Px.\n')

        ListIgnore = ['train', 'img_1', 'oids', 'flow', 'occ', 'mb']

        for subdir, dirs, files in tqdm(os.walk(Args.BasePath)):
            for file in files:
                FileNameNow = os.path.join(subdir, file)
                if(not FileNameNow.endswith(Args.ImgFormat) or any([True for a in ListIgnore if a in FileNameNow])):
                    # Skip filenames with right as these can be generated by replacing left with right
                    # Do not evaluate on Train names
                    continue
                else:
                    IBatch, I1Batch, I2Batch, P1Batch, P2Batch, Label1Batch, Label2Batch = bg.GenerateBatchTF(FileNameNow, Args, da, DataAugGen)

                    if(IBatch is None):
                        continue

                    FeedDict = {VN.InputPH: IBatch, I1PH: P1Batch, I2PH: P2Batch, Label1PH: Label1Batch, Label2PH: Label2Batch}
                    prValRet = sess.run([prVal], feed_dict=FeedDict)

                    A = np.squeeze(Label1Batch[0])
                    
                    if(Args.UncType == 'Aleatoric'):
                        B = np.squeeze(prValRet[0])[:,:,0:2]
                    else:
                        B = np.squeeze(prValRet[0])
                    # ADisp = np.uint8(np.array(mu.remap(A, 0., 255.))))
                    # BDisp = np.uint8(np.array(mu.remap(B, 0., 255.))))

                    # cv2.imshow('GT, Pred', np.hstack((ADisp, BDisp)))
                    # cv2.waitKey(0)

                    ErrorL1 = np.abs(A-B)
                    ErrorL2 = np.sqrt((A[:,:,0]-B[:,:,0])**2 + (A[:,:,1]-B[:,:,1])**2)
                    MeanErrorL1Px = np.mean(ErrorL1)
                    MeanErrorL2Px = np.mean(ErrorL2)

                    PredOuts.write('{}, {}, {}\n'.format(FileNameNow, MeanErrorL1Px, MeanErrorL2Px))
                   
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
    Parser.add_argument('--BasePath', default='/home/nitin/Datasets/FlyingChairs2', help='Base path of images, Default:/home/nitin/Datasets/FlyingChairs2')
    Parser.add_argument('--NetworkType', default='Large', help='Choice of Network type, choose from Small, Large, Default:Large')
    Parser.add_argument('--NetworkName', default='Network.ResNet', help='Name of network file, Default: Network.ResNet')
    Parser.add_argument('--CheckPointPath', default='/home/nitin/VariableBaseLineStereo/Trained/ResNetL1/49model.ckpt', \
        help='Path to save checkpoints, Default:/home/nitin/VariableBaseLineStereo/Trained/ResNetL1/49model.ckpt')
    Parser.add_argument('--GPUDevice', type=int, default=0, help='What GPU do you want to use? -1 for CPU, Default:0')
    Parser.add_argument('--DataAug', type=int, default=0, help='Do you want to do Data augmentation?, Default:0')
    Parser.add_argument('--InitNeurons', type=float, default=32, help='Number of starting neurons, Default: 32')
    Parser.add_argument('--ImgFormat', default='.png', help='Image Format, Default: .png')
    Parser.add_argument('--Suffix', default='', help='Suffix for Naming Network, Default: ''')
    Parser.add_argument('--UncType', default='None', help='What type of uncertainity do you want? Choose from None or Aleatoric, Default: None')
    
    Args = Parser.parse_args()
    
    # Import Network Module
    Args.Net = importlib.import_module(Args.NetworkName)

    # Set GPUDevice
    tu.SetGPU(Args.GPUDevice)

    # Setup all needed parameters including file reading
    Args.PatchSize = np.array([240, 320, 3])
    Args.NumOut = 2

    # Define PlaceHolder variables for Input and Predicted output
    InputPH = tf.placeholder(tf.float32, shape=(1, Args.PatchSize[0], Args.PatchSize[1], 2*Args.PatchSize[2]), name='Input')

    # PH for losses
    I1PH = tf.placeholder(tf.float32, shape=(1, Args.PatchSize[0], Args.PatchSize[1], Args.PatchSize[2]), name='I1')
    I2PH = tf.placeholder(tf.float32, shape=(1, Args.PatchSize[0], Args.PatchSize[1], Args.PatchSize[2]), name='I2')
        
    Label1PH =  tf.placeholder(tf.float32, shape=(1, Args.PatchSize[0], Args.PatchSize[1], Args.NumOut), name='Label1')
    Label2PH =  tf.placeholder(tf.float32, shape=(1, Args.PatchSize[0], Args.PatchSize[1], Args.NumOut), name='Label2')
        
   
    TestOperation(InputPH, I1PH, I2PH, Label1PH, Label2PH, Args)
    
    
if __name__ == '__main__':
    main()

