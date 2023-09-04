#!/usr/bin/env python

# Dependencies:
# opencv, do (pip install opencv-python)
# skimage, do (sudo)
# termcolor, do (pip install termcolor)
# tqdm, do (pip install tqdm)

# TODO:
# Clean print statements
# Global step only loss/epoch on tensorboard
# Print Num parameters in model as a function
# Clean comments
# Check Factor from network list
# ClearLogs command line argument
# Adapt more augmentation from: https://github.com/sthalles/deeplab_v3/blob/master/preprocessing/inception_preprocessing.py
# Tensorboard logging of images

import tensorflow as tf
import cv2
import sys
import os
import glob
import Misc.ImageUtils as iu
import random
from skimage import data, exposure, img_as_float
import matplotlib.pyplot as plt
from Network.HomographyNetICSTNSimpler import  ICSTN
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
import Misc.warpICSTN as warp
import Misc.warpICSTN2 as warp2
from Misc.DataHandling import *
from Misc.BatchCreationNP import *
from Misc.BatchCreationTF import *

# from pycallgraph import PyCallGraph
# from pycallgraph.output import GraphvizOutput

# Don't generate pyc codes
sys.dont_write_bytecode = True         


def SetupAll(ReadPath):
    """
    Inputs: 
    BasePath is the base path where Images are saved without "/" at the end
    Outputs:
    DirNames - Full path to all image files without extension
    Train/Val/Test - Idxs of all the images to be used for training/validation (held-out testing in this case)/testing
    Ratios - Ratios is a list of fraction of data used for [Train, Val, Test]
    CheckPointPath - Path to save checkpoints/model
    OptimizerParams - List of all OptimizerParams: depends on Optimizer
    SaveCheckPoint - Save checkpoint every SaveCheckPoint iteration in every epoch, checkpoint saved automatically after every epoch
    ImageSize - Size of the image
    NumTrain/Val/TestSamples - length(Train/Val/Test)
    NumTestRunsPerEpoch - Number of passes of Val data with MiniBatchSize 
    Train/Val/TestLabels - Labels corresponding to Train/Val/Test
    """
    # Setup DirNames
    DirNamesPath = ReadPath + os.sep + 'DirNames.txt'
    TestNames = ReadDirNames(DirNamesPath)
    
    # Image Input Shape
    PatchSize = np.array([240, 320, 3])
    ImageSize = np.array([540, 960, 3])
    NumTestSamples = len(TestNames)

    return TestNames, ImageSize, PatchSize, NumTestSamples

def ReadDirNames(DirNamesPath):
    """
    Inputs: 
    Path is the path of the file you want to read
    Outputs:
    DirNames is the data loaded from ./TxtFiles/DirNames.txt which has full path to all image files without extension
    """
    # Read DirNames fil1e
    DirNames = open(DirNamesPath, 'r')
    DirNames = DirNames.read()
    DirNames = DirNames.split()

    return DirNames

def main():
    """
    Inputs: 
    None
    Outputs:
    Runs the Training and testing code based on the Flag
    """
    # TODO: Make LogDir
    # TODO: Make logging file a parameter
    # TODO: Time to complete print

    # Parse Command Line arguments
    Parser = argparse.ArgumentParser()
    Parser.add_argument('--BasePath', default='/media/nitin/Education/Datasets/FlyingThings3D', help='Base path of images, Default:/home/nitin/Datasets/MSCOCO/train2014')
    Parser.add_argument('--MiniBatchSize', type=int, default=1, help='MiniBatchSize, Default:1')
    Parser.add_argument('--GPUDevice', type=int, default=-1, help='GPUDevice, Default:-1')

    Args = Parser.parse_args()
    BasePath = Args.BasePath
    MiniBatchSize = Args.MiniBatchSize
    GPUDevice = Args.GPUDevice

    tu.SetGPU(GPUDevice)
 
    TrainNames, ImageSize, PatchSize, NumTrainSamples = SetupAll(BasePath)

    Args.TrainNames = TrainNames
    Args.ImageSize = ImageSize
    Args.PatchSize = PatchSize
    Args.NumTrainSamples = NumTrainSamples
    Args.DataAug = True
    

    IPH = tf.placeholder(tf.float32, shape=(MiniBatchSize, PatchSize[0], PatchSize[1], PatchSize[2]), name='Input')
    ImgPH = tf.placeholder(tf.float32, shape=(MiniBatchSize, PatchSize[0], PatchSize[1], 2*PatchSize[2]), name='Input')

    # graphviz = GraphvizOutput()
    # graphviz.output_file = 'basic.png'
    
    # with PyCallGraph(output=graphviz):
    with tf.Session() as sess:
        # Create Data Augmentation Object
        Augmentations =  ['Brightness', 'Contrast', 'Hue', 'Saturation', 'Gamma', 'Gaussian']
        da = iu.DataAugmentationTF(sess, IPH, Augmentations)
        DataAugGen = da.RandPerturbBatch()
            
        # Generate Batch Generation Object
        bg = BatchGeneration(sess, IPH)
            
        Timer1 = mu.tic()
        IBatch, I1Batch, I2Batch, P1Batch, P2Batch, Label1Batch, Label2Batch = bg.GenerateBatchTF(Args, da, DataAugGen)
        print(mu.toc(Timer1))
        
    
    # for count in range(MiniBatchSize):
    #     cv2.imshow('I, IPerturb {}/{}'.format(count+1, Args.MiniBatchSize), np.hstack((P1Batch[count], PBatch[count])))
    #     cv2.waitKey(0)


if __name__ == '__main__':
    main()
