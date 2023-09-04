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
from StringIO import StringIO
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
from Network.SqueezeNet import *


# Don't generate pyc codes
sys.dont_write_bytecode = True

    
def TrainOperation(ImgPH, MiniBatchSize, LogsPath):

    # Create Network
    prHVal, prVal, WarpI1Patch = ICSTN(ImgPH, PatchSize, MiniBatchSize, opt)
    
    # Tensorboard
    # Merge all summaries into a single operation
    MergedSummaryOP = tf.summary.merge_all()

    # Setup Saver
    Saver = tf.train.Saver()
    
    with tf.Session() as sess:       
        sess.run(tf.global_variables_initializer())
        StartEpoch = 0
        print('New model initialized....')

        # Print Number of parameters in the network    
        tu.FindNumParams(1)
        
        # Tensorboard
        Writer = tf.summary.FileWriter(LogsPath, graph=tf.get_default_graph())

        for Epochs in tqdm(range(StartEpoch, NumEpochs)):
            NumIterationsPerEpoch = int(NumTrainSamples/MiniBatchSize/DivTrain)
            for PerEpochCounter in tqdm(range(NumIterationsPerEpoch)):
                IBatch, I1Batch, I2Batch, P1Batch, P2Batch, HBatch, ParamsBatch = bg.GenerateBatchTF(TrainNames, PatchSize, MiniBatchSize, HObj, BasePath, OriginalImageSize)

                FeedDict = {ImgPH: IBatch, I1PH: P1Batch, I2PH: P2Batch, LabelPH: ParamsBatch}
                _, LossThisBatch, Summary, WarpI1PatchIdealRet = sess.run([OptimizerUpdate, loss, MergedSummaryOP, WarpI1PatchIdeal], feed_dict=FeedDict)
                
                # Tensorboard
                Writer.add_summary(Summary, Epochs*NumIterationsPerEpoch + PerEpochCounter)
                # If you don't flush the tensorboard doesn't update until a lot of iterations!
                Writer.flush()

                # Save checkpoint every some SaveCheckPoint's iterations
                if PerEpochCounter % SaveCheckPoint == 0:
                    # Save the Model learnt in this epoch
                    SaveName =  CheckPointPath + str(Epochs) + 'a' + str(PerEpochCounter) + 'model.ckpt'
                    Saver.save(sess,  save_path=SaveName)
                    print(SaveName + ' Model Saved...')
                            
            # Save model every epoch
            SaveName = CheckPointPath + str(Epochs) + 'model.ckpt'
            Saver.save(sess, save_path=SaveName)
            print(SaveName + ' Model Saved...')


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
    Parser.add_argument('--LogsPath', default='/home/nitin/PRGEye/Logs3/', help='Path to save Logs, Default:/home/nitin/PRGEye/Logs/')
    Parser.add_argument('--GPUDevice', type=int, default=1, help='What GPU do you want to use? -1 for CPU, Default:0')

    
    Args = Parser.parse_args()
    MiniBatchSize = 32
    LogsPath = Args.LogsPath
    GPUDevice = Args.GPUDevice
    
    # Set GPUDevice
    tu.SetGPU(GPUDevice)

    # Image Input Shape
    PatchSize = np.array([128, 128, 3])
    
    # Define PlaceHolder variables for Input and Predicted output
    ImgPH = tf.placeholder(tf.float32, shape=(MiniBatchSize, PatchSize[0], PatchSize[1], 2*PatchSize[2]), name='Input')

    TrainOperation(ImgPH, MiniBatchSize, LogsPath)
        
    
if __name__ == '__main__':
    main()

