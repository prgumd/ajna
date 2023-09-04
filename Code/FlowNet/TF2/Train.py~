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
from Misc.BatchCreationTF import *
from Misc.Decorators import *
from Misc.FlowVisUtilsTF import *
# Import of network is done in main code
import importlib
from datetime import datetime
import getpass
import copy
import platform

# Don't generate pyc codes
sys.dont_write_bytecode = True

@Scope
def Loss(I1PH, I2PH, Label1PH, Label2PH, prVal, Args):
    def SSIM(I1, I2):
        # Adapted from: https://github.com/yzcjtr/GeoNet/blob/master/geonet_model.py
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        mu_x = tf.nn.avg_pool(I1, ksize = (3,3), strides=(1,1), padding='SAME')
        mu_y = tf.nn.avg_pool(I2, ksize = (3,3), strides=(1,1), padding='SAME')

        sigma_x  = tf.nn.avg_pool(I1 ** 2, ksize = (3,3), strides=(1,1), padding='SAME') - mu_x ** 2
        sigma_y  = tf.nn.avg_pool(I2 ** 2, ksize = (3,3), strides=(1,1), padding='SAME') - mu_y ** 2
        sigma_xy = tf.nn.avg_pool(I1 * I2 , ksize = (3,3), strides=(1,1), padding='SAME') - mu_x * mu_y
        
        SSIM_n = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2)
        
        SSIM = SSIM_n / SSIM_d
        return SSIM

    if(Args.UncType == 'Aleatoric'):
        # Ideas from https://github.com/pmorerio/dl-uncertainty/blob/master/aleatoric-uncertainty/model.py
        eps = 1e-6  # To avoid Inf
        MaxVal = 10.
        prDisparity = prVal[:,:,:,:Args.NumOut]
        prLogvar = prVal[:,:,:,Args.NumOut:] # tf.clip_by_value(prVal[:,:,:,Args.NumOut:], eps, MaxVal)
    if(Args.UncType == 'Inlier' or Args.UncType == 'LinearSoftplus'):
        prDisparity = prVal[:,:,:,:Args.NumOut]
        prLogvar = prVal[:,:,:,Args.NumOut:] # Inlier Mask
    else:
        prDisparity = prVal
        prLogvar = None
    
        # Choice of Loss Function
        if(Args.LossFuncName == 'SL2-1'):
            # Supervised L2 loss
            lossPhoto = tf.reduce_mean(tf.square(prDisparity - Label1PH))
        if(Args.LossFuncName == 'SL1-1'):
            # Supervised L1 loss
            lossPhoto = tf.reduce_mean(tf.abs(prDisparity - Label1PH))
        elif(Args.LossFuncName == 'PhotoL1-1'):        
            # Self-supervised Photometric L1 Losses
            DiffImg = prDisparity - Label1PH # iu.StandardizeInputsTF(WarpI1Patch[:,:,:,0:3] - I2PH)
            lossPhoto = tf.reduce_mean(tf.abs(DiffImg))
        elif(Args.LossFuncName == 'PhotoChab-1'):
            # Self-supervised Photometric Chabonier Loss
            DiffImg = prDisparity - Label1PH
            epsilon = 1e-3
            alpha = 0.45
            lossPhoto = tf.reduce_mean(tf.pow(tf.square(DiffImg) + tf.square(epsilon), alpha))
        elif(Args.LossFuncName == 'SSIM-1'):
            DiffImg = prDisparity - Label1PH

            AlphaSSIM = 0.005
            lossPhoto = tf.reduce_mean(tf.clip_by_value((1 - SSIM(prDisparity, Label1PH)) / 2, 0, 1) + AlphaSSIM*tf.abs(DiffImg))
        elif(Args.LossFuncName == 'PhotoRobust'):
            Epsa = 1e-3
            c = 1e-2 # 1e-1 was used before
            DiffImg = WarpI1Patch - I2PH
            a = C2PH/255.0
            a = tf.multiply((2.0 - 2.0*Epsa), tf.math.sigmoid(a)) + Epsa
            lossPhoto = tf.reduce_mean(nll(DiffImg, a, c = c))

    if(Args.RegFuncName == 'None'):
        lossReg = 0.
    else:
        print('Unknown Reg Func Type')
        sys.exit()

    if(prLogvar is not None):
        if(Args.UncType == 'Aleatoric'):
            # Custom Using L1 Inspired from "From What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision?" NIPS 2017
            lossUnc = 0.5*(tf.reduce_mean(tf.multiply(tf.exp(-prLogvar), tf.abs(prDisparity - Label1PH))) + tf.reduce_mean(prLogvar))   # tf.square(prDisparity - Label1PH)
            # L2 as in Lightweight Probabilistic Deep Networks
            # Gives Nans
            # eps = 1e-3
            # lossUnc = 0.5*(tf.reduce_mean(tf.math.sqrt(tf.multiply(tf.exp(-prLogvar) + eps, tf.math.square(prDisparity - Label1PH)))) + tf.reduce_mean(prLogvar))
        if(Args.UncType == 'Inlier'):
            # Idea from: https://github.com/tinghuiz/SfMLearner/blob/master/SfMLearner.py
            Lambda = 0.2
            InlierMask = prLogvar[:,:,:,1]
            RefMask = np.tile(np.array([0,1]), (Args.MiniBatchSize, Args.PatchSize[0], Args.PatchSize[1], 1))
            lossUnc = tf.reduce_mean(tf.multiply(tf.expand_dims(tf.nn.softmax(InlierMask), -1), tf.abs(prDisparity - Label1PH))) + \
                      Lambda*tf.reduce_mean(Lambda*tf.nn.softmax_cross_entropy_with_logits(labels=tf.reshape(RefMask, [-1, 2]), logits=tf.reshape(prLogvar, [-1, 2])))
        if(Args.UncType == 'LinearSoftplus'):
            Eps = 1e-3
            # prLogvar is linear in this case not logrithmic
            # lossUnc = tf.abs(prDisparity - Label1PH) + tf.reduce_mean(tf.math.softplus(prLogvar)) Try this!
            lossUnc = tf.reduce_mean((1/tf.math.softplus(prLogvar + Eps))*tf.abs(prDisparity - Label1PH)) + tf.reduce_mean(tf.math.softplus(prLogvar))
            # gives Inf
            # lossUnc = tf.reduce_mean((1/tf.math.softplus(prLogvar + Eps))*tf.abs(prDisparity - Label1PH)) + tf.reduce_mean(tf.math.sigmoid(prLogvar)) gives large values of prDisparity at 1e4 range
            # lossUnc = tf.reduce_mean((1/tf.math.sigmoid(prLogvar + Eps))*tf.abs(prDisparity - Label1PH)) + tf.reduce_mean(tf.math.sigmoid(prLogvar))
            
        return lossUnc + lossReg
    else:
        return lossPhoto + lossReg 


@Scope
def Optimizer(OptimizerParams, loss):
    Optimizer = tf.train.AdamOptimizer(learning_rate=OptimizerParams[0], beta1=OptimizerParams[1],
                                           beta2=OptimizerParams[2], epsilon=OptimizerParams[3])
    Gradients = Optimizer.compute_gradients(loss)
    OptimizerUpdate = Optimizer.apply_gradients(Gradients)
    # Optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-8).minimize(loss)
    # Optimizer = tf.train.MomentumOptimizer(learning_rate=1e-3, momentum=0.9, use_nesterov=True).minimize(loss)
    return OptimizerUpdate

def TensorBoard(loss, I1PH, I2PH, prVal, Label1PH, Label2PH, Args):
    # Create a summary to monitor loss tensor
    tf.summary.scalar('LossEveryIter', loss)
    tf.summary.image('I1Patch', I1PH[:,:,:,0:3], max_outputs=3)
    tf.summary.image('I2Patch', I2PH[:,:,:,0:3], max_outputs=3)
    # tf.summary.image('prVal', prVal[:,:,:,0:1], max_outputs=3)
    # tf.summary.image('Label1', Label1PH[:,:,:,0:1], max_outputs=3)
    # tf.summary.image('Label2', Label2PH[:,:,:,0:1], max_outputs=3)
    Label1FlowVis, _, _ = flow_viz_tf(Label1PH)
    Label2FlowVis, _, _ = flow_viz_tf(Label2PH)
    prFlowVis, _, _ = flow_viz_tf(prVal[:,:,:,0:2])
    tf.summary.image('Label1', Label1FlowVis[:,:,:,0:3], max_outputs=3)
    tf.summary.image('Label2', Label2FlowVis[:,:,:,0:3], max_outputs=3)
    tf.summary.image('prVal', prFlowVis[:,:,:,0:3], max_outputs=3)
    tf.summary.histogram('Label1Hist', Label1PH)
    tf.summary.histogram('Label2Hist', Label2PH)
    if(Args.UncType == 'Aleatoric'):
        eps = 1e-6  # To avoid Inf
        MaxVal = 10.
        prLogvarX = tf.clip_by_value(prVal[:,:,:,2:3], eps, MaxVal)
        prLogvarY = tf.clip_by_value(prVal[:,:,:,3:4], eps, MaxVal)
        prLogvar = tf.clip_by_value(prVal[:,:,:,2:4], eps, MaxVal)
        tf.summary.histogram('prLogValHist', prLogvar)
        tf.summary.histogram('prValHist', prVal[:,:,:,0:2])
        tf.summary.image('AleatoricUncX', tf.exp(prLogvarX), max_outputs=3)
        tf.summary.image('AleatoricUncY', tf.exp(prLogvarY), max_outputs=3)
        tf.summary.histogram('AleatoricUncHistX', tf.exp(prLogvarX))
        tf.summary.histogram('AleatoricUncHistY', tf.exp(prLogvarY))
    elif(Args.UncType == 'Inlier'):
        tf.summary.histogram('FlowPred', prVal[:,:,:,0:2])
        tf.summary.image('Inlier', prVal[:,:,:,2:3], max_outputs=3)
        tf.summary.histogram('Inlier', prVal[:,:,:,2:3])
    elif(Args.UncType == 'LinearSoftplus'):
        Eps = 1e-3
        MaxVal = 1e9
        # Sigmoid
        # tf.summary.image('ScaleX', tf.clip_by_value(1/tf.math.sigmoid(prVal[:,:,:,2:3] + Eps), -MaxVal, MaxVal))
        # tf.summary.image('ScaleY', tf.clip_by_value(1/tf.math.sigmoid(prVal[:,:,:,3:4] + Eps), -MaxVal, MaxVal))
        # tf.summary.histogram('Scale', tf.clip_by_value(1/tf.math.sigmoid(prVal + Eps), -MaxVal, MaxVal))
        # Softplus
        tf.summary.image('ScaleX', tf.clip_by_value(1/tf.math.softplus(prVal[:,:,:,2:3] + Eps), -MaxVal, MaxVal))
        tf.summary.image('ScaleY', tf.clip_by_value(1/tf.math.softplus(prVal[:,:,:,3:4] + Eps), -MaxVal, MaxVal))
        tf.summary.histogram('Scale', tf.clip_by_value(1/tf.math.softplus(prVal + Eps), -MaxVal, MaxVal))
        tf.summary.histogram('prValHist', prVal[:,:,:,0:2])
    else:
        tf.summary.histogram('prValHist', prVal)
    # Merge all summaries into a single operation
    MergedSummaryOP = tf.summary.merge_all()
    return MergedSummaryOP


def PrettyPrint(Args, NumParams, NumFlops, ModelSize, VN, OverideKbInput=False):
    # TODO: Write to file?
    Username = getpass.getuser()
    cprint('Running on {}'.format(Username), 'yellow')
    cprint('Network Statistics', 'yellow')
    cprint('Network Used: {}'.format(Args.NetworkName), 'yellow')
    cprint('Uncertainity Type: {}'.format(Args.UncType), 'yellow')
    cprint('GPU Used: {}'.format(Args.GPUDevice), 'yellow')
    cprint('Learning Rate: {}'.format(Args.LR), 'yellow')
    cprint('Init Neurons {}, Expansion Factor {}, NumBlocks {}, NumSubBlocks {}, DropOutFactor {}'.format(VN.InitNeurons, VN.ExpansionFactor,\
                                                                                                          VN.NumBlocks, VN.NumSubBlocks, VN.DropOutRate), 'yellow')
    cprint('Num Params: {}'.format(NumParams), 'green')
    cprint('Num FLOPs: {}'.format(NumFlops), 'green')
    cprint('Estimated Model Size (MB): {}'.format(ModelSize), 'green')
    cprint('Loss Function used: {}'.format(Args.LossFuncName), 'green')
    cprint('Loss Function Weights: {}'.format(Args.Lambda), 'green')
    cprint('Reg Function used: {}'.format(Args.RegFuncName), 'green')
    cprint('Augmentations Used: {}'.format(Args.Augmentations), 'green')
    cprint('CheckPoints are saved in: {}'.format(Args.CheckPointPath), 'red')
    cprint('Logs are saved in: {}'.format(Args.LogsPath), 'red')
    cprint('Images used for Training are in: {}'.format(Args.BasePath), 'red')
    if(OverideKbInput):
        Key = 'y'
    else:
        PythonVer = platform.python_version().split('.')[0]
        # Parse Python Version to handle super accordingly
        if (PythonVer == '2'):
            Key = raw_input('Enter y/Y/yes/Yes/YES to save to RunCommand.md, any other key to exit.')
        else:
            Key = input('Enter y/Y/yes/Yes/YES to save to RunCommand.md, any other key to exit.')
    if(Key.lower() == 'y' or Key.lower() == 'yes'):
        FileName = 'RunCommand.md'
        with open(FileName, 'a+') as RunCommand:
            RunCommand.write('\n\n')
            RunCommand.write('{}\n'.format(datetime.now()))
            RunCommand.write('Username: {}\n'.format(Username))
            RunCommand.write('Learning Rate: {}\n'.format(Args.LR))
            RunCommand.write('Network Used: {}\n'.format(Args.NetworkName))
            RunCommand.write('Uncertainity Type: {}\n'.format(Args.UncType))
            RunCommand.write('GPU Used: {}\n'.format(Args.GPUDevice))
            RunCommand.write('Init Neurons {}, Expansion Factor {}, NumBlocks {}, NumSubBlocks {}, DropOutFactor {}\n'.format(VN.InitNeurons, VN.ExpansionFactor,\
                                                                                                                              VN.NumBlocks, VN.NumSubBlocks,  VN.DropOutRate))
            RunCommand.write('Num Params: {}\n'.format(NumParams))
            RunCommand.write('Num FLOPs: {}\n'.format(NumFlops))
            RunCommand.write('Estimated Model Size (MB): {}\n'.format(ModelSize))
            RunCommand.write('Loss Function used: {}\n'.format(Args.LossFuncName))
            RunCommand.write('Loss Function Weights: {}\n'.format(Args.Lambda))
            RunCommand.write('Reg Function used: {}\n'.format(Args.RegFuncName))
            RunCommand.write('Augmentations Used: {}\n'.format(Args.Augmentations))
            RunCommand.write('CheckPoints are saved in: {}\n'.format(Args.CheckPointPath))
            RunCommand.write('Logs are saved in: {}\n'.format(Args.LogsPath))
            RunCommand.write('Images used for Training are in: {}\n'.format(Args.BasePath))
        cprint('Log written in {}'.format(FileName), 'yellow')
        FileName = Args.CheckPointPath + 'RunCommand.md'
        with open(FileName, 'w+') as RunCommand:
            RunCommand.write('\n\n')
            RunCommand.write('{}\n'.format(datetime.now()))
            RunCommand.write('Username: {}\n'.format(Username))
            RunCommand.write('Learning Rate: {}\n'.format(Args.LR))
            RunCommand.write('Network Used: {}\n'.format(Args.NetworkName))
            RunCommand.write('Uncertainity Type: {}\n'.format(Args.UncType))
            RunCommand.write('GPU Used: {}\n'.format(Args.GPUDevice))
            RunCommand.write('Init Neurons {}, Expansion Factor {}, NumBlocks {}, NumSubBlocks {}, DropOutFactor {}\n'.format(VN.InitNeurons, VN.ExpansionFactor,\
                                                                                                                              VN.NumBlocks, VN.NumSubBlocks, VN.DropOutRate))
            RunCommand.write('Num Params: {}\n'.format(NumParams))
            RunCommand.write('Num FLOPs: {}\n'.format(NumFlops))
            RunCommand.write('Estimated Model Size (MB): {}\n'.format(ModelSize))
            RunCommand.write('Loss Function used: {}\n'.format(Args.LossFuncName))
            RunCommand.write('Loss Function Weights: {}\n'.format(Args.Lambda))
            RunCommand.write('Reg Function used: {}\n'.format(Args.RegFuncName))
            RunCommand.write('Augmentations Used: {}\n'.format(Args.Augmentations))
            RunCommand.write('CheckPoints are saved in: {}\n'.format(Args.CheckPointPath))
            RunCommand.write('Logs are saved in: {}\n'.format(Args.LogsPath))
            RunCommand.write('Images used for Training are in: {}\n'.format(Args.BasePath))
        cprint('Log written in {}'.format(FileName), 'yellow')
    else:
        cprint('Log writing skipped', 'yellow')
        
    
def TrainOperation(InputPH, I1PH, I2PH, Label1PH, Label2PH, Args):
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

    # Compute Loss
    loss = Loss(I1PH, I2PH, Label1PH, Label2PH, prVal, Args)

    # Run Backprop and Gradient Update
    OptimizerUpdate = Optimizer(Args.OptimizerParams, loss)
        
    # Tensorboard
    MergedSummaryOP = TensorBoard(loss, I1PH, I2PH, prVal, Label1PH, Label2PH, Args)
 
    # Setup Saver
    Saver = tf.train.Saver()

    try:
        with tf.Session() as sess:       
            if Args.LatestFile is not None:
                Saver.restore(sess, Args.CheckPointPath + Args.LatestFile + '.ckpt')
                # Extract only numbers from the name
                StartEpoch = int(''.join(c for c in Args.LatestFile.split('a')[0] if c.isdigit())) + 1
                print('Loaded latest checkpoint with the name ' + Args.LatestFile + '....')
            else:
                sess.run(tf.global_variables_initializer())
                StartEpoch = 0
                print('New model initialized....')

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
            PrettyPrint(Args, NumParams, NumFlops, ModelSize, VN, OverideKbInput=False)

            # Tensorboard
            Writer = tf.summary.FileWriter(Args.LogsPath, graph=tf.get_default_graph())

            for Epochs in tqdm(range(StartEpoch, Args.NumEpochs)):
                NumIterationsPerEpoch = int(Args.NumTrainSamples/Args.MiniBatchSize/Args.DivTrain)
                for PerEpochCounter in tqdm(range(NumIterationsPerEpoch)):
                    IBatch, I1Batch, I2Batch, P1Batch, P2Batch, Label1Batch, Label2Batch = bg.GenerateBatchTF(Args, da, DataAugGen)

                    FeedDict = {VN.InputPH: IBatch, I1PH: P1Batch, I2PH: P2Batch, Label1PH: Label1Batch, Label2PH: Label2Batch}
                    _, LossThisBatch, Summary = sess.run([OptimizerUpdate, loss, MergedSummaryOP], feed_dict=FeedDict)
                   
                    # Tensorboard
                    Writer.add_summary(Summary, Epochs*NumIterationsPerEpoch + PerEpochCounter)
                    # If you don't flush the tensorboard doesn't update until a lot of iterations!
                    Writer.flush()

                    # Save checkpoint every some SaveCheckPoint's iterations
                    if PerEpochCounter % Args.SaveCheckPoint == 0:
                        # Save the Model learnt in this epoch
                        SaveName =  Args.CheckPointPath + str(Epochs) + 'a' + str(PerEpochCounter) + 'model.ckpt'
                        Saver.save(sess,  save_path=SaveName)
                        print(SaveName + ' Model Saved...')

                # Save model every epoch
                SaveName = Args.CheckPointPath + str(Epochs) + 'model.ckpt'
                Saver.save(sess, save_path=SaveName)
                print(SaveName + ' Model Saved...')

        # Pretty Print Stats before exiting
        PrettyPrint(Args, NumParams, NumFlops, ModelSize, VN, OverideKbInput=True)
    
    except KeyboardInterrupt:
        # Pretty Print Stats before exitting
        PrettyPrint(Args, NumParams, NumFlops, ModelSize, VN, OverideKbInput=False)


        
        
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
    Parser.add_argument('--LabelBasePath', default='/media/nitin/Research2/FlyingThings3D/flyingthings3d_optical_flow/', help='Base path of images, Default://media/nitin/Research2/FlyingThings3D/flyingthings3d__optical_flow/optical_flow')
    Parser.add_argument('--NumEpochs', type=int, default=50, help='Number of Epochs to Train for, Default:50')
    Parser.add_argument('--DivTrain', type=int, default=1, help='Factor to reduce Train data by per epoch, Default:1')
    Parser.add_argument('--MiniBatchSize', type=int, default=32, help='Size of the MiniBatch to use, Default:32')
    Parser.add_argument('--LoadCheckPoint', type=int, default=0, help='Load Model from latest Checkpoint from CheckPointPath?, Default:0')
    Parser.add_argument('--RemoveLogs', type=int, default=0, help='Delete log Files from ./Logs?, Default:0')
    Parser.add_argument('--LossFuncName', default='SL1-1', help='Choice of Loss functions, choose from SL2, PhotoL1, PhotoChab, PhotoRobust. Default:SL1-1')
    Parser.add_argument('--RegFuncName', default='None', help='Choice of regularization function, choose from None, C (Cornerness). Default:None')
    Parser.add_argument('--NetworkType', default='Large', help='Choice of Network type, choose from Small, Large, Default:Large')
    Parser.add_argument('--NetworkName', default='Network.ResNet', help='Name of network file, Default: Network.VanillaNet')
    Parser.add_argument('--CheckPointPath', default='/media/nitin/Education/PRGEyeOmni/CheckPoints/', help='Path to save checkpoints, Default:/media/nitin/Education/PRGEyeOmni/CheckPoints/')
    Parser.add_argument('--LogsPath', default='/media/nitin/Education/PRGEyeOmni/Logs/', help='Path to save Logs, Default:/media/nitin/Education/PRGEyeOmni/Logs/')
    Parser.add_argument('--GPUDevice', type=int, default=0, help='What GPU do you want to use? -1 for CPU, Default:0')
    Parser.add_argument('--DataAug', type=int, default=0, help='Do you want to do Data augmentation?, Default:0')
    Parser.add_argument('--LR', type=float, default=1e-4, help='Learning Rate, Default: 1e-4')
    Parser.add_argument('--InitNeurons', type=float, default=32, help='Learning Rate, Default: 32')
    Parser.add_argument('--Suffix', default='', help='Suffix for Naming Network, Default: ''')
    Parser.add_argument('--UncType', default='None', help='What type of uncertainity do you want? Choose from None or Aleatoric, Default: None')
    Parser.add_argument('--Dataset', default='FC2', help='Dataset: FC2 for Flying Chairs 2 or FT3D for Flying Things 3D, Default: FC2')
    Parser.add_argument('--Quant', default=0, type=int, help='Rescale Flow values from 0 to 255?, Default: 0')
        
    
    Args = Parser.parse_args()
    
    # Import Network Module
    Args.Net = importlib.import_module(Args.NetworkName)

    # Set GPUDevice
    tu.SetGPU(Args.GPUDevice)

    if(Args.RemoveLogs is not 0):
        shutil.rmtree(os.getcwd() + os.sep + 'Logs' + os.sep)

    # Setup all needed parameters including file reading
    Args = SetupAll(Args)    

    # If CheckPointPath doesn't exist make the path
    if(not (os.path.isdir(Args.CheckPointPath))):
       os.makedirs(Args.CheckPointPath)
    
    # Find Latest Checkpoint File
    if Args.LoadCheckPoint==1:
        Args.LatestFile = FindLatestModel(Args.CheckPointPath)
    else:
        Args.LatestFile = None
        
    # Define PlaceHolder variables for Input and Predicted output
    InputPH = tf.placeholder(tf.float32, shape=(Args.MiniBatchSize, Args.PatchSize[0], Args.PatchSize[1], 2*Args.PatchSize[2]), name='Input')

    # PH for losses
    I1PH = tf.placeholder(tf.float32, shape=(Args.MiniBatchSize, Args.PatchSize[0], Args.PatchSize[1], Args.PatchSize[2]), name='I1')
    I2PH = tf.placeholder(tf.float32, shape=(Args.MiniBatchSize, Args.PatchSize[0], Args.PatchSize[1], Args.PatchSize[2]), name='I2')
    
    Label1PH =  tf.placeholder(tf.float32, shape=(Args.MiniBatchSize, Args.PatchSize[0], Args.PatchSize[1], Args.NumOut), name='Label1')
    Label2PH =  tf.placeholder(tf.float32, shape=(Args.MiniBatchSize, Args.PatchSize[0], Args.PatchSize[1], Args.NumOut), name='Label2')
   
    TrainOperation(InputPH, I1PH, I2PH, Label1PH, Label2PH, Args)
    
    
if __name__ == '__main__':
    main()

