#!/usr/bin/env python3

import Misc.MiscUtils as mu
import random
import os
import cv2
import numpy as np
import tensorflow as tf
import Misc.ImageUtils as iu
import Misc.MiscUtils as mu
import scipy.io as sio
import imageio
import argparse


def main():
    """
    Inputs: 
    None
    Outputs:
    Runs the Training and testing code based on the Flag
    """
    # Parse Command Line arguments
    Parser = argparse.ArgumentParser()
    Parser.add_argument('--ReadPath', default='/media/nitin/Research2/FlyingThings3D/flyingthings3d_optical_flow/optical_flow/TRAIN/A/0000/into_future/left/OpticalFlowIntoFuture_0006_L.pfm', help='Flow read path, Default:/media/nitin/Research2/FlyingThings3D/flyingthings3d_optical_flow/optical_flow/TRAIN/A/0000/into_future/left/OpticalFlowIntoFuture_0006_L.pfm')
    Parser.add_argument('--NumBins', default=0, type=int, help='Number of Bins, Default: 0 (No discretization, sets it to 255 for uint8)')
    Parser.add_argument('--NumMagBins', default=0, type=int, help='Number of Bins, Default: 0 (No discretization, sets it to 255 for uint8)')
    Parser.add_argument('--NumAngBins', default=0, type=int, help='Number of Bins, Default: 0 (No discretization, sets it to 255 for uint8)')
    # TODO: Set Mag and Ang Bins
    
    Args = Parser.parse_args()
    
    if(Args.NumBins==0):
        Args.NumBins=255.

    Label1 = np.float32(mu.readFlow(Args.ReadPath))
    # Scale = 2.5
    # Label1 = Label1/Scale
    # print(np.amin(Label1))
    # print(np.amax(Label1))
    # A = np.float32(np.digitize(Label1, np.linspace(np.amin(Label1), np.amax(Label1), Args.NumMagBins)))
    # EachMagBinRes = 255./Args.NumMagBins
    # A *= EachMagBinRes
    # print(EachMagBinRes)
    # print(A)
    # A = np.floor(A)
    # print(np.amax(A))
    # print(np.amin(A))
    # input('q')
    # Label1 += MinVal
    # MinVal = -127.
    # Label1 = np.clip(Label1, MinVal, 127.)
    # Label1 = iu.remap(np.floor(iu.remap(Label1, 0., Args.NumBins)), 0., 255.)
    # Label1Mag = np.sqrt(Label1[:,:,0]**2 + Label1[:,:,1]**2)
    # Label1Ang = np.arctan2(Label1[:,:,1], Label1[:,:,0]) # In Radians #*180./np.pi
    # MagDR = np.amax(Label1Mag) - np.amin(Label1Mag)
    # Label1MagQuant = iu.remap(Label1Mag, 0., 255.)
    # Label1MagQuant = np.float32(np.digitize(Label1MagQuant, np.linspace(0., 255., Args.NumMagBins)))
    # TODO: Fix Mag Quant to use NumBins instead of just scale values which are divided
    # Label1MagQuant = np.floor(Label1Mag/(MagDR/Args.NumMagBins))*(MagDR/Args.NumMagBins)
    # Label1AngQuant = np.floor(Label1Ang/(2*np.pi/Args.NumAngBins))*(2*np.pi/Args.NumAngBins)
    # Label1Quant = np.concatenate((Label1MagQuant[:,:,np.newaxis]*np.cos(Label1AngQuant[:,:,np.newaxis]),\
    #                               Label1MagQuant[:,:,np.newaxis]*np.sin(Label1AngQuant[:,:,np.newaxis])), axis=2)
    # Label1Quant = iu.remap(Label1Quant, 0., 255.)
    Label1Disp = iu.remap(Label1, 0., 255.)
    
    # print(np.amin(Label1MagQuant))
    # print(np.amax(Label1MagQuant))
    # print(np.amin(Label1AngQuant))
    # print(np.amax(Label1AngQuant))
    # Label1 = np.floor(Label1)
    
    cv2.imshow('FlowX, FlowY', np.hstack((np.uint8(Label1Disp[:,:,0]), np.uint8(Label1Disp[:,:,1]))))
    # cv2.imshow('QuantFlowX, QuantFlowY', np.hstack((np.uint8(Label1Quant[:,:,0]), np.uint8(Label1Quant[:,:,1]))))
    # cv2.imshow('DiffX, DiffY', np.abs(np.hstack((np.uint8(Label1Quant[:,:,0]), np.uint8(Label1Quant[:,:,1])))-np.hstack((np.uint8(Label1Disp[:,:,0]), np.uint8(Label1Disp[:,:,1])))))
    cv2.waitKey(0)
    
    
if __name__ == '__main__':
    main()
