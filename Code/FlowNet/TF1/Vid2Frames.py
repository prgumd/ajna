#!/usr/bin/env python3

# Dependencies:
# opencv, do (pip install opencv-python)
# skimage, do (sudo)
# termcolor, do (pip install termcolor)
# tqdm, do (pip install tqdm)


# TODO: Adapt more augmentation from: https://github.com/sthalles/deeplab_v3/blob/master/preprocessing/inception_preprocessing.py

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
# Import of network is done in main code
import importlib
from datetime import datetime
import getpass
import copy
import platform


def Vid2Frames(Args):
    cap = cv2.VideoCapture(Args.VidPath)

    # Check if camera opened successfully
    if (cap.isOpened()== False):
        print("Error opening video stream or file")

    count = 0
    # Read until video is completed
    while(cap.isOpened()):
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret == True:
            # Display the resulting frame
            frame = cv2.resize(frame, (Args.ImgSize[0], Args.ImgSize[1]))
            cv2.imshow('Frame', frame)
            FileName = Args.WritePath + 'Frame%04d.png'%count
            cv2.imwrite(FileName, frame)
            count += 1
            # Press Q on keyboard to  exit
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
            # Break the loop
        else:
            break
    # When everything done, release the video capture object
    cap.release()




def main():
    """
    Inputs: 
    None
    Outputs:
    Runs the Training and testing code based on the Flag
    """
    # Parse Command Line arguments
    Parser = argparse.ArgumentParser()
    Parser.add_argument('--VidPath', default='Vid.mp4', help='Path to the video, Default:Vid.mp4')
    Parser.add_argument('--WritePath', default='/home/nitin/BlenderScenes/Frames4/Frames3/', help='Path to save frames, Default:/home/nitin/BlenderScenes/Frames4/Frames3/')
    Parser.add_argument('--ImgSize', default='[640,480,3]', help='Output Image Size as list, Default: [640,480,3]')

    # Input Size
    Args = Parser.parse_args()

    Args.ImgSize = Args.ImgSize.replace('[', ' ').replace(']', ' ').replace(',', ' ').split()
    Args.ImgSize = np.array([int(i) for i in Args.ImgSize])

    if(not (os.path.isdir(Args.WritePath))):
        os.makedirs(Args.WritePath)

    Vid2Frames(Args)
    
    
    
    
    Args = Parser.parse_args()

if __name__ == '__main__':
    main()
