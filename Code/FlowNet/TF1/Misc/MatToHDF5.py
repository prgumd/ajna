#!/usr/bin/env python


# Dependencies:
# opencv, do (pip install opencv-python)
# termcolor, do (pip install termcolor)
# tqdm, do (pip install tqdm)

import glob
import os
from termcolor import colored, cprint
import argparse
import cv2
import numpy as np
from tqdm import tqdm
import scipy.io as sio
import random
import re
import h5py
import ImageUtils as iu
import MiscUtils as mu


def Mat2HDF5(ReadPath, WritePath):
    for dirs in tqdm(glob.glob(ReadPath + os.sep + '*' + '.mat')):
        Timer1 = mu.tic()
        Heatmap = sio.loadmat(dirs)['heatmap']
        print(mu.toc(Timer1))
        Delimiters = '.mat', os.sep
        RegexPattern = '|'.join(map(re.escape, Delimiters))
        WriteName = WritePath + os.sep + re.split(RegexPattern, dirs)[-2] + '.h5'
        Hf = h5py.File(WriteName, 'w')
        Hf.create_dataset('heatmap', data=Heatmap)
        Hf.close()
        Timer2 = mu.tic()
        hf = np.array(h5py.File(WriteName, 'r').get('heatmap')) # SLOWER by a factor of 10, 0.000406980514526 and 0.0044801235199 s respectively
        print(mu.toc(Timer2))
        input('q')       

def main():
    # Parse Command Line arguments
    Parser = argparse.ArgumentParser()
    Parser.add_argument('--ReadPath', default='/media/analogicalnexus/00EA777C1E864BA9/2018/EVDodge/dataset',
                        help='Base path of images, Default:/media/analogicalnexus/00EA777C1E864BA9/2018/EVDodge/dataset')
    Parser.add_argument('--WritePath', default='/media/analogicalnexus/00EA777C1E864BA9/2018/EVDodge/processed',
                        help='Base path of images, Default:/media/analogicalnexus/00EA777C1E864BA9/2018/EVDodge/processed')
    
    
    Args = Parser.parse_args()
    ReadPath = Args.ReadPath
    WritePath = Args.WritePath
    
    if(not os.path.exists(WritePath)):
        cprint("WARNING: %s doesnt exist, Creating it."%WritePath, 'yellow')
        os.mkdir(WritePath)
        
    Mat2HDF5(ReadPath, WritePath)
    
if __name__ == '__main__':
    main()

    
