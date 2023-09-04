#!/usr/bin/env python


import argparse
import re
import numpy as np
import matplotlib.pyplot as plt

def main(ReadPath):
    SkipLines = 8
    Count = 0
    File = open(ReadPath, 'r')

    Delimiters = ",", "\n"
    RegexPattern = '|'.join(map(re.escape, Delimiters))
    ErrorL1Px = []
    ErrorL2Px = []    

    
    for Count, Line in enumerate(File):
        if(Count > SkipLines):
            LineSplit = re.split(RegexPattern, Line)
            ErrorL1Px.append(float(LineSplit[1]))
            ErrorL2Px.append(float(LineSplit[2]))


    print('ErrorL1Px. Median {} Px'.format(np.median(ErrorL1Px)))
    print('ErrorL1Px. Mean {} Px'.format(np.mean(ErrorL1Px)))

    print('ErrorL2Px. Median {} Px'.format(np.median(ErrorL2Px)))
    print('ErrorL2Px. Mean {} Px'.format(np.mean(ErrorL2Px)))

    fig, (ax1) = plt.subplots(1, 1, sharex=True)
    ax1.plot(ErrorL1Px, 'b.')
    ax1.plot(ErrorL2Px, 'r.')
    plt.show()

if __name__=="__main__":
    Parser = argparse.ArgumentParser()
    Parser.add_argument('--ReadPath', dest='ReadPath', default='/home/nitin/VariableBaseLineStereo/Trained/ResNetL1/PredOuts.txt',\
                         help='Path to load Predictions from, Default:/home/nitin/VariableBaseLineStereo/Trained/ResNetL1/PredOuts.txt')

                        
    Args = Parser.parse_args()
    ReadPath = Args.ReadPath
    
    main(ReadPath)
