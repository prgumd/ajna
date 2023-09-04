import Misc.ImageUtils as iu
import os
import numpy as np

def SetupAll(BasePath, LearningRate, MiniBatchSize, warpType = None):
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
    DirNamesPath = BasePath + os.sep + 'DirNames.txt'
    # LabelNamesPath = BasePath + os.sep + 'Labels.txt'
    TrainPath = BasePath + os.sep + 'Train.txt'
    ValPath = BasePath + os.sep + 'Val.txt'
    TestPath = BasePath + os.sep + 'Test.txt'
    DirNames, TrainNames, ValNames, TestNames=\
              ReadDirNames(DirNamesPath, TrainPath, ValPath, TestPath)


    # Setup Neural Net Params
    # List of all OptimizerParams: depends on Optimizer
    # For ADAM Optimizer: [LearningRate, Beta1, Beta2, Epsilion]
    UseDefaultFlag = 0 # Set to 0 to use your own params, do not change default parameters
    if UseDefaultFlag:
        # Default Parameters
        OptimizerParams = [1e-3, 0.9, 0.999, 1e-8]
    else:
        # Custom Parameters
        OptimizerParams = [LearningRate, 0.9, 0.999, 1e-8]   
        
    # Save checkpoint every SaveCheckPoint iteration in every epoch, checkpoint saved automatically after every epoch
    SaveCheckPoint = 1000 
    # Number of passes of Val data with MiniBatchSize 
    NumTestRunsPerEpoch = 5
    
    # Image Input Shape
    OriginalImageSize = np.array([300, 300, 3])
    PatchSize = np.array([128, 128, 3])
    NumTrainSamples = len(TrainNames)
    NumValSamples = len(ValNames)
    NumTestSamples = len(TestNames)
    # Warp Parameters
    if(warpType is None):
        warpType = ['scale', 'scale', 'translation', 'translation'] # ['pseudosimilarity', 'pseudosimilarity', 'pseudosimilarity', 'pseudosimilarity'] # ['scale', 'scale', 'translation', 'translation'] # ['pseudosimilarity', 'pseudosimilarity']
    # Homography Perturbation Parameters
    MaxParams = np.array([0.25, 0.2, 0.2])
    # MODIFY THIS DEPENDING ON ARCH!
    # HObj = iu.HomographyICTSN(MaxParams = MaxParams, TransformType = warpType[-1], MiniBatchSize = MiniBatchSize)
    HObj = iu.HomographyICTSN(MaxParams = MaxParams, TransformType = 'pseudosimilarity', MiniBatchSize = MiniBatchSize)

    return TrainNames, ValNames, TestNames, OptimizerParams,\
        SaveCheckPoint, PatchSize, NumTrainSamples, NumValSamples, NumTestSamples,\
        NumTestRunsPerEpoch, OriginalImageSize, HObj, warpType      


def ReadDirNames(DirNamesPath, TrainPath, ValPath, TestPath):
    """
    Inputs: 
    Path is the path of the file you want to read
    Outputs:
    DirNames is the data loaded from ./TxtFiles/DirNames.txt which has full path to all image files without extension
    """
    # Read DirNames and LabelNames files
    DirNames = open(DirNamesPath, 'r')
    DirNames = DirNames.read()
    DirNames = DirNames.split()
    
    # Read Train, Val and Test Idxs
    TrainIdxs = open(TrainPath, 'r')
    TrainIdxs = TrainIdxs.read()
    TrainIdxs = TrainIdxs.split()
    TrainIdxs = [int(val) for val in TrainIdxs]
    TrainNames = [DirNames[i] for i in TrainIdxs]
    # TrainLabels = [LabelNames[i] for i in TrainIdxs]

    ValIdxs = open(ValPath, 'r')
    ValIdxs = ValIdxs.read()
    ValIdxs = ValIdxs.split()
    ValIdxs = [int(val) for val in ValIdxs]
    ValNames = [DirNames[i] for i in ValIdxs]
    # ValLabels = [LabelNames[i] for i in ValIdxs]

    TestIdxs = open(TestPath, 'r')
    TestIdxs = TestIdxs.read()
    TestIdxs = TestIdxs.split()
    TestIdxs = [int(val) for val in TestIdxs]
    TestNames = [DirNames[i] for i in TestIdxs]
    # TestLabels = [LabelNames[i] for i in TestIdxs]

    return DirNames, TrainNames, ValNames, TestNames
