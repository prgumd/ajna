import random
import os
import cv2
import numpy as np
import tensorflow as tf
import Misc.ImageUtils as iu
import Misc.warpICSTN2 as warp2
import Misc.MiscUtils as mu
import scipy.io as sio

class BatchGeneration():
    def __init__(self, sess, WarpI1PatchIdealGen, IOrgPH, HPH, SuperPointFlag = False):
        self.sess = sess
        self.WarpI1PatchIdealGen = WarpI1PatchIdealGen
        self.IOrgPH = IOrgPH
        self.HPH = HPH
        self.SuperPointFlag = SuperPointFlag


    def ReadSuperPointCornerness(self, PicklePath, Negate=True, Vis=False):
        # Cornerness = pickle.load(open(PicklePath, 'rb'))
        Cornerness = sio.loadmat(PicklePath)['heatmap']

        if(Negate is True):
            Cornerness = 1.0 - Cornerness

        if(Vis):
            # Jet colormap for visualization.
            myjet = np.array([[0.        , 0.        , 0.5       ],
                              [0.        , 0.        , 0.99910873],
                              [0.        , 0.37843137, 1.        ],
                              [0.        , 0.83333333, 1.        ],
                              [0.30044276, 1.        , 0.66729918],
                              [0.66729918, 1.        , 0.30044276],
                              [1.        , 0.90123457, 0.        ],
                              [1.        , 0.48002905, 0.        ],
                              [0.99910873, 0.07334786, 0.        ],
                              [0.5       , 0.        , 0.        ]])
            CornernessDisp = myjet[np.round(np.clip(Cornerness*10, 0, 9)).astype('int'), :]
            CornernessDisp = (CornernessDisp*255).astype('uint8')
            cv2.imshow('Cornerness', CornernessDisp)
            cv2.waitKey(0)
        return Cornerness
        
    def RandSimilarityPerturbationTF(self, I1, HObj, PatchSize, MiniBatchSize, Cornerness1Batch, ImageSize=None, Vis=False):
        if(ImageSize is None):
            ImageSize = np.array(np.shape(I1))[1:]
            # TODO: Extract MiniBatchSize here

        H, Params = HObj.GetRandReducedHICSTN()

        # Maybe there is a better way? https://dominikschmidt.xyz/tensorflow-data-pipeline/
        FeedDict = {self.IOrgPH: I1, self.HPH: H}
        I2 = np.uint8(self.sess.run([self.WarpI1PatchIdealGen], feed_dict=FeedDict)[0]) # self.WarpI1PatchIdealGen.eval(feed_dict=FeedDict)
        if(Cornerness1Batch is not None):
            Cornerness1Batch = (np.array(Cornerness1Batch)*255).astype('uint8')
            FeedDict = {self.IOrgPH: Cornerness1Batch, self.HPH: H}
            Cornerness2Batch = np.uint8(self.sess.run([self.WarpI1PatchIdealGen], feed_dict=FeedDict)[0]) # self.WarpI1PatchIdealGen.eval(feed_dict=FeedDict)
        
        # Crop in center for PatchSize
        P1 = iu.CenterCrop(I1, PatchSize)
        P2 = iu.CenterCrop(I2, PatchSize)
        if(Cornerness1Batch is not None):
            C1 = iu.CenterCrop(np.array(Cornerness1Batch), PatchSize)
            C2 = iu.CenterCrop(np.array(Cornerness2Batch), PatchSize)
        else:
            C1 = None
            C2 = None

        if(Vis is True):
            for count in range(MiniBatchSize):
                A = (I1[count]).astype('uint8')
                B = (I2[count]).astype('uint8')
                AP = (P1[count]).astype('uint8')
                BP = (P2[count]).astype('uint8')
                cv2.imshow('I1, I2', np.hstack((A, B)))
                cv2.imshow('P1, P2', np.hstack((AP, BP)))
                if(Cornerness1Batch is not None):
                    CA =  Cornerness1Batch[count]
                    CB =  Cornerness2Batch[count]
                    CAP = C1[count]
                    CBP = C2[count]
                    cv2.imshow('C1, C2', np.hstack((CA, CB)))
                    cv2.imshow('CP1, CP2', np.hstack((CAP, CBP)))
                cv2.waitKey(0)

        # P1 is I1 cropped to patch Size
        # P2 is I1 Crop Warped (I2 Crop)
        # H is Homography
        # Params is the stuff H is made from 
        return I1, I2, P1, P2, C1, C2, H, Params


    def GenerateBatchTF(self, TrainNames, PatchSize, MiniBatchSize, HObj, BasePath, OriginalImageSize, Args, da, DataAugGen):
        """
        Inputs: 
        DirNames - Full path to all image files without extension
        NOTE that Train can be replaced by Val/Test for generating batch corresponding to validation (held-out testing in this case)/testing
        TrainLabels - Labels corresponding to Train
        NOTE that TrainLabels can be replaced by Val/TestLabels for generating batch corresponding to validation (held-out testing in this case)/testing
        ImageSize - Size of the Image
        MiniBatchSize is the size of the MiniBatch
        Outputs:
        I1Batch - Batch of I1 images after standardization and cropping/resizing to ImageSize
        HomeVecBatch - Batch of Homing Vector labels
        """
        
        IOrgBatch = []
        if(self.SuperPointFlag):
            Cornerness1Batch = []
        else:
            Cornerness1Batch = None
        
        ImageNum = 0
        while ImageNum < MiniBatchSize:
            # Generate random image
            RandIdx = random.randint(0, len(TrainNames)-1)
            RandImageName = BasePath + os.sep + TrainNames[RandIdx]
            I = cv2.imread(RandImageName)
            if(self.SuperPointFlag):
                RandPicklePath = BasePath + 'Pickle' + os.sep + TrainNames[RandIdx][:-4] + '.mat'
                Cornerness = self.ReadSuperPointCornerness(RandPicklePath, Negate=True, Vis=False)
                Stack = iu.StackImages(I, Cornerness)
                I = iu.RandomCrop(Stack, OriginalImageSize)
            else:
                I = iu.RandomCrop(I, OriginalImageSize)
            if (I is None):
                continue
            try:
                I, Cornerness = iu.UnstackImages(I)
                Cornerness = np.tile(Cornerness, [1, 1, 3])
            except:
                continue
            ImageNum += 1
            IOrgBatch.append(I)
            if(self.SuperPointFlag):
                Cornerness1Batch.append(Cornerness)

        # Cast Lists as np arrays
        IOrgBatch = np.array(IOrgBatch)
        # Similarity and Patch generation 
        I1Batch, I2Batch, P1Batch, P2Batch, C1Batch, C2Batch, HBatch, ParamsBatch = \
            self.RandSimilarityPerturbationTF(IOrgBatch, HObj, PatchSize, MiniBatchSize, Cornerness1Batch, ImageSize = None, Vis = False)

        # Augment Data if asked for
        if(Args.DataAug):
            FeedDict = {da.ImgPH: P1Batch}
            P1Batch = np.uint8(da.sess.run([DataAugGen], feed_dict=FeedDict)[0])
            FeedDict = {da.ImgPH: P2Batch}
            P2Batch = np.uint8(da.sess.run([DataAugGen], feed_dict=FeedDict)[0])
            
        if(Args.Input == 'G'):
            P1Batch = np.tile(iu.rgb2gray(P1Batch)[:,:,:,np.newaxis], (1,1,1,3))
            P2Batch = np.tile(iu.rgb2gray(P2Batch)[:,:,:,np.newaxis], (1,1,1,3))
        elif(Args.Input == 'HP'):
            P1Batch = iu.HPFilterBatch(P1Batch)
            P2Batch = iu.HPFilterBatch(P2Batch)
        elif(Args.Input == 'SP'):
            P1Batch = C1Batch
            P2Batch = C2Batch
        elif(Args.Input == 'I'):
            pass
        else:
            print('ERROR: Unrecognized Input Type ')
            os.exit()
        ICombined = np.concatenate((P1Batch, P2Batch), axis=3)
        
        # Normalize Dataset
        # https://stackoverflow.com/questions/42275815/should-i-substract-imagenet-pretrained-inception-v3-model-mean-value-at-inceptio
        IBatch = iu.StandardizeInputs(np.float32(ICombined))

        return IBatch, I1Batch, I2Batch, P1Batch, P2Batch, C1Batch, C2Batch, HBatch, ParamsBatch

