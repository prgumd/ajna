import random
import os
import cv2
import numpy as np
import tensorflow as tf
import Misc.ImageUtils as iu
import Misc.warpICSTN2 as warp2
import Misc.MiscUtils as mu
import scipy.io as sio
import imageio 

class BatchGeneration():
    def __init__(self, sess, IOrgPH):
        self.sess = sess
        self.IOrgPH = IOrgPH

    def GenerateBatchTF(self, Img1, Img2, GT1, Args, da, DataAugGen):
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
        I1Batch = []
        I2Batch = []
        P1Batch = []
        P2Batch = []
        Label1Batch = []

        I1 = cv2.imread(Img1)           
        I2 = cv2.imread(Img2)
        
        if(GT1 != 'None'):
            try:
                Label1 = mu.readFlow(GT1)
            except:
                Label1 = np.zeros((np.shape(I1)[0], np.shape(I1)[1], 2))
                
            MaxScale = 1.0
            Label1 = np.divide(Label1, MaxScale)
        else:
            Label1 = np.zeros((np.shape(I1)[0], np.shape(I1)[1], 2))

        I = np.concatenate((I1, I2, Label1), axis=2)
        IOrg = np.concatenate((I1, I2), axis=2)

        I = iu.CenterCrop(I, Args.PatchSize)
        if(Args.Attack):
            AttackPatch = cv2.imread(Args.AttackPath)
            AttackPatch = cv2.resize(AttackPatch, (Args.AttackSize, Args.AttackSize))
            ImageSize = np.shape(I)
            CenterX = ImageSize[0]/2
            CenterY = ImageSize[1]/2
            Shift = 20.
            Mask = np.uint8((AttackPatch[:,:,0]==127) & (AttackPatch[:,:,1]==127) & (AttackPatch[:,:,2]==127)) # Remove Gray pixels
            Mask = np.float32(np.tile(Mask[:,:,np.newaxis], (1,1,3)))
            I[int(np.ceil(CenterX-Args.AttackSize/2)):int(np.ceil(CenterX+Args.AttackSize/2)),\
                  int(np.ceil(CenterY-Args.AttackSize/2)):int(np.ceil(CenterY+Args.AttackSize/2)), 0:3] = I[int(np.ceil(CenterX-Args.AttackSize/2)):int(np.ceil(CenterX+Args.AttackSize/2)),\
                  int(np.ceil(CenterY-Args.AttackSize/2)):int(np.ceil(CenterY+Args.AttackSize/2)), 0:3]*Mask + AttackPatch*(1.-Mask)
            I[int(np.ceil(CenterX-Args.AttackSize/2)):int(np.ceil(CenterX+Args.AttackSize/2)),\
                  int(np.ceil(CenterY-Args.AttackSize/2)):int(np.ceil(CenterY+Args.AttackSize/2)), 3:6] = I[int(np.ceil(CenterX-Args.AttackSize/2)):int(np.ceil(CenterX+Args.AttackSize/2)),\
                  int(np.ceil(CenterY-Args.AttackSize/2)):int(np.ceil(CenterY+Args.AttackSize/2)), 3:6]*Mask + AttackPatch*(1.-Mask)

            # cv2.imwrite('/home/nitin/Datasets/FlowAttack/1.png', I[:,:,:3])
            # cv2.imwrite('/home/nitin/Datasets/FlowAttack/2.png', I[:,:,3:6])
            # input('q')
        

        P1 = I[:,:,:3]
        P2 = I[:,:,3:6]
        I1 = IOrg[:,:,:3]
        I2 = IOrg[:,:,3:6]
        Label1 = I[:,:,6:8]
            
        IOrgBatch.append(I)
        I1Batch.append(I1)
        I2Batch.append(I2)
        P1Batch.append(P1)
        P2Batch.append(P2)
        Label1Batch.append(Label1)
      
        
        # Augment Data if asked for
        if(Args.DataAug):
            FeedDict = {da.ImgPH: P1Batch}
            P1Batch = np.uint8(da.sess.run([DataAugGen], feed_dict=FeedDict)[0])
            FeedDict = {da.ImgPH: P2Batch}
            P2Batch = np.uint8(da.sess.run([DataAugGen], feed_dict=FeedDict)[0])

            
        ICombined = np.concatenate((P1Batch, P2Batch), axis=3)
        
        # Normalize Dataset
        # https://stackoverflow.com/questions/42275815/should-i-substract-imagenet-pretrained-inception-v3-model-mean-value-at-inceptio
        IBatch = iu.StandardizeInputs(np.float32(ICombined))
        # Label1Batch = iu.StandardizeInputs(np.float32(Label1Batch))

        return IBatch, I1Batch, I2Batch, P1Batch, P2Batch, Label1Batch

