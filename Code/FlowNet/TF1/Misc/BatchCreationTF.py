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

    def GenerateBatchTF(self, Args, da, DataAugGen):
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
        Label2Batch = []
        
        ImageNum = 0
        while ImageNum < Args.MiniBatchSize:
            # Generate random image
            RandIdx = random.randint(0, len(Args.TrainNames)-1)
            RandImageName1 = Args.BasePath + os.sep + Args.TrainNames[RandIdx]

            if(Args.Dataset == 'FC2'):
                RandImageName2 = RandImageName1.replace('img_0', 'img_1')
                
                LabelName1 = RandImageName1.replace('img_0.png', 'flow_01.flo')
                LabelName2 = RandImageName1.replace('img_0.png', 'flow_10.flo')
            elif(Args.Dataset == 'FT3D'):
                # Increment Image Number by 1
                ImgName = RandImageName1.rsplit('/', 1)[-1]
                ImgName = ImgName.split('.png')[0]

                RandImageName2 = RandImageName1.rsplit('/', 1)[0] + '/%04d'%(int(ImgName)+1) + '.png'
                
                LabelName1 = RandImageName1.replace(Args.BasePath, Args.LabelBasePath)
                LabelName1 = LabelName1.replace('frames_cleanpass', 'optical_flow')
                LabelName1 = LabelName1.replace('left', 'into_future/left')
                LabelName1 = LabelName1.replace(RandImageName1.rsplit('/', 1)[-1], 'OpticalFlowIntoFuture_' + RandImageName1.rsplit('/', 1)[-1])
                LabelName1 = LabelName1.replace('.png', '_L.pfm')
                
                LabelName2 = LabelName1.replace('OpticalFlowIntoFuture_', 'OpticalFlowIntoPast_')
                LabelName2 = LabelName2.replace('into_future', 'into_past')

            I1 = cv2.imread(RandImageName1)           
            I2 = cv2.imread(RandImageName2)

            if(I1 is None or I2 is None):
                continue
            
            try:
                if(Args.Dataset == 'FC2' or Args.Dataset == 'FT3D'):
                    Label1 = mu.readFlow(LabelName1)
                    Label2 = mu.readFlow(LabelName2)                   
            except:
                continue

            if(not Args.Quant):
                MaxScale = 2.0 # 25.0 # 25.0 # 1.0
                
                Label1 = np.divide(Label1, MaxScale)
                Label2 = np.divide(Label2, MaxScale)
            else:
                # Fails if no input range
                Label1 = iu.remap(np.float32(Label1), 0., 255.)
                Label2 = iu.remap(np.float32(Label2), 0., 255.)

            try:
                I = np.concatenate((I1, I2, Label1, Label2), axis=2)
                IOrg = np.concatenate((I1, I2), axis=2)
            except:
                continue
            
            I = iu.RandomCrop(I, Args.PatchSize)

            if (I is None):
                continue

            P1 = I[:,:,:3]
            P2 = I[:,:,3:6]
            I1 = IOrg[:,:,:3]
            I2 = IOrg[:,:,3:6]
            Label1 = I[:,:,6:8]
            Label2 = I[:,:,8:10]
                
            
            ImageNum += 1
            IOrgBatch.append(I)
            I1Batch.append(I1)
            I2Batch.append(I2)
            P1Batch.append(P1)
            P2Batch.append(P2)
            Label1Batch.append(Label1)
            Label2Batch.append(Label2)

            
        
        # Augment Data if asked for
        if(Args.DataAug):
            FeedDict = {da.ImgPH: P1Batch}
            P1Batch = np.uint8(da.sess.run([DataAugGen], feed_dict=FeedDict)[0])
            FeedDict = {da.ImgPH: P2Batch}
            P2Batch = np.uint8(da.sess.run([DataAugGen], feed_dict=FeedDict)[0])

            
        ICombined = np.concatenate((P1Batch, P2Batch), axis=3)
        
        # Normalize Dataset
        # https://stackoverflow.com/questions/42275815/should-i-substract-imagenet-pretrained-inception-v3-model-mean-value-at-inceptio
        if(not Args.Quant):
            IBatch = iu.StandardizeInputs(np.float32(ICombined)) # np.float32(ICombined)
        else:
            IBatch = np.float32(ICombined)
        # Label1Batch = iu.StandardizeInputs(np.float32(Label1Batch))

        return IBatch, I1Batch, I2Batch, P1Batch, P2Batch, Label1Batch, Label2Batch
