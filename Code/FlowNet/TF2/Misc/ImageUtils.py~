# TODO: Test ComposeReducedH function

import cv2
import numpy as np
import random
import skimage
import PIL
import sys
import tensorflow as tf
# Don't generate pyc codes
sys.dont_write_bytecode = True

def CenterCrop(I, OutShape):
    ImageSize = np.shape(I)
    CenterX = ImageSize[0]/2
    CenterY = ImageSize[1]/2
    try:
        ICrop = I[int(np.ceil(CenterX-OutShape[0]/2)):int(np.ceil(CenterX+OutShape[0]/2)),\
                  int(np.ceil(CenterY-OutShape[1]/2)):int(np.ceil(CenterY+OutShape[1]/2)), :]
    except:
        ICrop = None
    return ICrop

def CenterCropFactor(I, Factor):
    ImageSize = np.shape(I)
    CenterX = ImageSize[0]/2
    CenterY = ImageSize[1]/2
    OutShape = ImageSize - (np.mod(ImageSize,2**Factor))
    OutShape[2] = ImageSize[2]
    try:
        ICrop = I[int(np.ceil(CenterX-OutShape[0]/2)):int(np.ceil(CenterX+OutShape[0]/2)),\
                  int(np.ceil(CenterY-OutShape[1]/2)):int(np.ceil(CenterY+OutShape[1]/2)), :]
    except:
        ICrop = None
        OutShape = None
    return (ICrop, OutShape)

def RandomCrop(I1, OutShape):
    ImageSize = np.shape(I1)
    try:
        RandX = random.randint(0, ImageSize[0]-OutShape[0])
        RandY = random.randint(0, ImageSize[1]-OutShape[1])
        I1Crop = I1[RandX:RandX+OutShape[0], RandY:RandY+OutShape[1], :]
    except:
        I1Crop = None
    return (I1Crop)

def StackImages(I1, I2):
    return np.dstack((I1, I2))

def UnstackImages(I, NumChannels=3):
    return I[:,:,:NumChannels], I[:,:,NumChannels:]
    

def GaussianNoise(I1):
    IN1 = skimage.util.random_noise(I1, mode='gaussian', var=0.01)
    IN1 = np.uint8(IN1*255)
    return (IN1)

def ShiftHue(I1):
    IHSV1 = cv2.cvtColor(I1, cv2.COLOR_BGR2HSV)
    MaxShift = 30
    RandShift = random.randint(-MaxShift, MaxShift)
    IHSV1[:, :, 0] = IHSV1[:, :, 0] + RandShift
    IHSV1 = np.uint8(np.clip(IHSV1, 0, 255))
    return (cv2.cvtColor(IHSV1, cv2.COLOR_HSV2BGR))

def ShiftSat(I1):
    IHSV1 = cv2.cvtColor(I1, cv2.COLOR_BGR2HSV)
    MaxShift = 30
    RandShift = random.randint(-MaxShift, MaxShift)
    IHSV1 = np.int_(IHSV1)
    IHSV1[:, :, 1] = IHSV1[:, :, 1] + RandShift
    IHSV1 = np.uint8(np.clip(IHSV1, 0, 255))
    return (cv2.cvtColor(IHSV1, cv2.COLOR_HSV2BGR))

def Gamma(I1):
    MaxShift = 2.5
    RandShift = random.uniform(0, MaxShift)
    IG1 = skimage.exposure.adjust_gamma(I1, RandShift)
    return (IG1)

def Resize(I1, OutShape):
    ImageSize = np.shape(I1)
    I1Resize = cv2.resize(I1, (OutShape[0], OutShape[1]))
    return (I1Resize)

def StandardizeInputs(I):
    I /= 255.0
    I -= 0.5
    I *= 2.0
    return I

def StandardizeInputsTF(I):
    I = tf.math.multiply(tf.math.subtract(tf.math.divide(I, 255.0), 0.5), 2.0)
    return I

def HPFilter(I, Radius = 10):
    # Code adapted from: https://akshaysin.github.io/fourier_transform.html#.XSYBbnVKhhF
    if(len(np.shape(I)) == 3):
        I = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)

    F = cv2.dft(np.float32(I), flags=cv2.DFT_COMPLEX_OUTPUT)
    FShift = np.fft.fftshift(F)
       
    # Circular HPF mask, center circle is 0, remaining all ones
    Rows, Cols = I.shape
    Mask = np.ones((Rows, Cols, 2), np.uint8)
    Center = [int(Rows / 2), int(Cols / 2)]
    x, y = np.ogrid[:Rows, :Cols]
    MaskArea = (x - Center[0]) ** 2 + (y - Center[1]) ** 2 <= Radius**2
    Mask[MaskArea] = 0

    # Filter by Masking FFT Spectrum
    FShiftFilt = np.multiply(FShift, Mask)
    FFilt = np.fft.ifftshift(FShiftFilt)
    IFilt = cv2.idft(FFilt)
    IFilt = cv2.magnitude(IFilt[:, :, 0], IFilt[:, :, 1])
    return IFilt

def rgb2gray(rgb):
    # Code adapted from: https://stackoverflow.com/questions/12201577/how-can-i-convert-an-rgb-image-into-grayscale-in-python
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray

class Homography:
    def __init__(self, ImageSize=[128., 128., 3.], MaxR = np.zeros((1,3)), MaxT = np.array([[0.25], [0.25], [0.25]]), MaxYaw = 45.0, MaxMinScale = np.array([0.7, 1.4])):
        self.ImageSize = ImageSize
        self.ScaleMtrx = np.eye(3) # Scales from [-1, 1] ImageCoordinates to Actual Image Coordinates
        self.ScaleMtrx[0,0] = ImageSize[1]/2
        self.ScaleMtrx[0,2] = ImageSize[1]/2
        self.ScaleMtrx[1,1] = ImageSize[0]/2
        self.ScaleMtrx[1,2] = ImageSize[0]/2
        self.MaxR = MaxR # +- Degrees Euler Angles ZYX
        self.MaxT = MaxT # +- T in f units or px. if using 2D
        self.MaxYaw = MaxYaw # +- Degrees
        self.MaxMinScale = MaxMinScale # [Min, Max]        
        
    def ComposeHFromRTN(self, R = np.eye(3), T = np.zeros((3, 1)), N = np.array([[0.], [0.], [1.]]), ScaleToPx = False):
        H = np.add(R, np.matmul(T, N.T)) # R + TN'
        H = np.divide(H, H[2,2]) # Nornalize by making last element 1
        if(ScaleToPx):
            H =  np.matmul(self.ScaleMtrx, np.matmul(H, np.linalg.inv( self.ScaleMtrx))) # Scale to bring to Image Coordinates
           
        return H

    def ComposeReducedH(self, TransformType = ['Yaw', 'Scale', 'T2D'], T2D = np.zeros((2, 1)), Yaw = 0.0, Scale =  np.ones((2, 1)), Shear = np.zeros((2, 1)), ScaleToPx = False):
        # T2D is in px.
        # Yaw is in degrees
        # Scale is percentage of f, 1.0 gives original scale
        # Transformation order is always Yaw -> Scale -> Shear -> Translation
        # Notes from here: https://courses.cs.washington.edu/courses/csep576/11sp/pdf/Transformations.pdf
        def HFromYaw(Yaw):
            Yawr = np.radians(Yaw)
            cosYaw = np.cos(Yawr)
            sinYaw = np.sin(Yawr)
            HNow = np.eye(3)
            HNow[0,0] = cosYaw
            HNow[0,1] = -sinYaw
            HNow[1,0] = sinYaw
            HNow[1,1] = cosYaw
            return HNow
       
        def HFromScale(Scale):
            HNow = np.eye(3)
            HNow[0,0] = Scale[0]
            HNow[1,1] = Scale[1]
            return HNow

        def HFromShear(Shear):
            HNow = np.eye(3)
            HNow[0,1] = Shear[0]
            HNow[1,0] = Shear[1]
            return HNow

        def HFromTranslation2D(T2D):
            HNow = np.eye(3)
            HNow[0,2] = T2D[0]
            HNow[1,2] = T2D[1]
            return HNow


        # If TranformType is not list, make it into a list
        if(not isinstance(TransformType, list)):
            TransformType = list(TransformType)
            
        # Any combination composition is possible, list order does not matter
        H = np.eye(3)
        for count in TransformType:
            if 'Yaw' in TransformType:
                HNow = HFromYaw(Yaw)
                H = np.matmul(H, HNow)
            if 'Scale' in TransformType:
                HNow = HFromScale(Scale)
                H = np.matmul(H, HNow)
            if 'Shear' in TransformType:
                HNow = HFromShear(Shear)
                H = np.matmul(H, HNow)
            if 'T2D' in TransformType:
                HNow = HFromTranslation2D(T2D)
                H = np.matmul(H, HNow)

        if(ScaleToPx):
            H =  np.matmul(self.ScaleMtrx, np.matmul(H, np.linalg.inv(self.ScaleMtrx))) # Scale to bring to Image Coordinates
            
        return H 
        
    def DecomposeHToRTN(self):
        # retval, rotations, translations, normals   =  cv.decomposeHomographyMat(H, K[, rotations[, translations[, normals]]])
        pass
    
    def WarpImg(self, I, H, Disp=False, DispName='WarpedImg', WaitTime=0):
        WarpedImg = cv2.warpPerspective(I, HInv, (ImageSize[1],ImageSize[0]))
        if(Disp):
            cv2.imshow(DispName, WarpedImg)
            cv2.waitKey(WaitTime)
        return WarpedImg
    
    def WarpPtsUsingHomography(Pts, H, AddOffset=None):
        PerturbPts = []
        for pt in Pts:
            # Apply Homography
            PerturbPtsNow = np.matmul(H, [[pt[0]], [pt[1]], [1.0]])
            # Normalize to be on Image Plane
            PerturbPtsNow = np.divide(PerturbPtsNow, PerturbPtsNow[2])[:2]
            # Add offset if needed
            if(AddOffset is not None):
                PerturbPtsNow = np.add(PerturbPtsNow,  AddOffset)
                PerturbPts.append(PerturbPtsNow)
        return PerturbPts

    def DispWarpLines(self, I, Pts, Disp=True,  DispName='HomographyLines', ColorSpec=(255,255,255), WaitTime=0):
        ImgDisp = I.copy()
        cv2.polylines(ImgDisp, [np.int32(Pts)], 1, ColorSpec)
        if(Disp is True):
            cv2.imshow(ImgTitle, ImgDisp)
            cv2.waitKey(WaitTime)
        return ImgDisp

    def GetRandR(self, MaxR = None, EulOrder='zyx'):
        # MaxR is given in Degrees
        if MaxR is not None:
            # Overwrite value
            self.MaxR = np.radians(MaxR)
        # Generate random value of euler angles
        EulAng = 2*self.MaxR*([np.random.rand() - 0.5, np.random.rand() - 0.5, np.random.rand() - 0.5])
        R = Rot.from_euler(EulOrder, EulAng, degrees=True).as_dcm()
        return R, EulAng

    def GetRandT(self, MaxT = None, Flag2D = False):
        # MaxT is given in Percentage of focal length (0.1 means 0.1f)
        if MaxT is not None:
            # Overwrite value
            self.MaxT = MaxT
        # Generate random value of translation
        if(not Flag2D):
            T = np.array(4*self.MaxT*([[np.random.rand() - 0.5],[np.random.rand() - 0.5],[np.random.rand() - 0.5]]))  # 2x2, 2 for ImageSize/2 and 2 for rand scaling
        else:
            self.MaxT = self.MaxT[0:2] # Extract first two elements if three are given
            T = np.array(4*self.MaxT*([[np.random.rand() - 0.5],[np.random.rand() - 0.5]])) # 2x2, 2 for ImageSize/2 and 2 for rand scaling
        return T

    def GetRandYaw(self, MaxYaw = None):
        if MaxYaw is not None:
            # Overwrite value
            self.MaxYaw = MaxYaw
        return 2*self.MaxYaw*(np.random.rand() - 0.5)

    def GetRandScale(self, MaxMinScale = None, Uniform = True):
        if MaxMinScale is not None:
            # Overwrite value
            self.MaxMinScale = MaxMinScale
        if Uniform:
            Scale = (self.MaxMinScale[1] - self.MaxMinScale[0])*np.random.rand() + self.MaxMinScale[0]
            Scale = np.tile(Scale, [2,1])
        else:
            Scale = [[(self.MaxMinScale[1] - self.MaxMinScale[0])*np.random.rand() + self.MaxMinScale[0]],\
                     [(self.MaxMinScale[1] - self.MaxMinScale[0])*np.random.rand() + self.MaxMinScale[0]]]
        return Scale

    def GetRandReducedH(self, TransformType = ['Yaw', 'Scale', 'T2D'], MaxT = None, MaxYaw = None, MaxMinScale = None, ScaleToPx = False):
        H = self.ComposeReducedH(TransformType, T2D = self.GetRandT(MaxT, True), Yaw = self.GetRandYaw(MaxYaw), Scale = self.GetRandScale(MaxMinScale, True), ScaleToPx = ScaleToPx)
        return H
                            
        
    

            
        
        
    
