import numpy as np
import scipy.linalg
import tensorflow as tf


class Options:
    def __init__(self, PatchSize=[128,128,3], MiniBatchSize=32, warpType='homography', NumBlocks=4, pertScale=0.25, transScale=0.25, AddTranslation=False):
        self.W = PatchSize[0].astype(np.int32) # PatchSize is Width, Height, NumChannels
        self.H = PatchSize[1].astype(np.int32) 
        self.batchSize = np.array(MiniBatchSize).astype(np.int32)
        self.NumBlocks = NumBlocks
        self.warpType = warpType
        if(isinstance(self.warpType, list)): # If you don't need different warps, send single string for warpType instead
            # Also update NumBlocks here
            self.NumBlocks = len(self.warpType)
            self.warpDim = []
            for val in self.warpType:
                if val == 'yaw':
                    self.warpDim.append(1)
                if val == 'scale':
                    self.warpDim.append(1)
                if val == 'translation':
                    self.warpDim.append(2)
                elif val == 'similarity':
                    self.warpDim.append(4)
                elif val == 'affine':
                    self.warpDim.append(6)
                elif val == 'homography':
                    self.warpDim.append(8)
            self.pInit = tf.zeros([MiniBatchSize, self.warpDim[0]])
        else:
            self.warpType = warpType
            if self.warpType == 'yaw':
                self.warpDim = 1
            if self.warpType == 'scale':
                self.warpDim = 1
            if self.warpType == 'translation':
                self.warpDim = 2
            elif self.warpType == 'similarity':
                self.warpDim = 4
            elif self.warpType == 'affine':
                self.warpDim = 6
            elif self.warpType == 'homography':
                self.warpDim = 8
            self.pInit = tf.zeros([MiniBatchSize, self.warpDim])
        self.canon4pts = np.array([[-1,-1],[-1,1],[1,1],[1,-1]],dtype=np.float32)
        self.image4pts = np.array([[0,0],[0,PatchSize[1]-1],[PatchSize[0]-1,PatchSize[1]-1],[PatchSize[0]-1,0]],dtype=np.float32)
        self.refMtrx = fit(Xsrc=self.canon4pts, Xdst=self.image4pts)
        self.pertScale = pertScale
        self.transScale = transScale
        self.AddTranslation = bool(AddTranslation)
        self.currBlock = 0 # Only used if self.warpTypeMultiple is True

# fit (affine) warp between two sets of points 
def fit(Xsrc,Xdst):
        ptsN = len(Xsrc)
        X,Y,U,V,O,I = Xsrc[:,0],Xsrc[:,1],Xdst[:,0],Xdst[:,1],np.zeros([ptsN]),np.ones([ptsN])
        A = np.concatenate((np.stack([X,Y,I,O,O,O],axis=1),
                            np.stack([O,O,O,X,Y,I],axis=1)),axis=0)
        b = np.concatenate((U,V),axis=0)
        p1,p2,p3,p4,p5,p6 = scipy.linalg.lstsq(A,b)[0].squeeze()
        pMtrx = np.array([[p1,p2,p3],[p4,p5,p6],[0,0,1]],dtype=np.float32)
        return pMtrx

# compute composition of warp parameters
def compose(opt,pMtrx,dpMtrx):
        with tf.name_scope("compose"):
                # pMtrx = vec2mtrx(opt,p)
                # dpMtrx = vec2mtrx(opt,dp)
                pMtrxNew = tf.matmul(dpMtrx,pMtrx)
                pMtrxNew /= pMtrxNew[:,2:3,2:3]
                # pNew = mtrx2vec(opt,pMtrxNew)
        return pMtrxNew

# compute inverse of warp parameters
def inverse(opt,p):
        with tf.name_scope("inverse"):
                pMtrx = vec2mtrx(opt,p)
                pInvMtrx = tf.matrix_inverse(pMtrx)
                pInv = mtrx2vec(opt,pInvMtrx)
        return pInv

# convert warp parameters to matrix
def vec2mtrx(opt,p):
        with tf.name_scope("vec2mtrx"):
                O = tf.zeros([opt.batchSize])
                I = tf.ones([opt.batchSize])
                if(isinstance(opt.warpType, list)):
                        # If you don't need different warps, send single string for warpType instead
                        CompareVal = opt.warpType[opt.currBlock]
                else:
                        CompareVal =  opt.warpType
                if CompareVal == "yaw":
                       # value of sinpsi is regressed directly
                       sinpsi = tf.squeeze(p) # tf.unstack(p,axis=1)
                       cospsi = tf.math.sqrt(tf.math.subtract(1.0, tf.math.pow(sinpsi,2)))
                       pMtrx = tf.transpose(tf.stack([[cospsi,-sinpsi,O],[cospsi,sinpsi,O],[O,O,I]]),perm=[2,0,1])
                if CompareVal == "scale":
                       scale = tf.squeeze(p) # tf.unstack(p,axis=1)
                       pMtrx = tf.transpose(tf.stack([[scale,O,O],[O,scale,O],[O,O,I]]),perm=[2,0,1])
                if CompareVal == "translation":
                       tx,ty = tf.unstack(p,axis=1)
                       pMtrx = tf.transpose(tf.stack([[I,O,tx],[O,I,ty],[O,O,I]]),perm=[2,0,1])
                if CompareVal == "similarity":
                       pc,ps,tx,ty = tf.unstack(p,axis=1)
                       pMtrx = tf.transpose(tf.stack([[I+pc,-ps,tx],[ps,I+pc,ty],[O,O,I]]),perm=[2,0,1])
                if CompareVal == "affine":
                       p1,p2,p3,p4,p5,p6,p7,p8 = tf.unstack(p,axis=1)
                       pMtrx = tf.transpose(tf.stack([[I+p1,p2,p3],[p4,I+p5,p6],[O,O,I]]),perm=[2,0,1])
                if CompareVal == "homography":
                       p1,p2,p3,p4,p5,p6,p7,p8 = tf.unstack(p,axis=1)
                       pMtrx = tf.transpose(tf.stack([[I+p1,p2,p3],[p4,I+p5,p6],[p7,p8,I]]),perm=[2,0,1])
        return pMtrx

# convert warp matrix to parameters
def mtrx2vec(opt,pMtrx):
        with tf.name_scope("mtrx2vec"):
                [row0,row1,row2] = tf.unstack(pMtrx,axis=1)
                [e00,e01,e02] = tf.unstack(row0,axis=1)
                [e10,e11,e12] = tf.unstack(row1,axis=1)
                [e20,e21,e22] = tf.unstack(row2,axis=1)
                if(isinstance(opt.warpType, list)):
                        # If you don't need different warps, send single string for warpType instead
                        CompareVal = opt.warpType[opt.currBlock]
                else:
                        CompareVal =  opt.warpType

                if CompareVal == "yaw": p = [[e11]] # value of sinpsi is regressed directly, this might make cospsi unconstrained?
                if CompareVal == "scale": p = [[e00]] # this might make e00 != e11?
                if CompareVal == "translation": p = tf.stack([e02,e12],axis=1)
                if CompareVal == "similarity": p = tf.stack([e00-1,e10,e02,e12],axis=1)
                if CompareVal == "affine": p = tf.stack([e00-1,e01,e02,e10,e11-1,e12],axis=1)
                if CompareVal == "homography": p = tf.stack([e00-1,e01,e02,e10,e11-1,e12,e20,e21],axis=1)
        return p

# warp the image
def transformImage(opt,image,pMtrx):
        with tf.name_scope("transformImage"):
               # opt.refMtrx = warp.fit(Xsrc=opt.canon4pts,Xdst=opt.image4pts)
               refMtrx = tf.tile(tf.expand_dims(opt.refMtrx,axis=0),[opt.batchSize,1,1])
               transMtrx = tf.matmul(refMtrx,pMtrx)
               # warp the canonical coordinates
               X,Y = np.meshgrid(np.linspace(-1,1,opt.W),np.linspace(-1,1,opt.H))
               X,Y = X.flatten(),Y.flatten()
               XYhom = np.stack([X,Y,np.ones_like(X)],axis=1).T
               XYhom = np.tile(XYhom,[opt.batchSize,1,1]).astype(np.float32)
               XYwarpHom = tf.matmul(transMtrx,XYhom)
               XwarpHom,YwarpHom,ZwarpHom = tf.unstack(XYwarpHom,axis=1)
               Xwarp = tf.reshape(XwarpHom/(ZwarpHom+1e-8),[opt.batchSize,opt.H,opt.W])
               Ywarp = tf.reshape(YwarpHom/(ZwarpHom+1e-8),[opt.batchSize,opt.H,opt.W])
               # get the integer sampling coordinates
               Xfloor,Xceil = tf.floor(Xwarp),tf.ceil(Xwarp)
               Yfloor,Yceil = tf.floor(Ywarp),tf.ceil(Ywarp)
               XfloorInt,XceilInt = tf.to_int32(Xfloor),tf.to_int32(Xceil)
               YfloorInt,YceilInt = tf.to_int32(Yfloor),tf.to_int32(Yceil)
               imageIdx = np.tile(np.arange(opt.batchSize).reshape([opt.batchSize,1,1]),[1,opt.H,opt.W])
               imageVec = tf.reshape(image,[-1,int(image.shape[-1])])
               imageVecOut = tf.concat([imageVec,tf.zeros([1,int(image.shape[-1])])],axis=0)
               idxUL = (imageIdx*opt.H+YfloorInt)*opt.W+XfloorInt
               idxUR = (imageIdx*opt.H+YfloorInt)*opt.W+XceilInt
               idxBL = (imageIdx*opt.H+YceilInt)*opt.W+XfloorInt
               idxBR = (imageIdx*opt.H+YceilInt)*opt.W+XceilInt
               idxOutside = tf.fill([opt.batchSize,opt.H,opt.W],opt.batchSize*opt.H*opt.W)
               def insideImage(Xint,Yint):
                       return (Xint>=0)&(Xint<opt.W)&(Yint>=0)&(Yint<opt.H)
               idxUL = tf.where(insideImage(XfloorInt,YfloorInt),idxUL,idxOutside)
               idxUR = tf.where(insideImage(XceilInt,YfloorInt),idxUR,idxOutside)
               idxBL = tf.where(insideImage(XfloorInt,YceilInt),idxBL,idxOutside)
               idxBR = tf.where(insideImage(XceilInt,YceilInt),idxBR,idxOutside)
               # bilinear interpolation
               Xratio = tf.reshape(Xwarp-Xfloor,[opt.batchSize,opt.H,opt.W,1])
               Yratio = tf.reshape(Ywarp-Yfloor,[opt.batchSize,opt.H,opt.W,1])
               imageUL = tf.to_float(tf.gather(imageVecOut,idxUL))*(1-Xratio)*(1-Yratio)
               imageUR = tf.to_float(tf.gather(imageVecOut,idxUR))*(Xratio)*(1-Yratio)
               imageBL = tf.to_float(tf.gather(imageVecOut,idxBL))*(1-Xratio)*(Yratio)
               imageBR = tf.to_float(tf.gather(imageVecOut,idxBR))*(Xratio)*(Yratio)
               imageWarp = imageUL+imageUR+imageBL+imageBR
        return imageWarp

# # generate training batch
# def genPerturbationsNP(opt):
# 	# DOESNT WORK FOR batchSize = 1
# 	X = np.tile(opt.canon4pts[:,0],[opt.batchSize,1]) # opt.canon4pts = np.array([[-1,-1],[-1,1],[1,1],[1,-1]],dtype=np.float32) BS x 4
# 	Y = np.tile(opt.canon4pts[:,1],[opt.batchSize,1]) # BS x 4
# 	dX = np.random.normal(size=[opt.batchSize,4])*opt.pertScale + \
# 		np.random.normal(size=[opt.batchSize,1])*opt.transScale # transScale = 0.25, pertScale = 0.25 BS x 4
# 	dY = np.random.normal(size=[opt.batchSize,4])*opt.pertScale + \
# 		np.random.normal(size=[opt.batchSize,1])*opt.transScale # transScale = 0.25, pertScale = 0.25 BS x 4
# 	O = np.zeros([opt.batchSize,4],dtype=np.float32)
# 	I = np.ones([opt.batchSize,4],dtype=np.float32)
        
# 	# fit warp parameters to generated displacements
# 	if opt.warpType=="homography":
# 		A = np.concatenate([np.stack([X,Y,I,O,O,O,-X*(X+dX),-Y*(X+dX)],axis=-1),
# 				    np.stack([O,O,O,X,Y,I,-X*(Y+dY),-Y*(Y+dY)],axis=-1)],1) # Normal Ax = b (8 parameters) 
# 		b = np.expand_dims(np.concatenate([X+dX,Y+dY],1),-1)
# 		pPert = np.linalg.solve(A,b)[:,:,0]  # Homography matrix as a vector
# 		pPert -= np.asarray([[1,0,0,0,1,0,0,0]]).astype(float) # 1 is subtracted for homogeneous transformation
# 	else:
# 		if opt.warpType=="translation":
# 			J = np.concatenate([np.stack([I,O],axis=-1),
# 					    np.stack([O,I],axis=-1)],axis=1)
# 		if opt.warpType=="similarity":
# 			J = np.concatenate([np.stack([X,Y,I,O],axis=-1),
# 					    np.stack([-Y,X,O,I],axis=-1)],axis=1)
# 		if opt.warpType=="affine":
# 			J = np.concatenate([np.stack([X,Y,I,O,O,O],axis=-1),
# 					    np.stack([O,O,O,X,Y,I],axis=-1)],axis=1)
# 		        dXY = np.expand_dims(np.concatenate([dX,dY],1),-1)
# 		        pPert = np.linalg.lstsq(J,dXY)[:,:,0]
# 	return pPert

# # convert warp parameters to matrix
# # This is in Canon4Pts domain, i.e., [-1,1]
# def vec2mtrxNP(opt, p):
# 	# DOESNT WORK FOR batchSize = 1
# 	O = np.zeros([opt.batchSize])
# 	I = np.ones([opt.batchSize])
# 	if opt.warpType=="translation":
# 		tx, ty = np.squeeze(np.split(p, opt.warpDim, axis=1))
# 		pMtrx = np.transpose(np.stack([[I,O,tx],[O,I,ty],[O,O,I]]), axes=[2,0,1])
# 	if opt.warpType=="similarity":
# 		pc,ps,tx,ty = np.squeeze(np.split(p, opt.warpDim, axis=1))
# 		pMtrx = np.transpose(np.stack([[I+pc,-ps,tx],[ps,I+pc,ty],[O,O,I]]), axes=[2,0,1])
# 	if opt.warpType=="affine":
# 		p1,p2,p3,p4,p5,p6,p7,p8 = np.squeeze(np.split(p, opt.warpDim, axis=1))
# 		pMtrx = np.transpose(np.stack([[I+p1,p2,p3],[p4,I+p5,p6],[O,O,I]]), axes=[2,0,1])
# 	if opt.warpType=="homography":
# 		p1,p2,p3,p4,p5,p6,p7,p8 = np.squeeze(np.split(p, opt.warpDim, axis=1))
# 		pMtrx = np.transpose(np.stack([[I+p1,p2,p3],[p4,I+p5,p6],[p7,p8,I]]), axes=[2,0,1])
# 	        refMtrx = np.tile(np.expand_dims(opt.refMtrx,axis=0),[opt.batchSize,1,1])
# 	        transMtrx = np.matmul(refMtrx, pMtrx)
# 	return pMtrx, transMtrx
