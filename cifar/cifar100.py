#--Ayan Chakrabarti <ayan@wustl.edu>
import numpy as np
import pickle as p

data = p.load(open('data/cifar100/train','rb'),encoding='latin1')
im = np.float32(data['data'])/255.
im = im - np.mean(im,1,keepdims=True)
train_im = np.reshape(im,[-1,3,32,32])
train_im = np.transpose(train_im,[0,2,3,1]).copy()
train_lb = np.reshape(np.int32(data['fine_labels']),[-1,1,1,1])

data = p.load(open('data/cifar100/test','rb'),encoding='latin1')
im = np.float32(data['data'])/255.
im = im - np.mean(im,1,keepdims=True)
val_im = np.reshape(im,[-1,3,32,32])
val_im = np.transpose(val_im,[0,2,3,1]).copy()
val_lb = np.reshape(np.int32(data['fine_labels']),[-1,1,1,1])

data=None
AVPOOL=True

ISZ = [0,32,32,3]
NCLS=100

def augment(img):
    BSZ = img.shape[0]
    img = np.pad(img,((0,0),(4,4),(4,4),(0,0)),'constant')
    nx = np.random.randint(9)
    ny = np.random.randint(9)
    img = img[:,ny:(ny+32),nx:(nx+32),:]
    img[:BSZ//2,:,:,:] = img[:BSZ//2,:,::-1,:]
    return img
    
MAXITER = 64000
VALITER = 400

WD=2e-4
def get_lr(ep):
    if ep < 400:
        return 1e-2
    elif ep < 32000:
        return 1e-1
    elif ep < 48000:
        return 1e-2
    else:
        return 1e-3
