#--Ayan Chakrabarti <ayan@wustl.edu>
import numpy as np

data = np.load('data/cifar10/train.npz')
im = np.float32(data['imgs'])/255.
im = im - np.mean(im,1,keepdims=True)
train_im = np.reshape(im,[-1,32,32,3])
train_lb = np.reshape(data['labels'],[-1,1,1,1])

data = np.load('data/cifar10/test.npz')
im = np.float32(data['imgs'])/255.
im = im - np.mean(im,1,keepdims=True)
val_im = np.reshape(im,[-1,32,32,3])
val_lb = np.reshape(data['labels'],[-1,1,1,1])

data = None
AVPOOL=True

ISZ = [0,32,32,3]
NCLS=10

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
