#!/usr/bin/env python3
#--Ayan Chakrabarti <ayan@wustl.edu>
import numpy as np
import ctrlc
import time
import sys
import os
import importlib
import argparse
np.random.seed(0)

DISPITER=10
def mprint(s):
    sys.stdout.write(time.strftime("%Y-%m-%d %H:%M:%S ")+s+"\n")
    sys.stdout.flush()
    
parser = argparse.ArgumentParser()
parser.add_argument('-save',help='Path to model file name',required=True)
parser.add_argument('-qtype',type=int,default=0,
                    help='Quantization type: 0: Exact Training. 4, 8: Quantize Activations to 4 or 8 bits. ' \
                         'Defalut: 0.')

parser.add_argument('-dset',default='cifar100',help='Dataset: cifar100 / cifar10. Default cifar100.')
parser.add_argument('-seed',type=int,default=0,help='Random Seed for training set shuffling. Default 0.')
parser.add_argument('-bsz',type=int,default=128,help='Batch Size. Default 128.')
parser.add_argument('-rblocks',type=int,default=18,help='No of res-blocks in each group. Total layers = 9*l+2. Default: 18.')

opts = parser.parse_args()

BSZ= opts.bsz
ng = opts.rblocks
saveloc = opts.save

##########################################################################################
# Load data

dset = importlib.import_module(opts.dset)
train_im = dset.train_im
train_lb = dset.train_lb
val_im = dset.val_im
val_lb = dset.val_lb

ISZ=dset.ISZ
ISZ[0] = BSZ

##########################################################################################

from Resnet import Model
g = Model(ISZ,dset.NCLS,ng,dset.AVPOOL,opts.qtype,dset.WD)

batches = range(0,train_lb.shape[0]-BSZ+1,BSZ)
ESIZE = len(batches)

rsv = np.random.RandomState(0)
validx = rsv.permutation(len(val_lb))


rs = np.random.RandomState(opts.seed)

origiter = 0
if os.path.isfile(saveloc):
    origiter = g.load(saveloc)
    for k in range( (origiter+ESIZE-1) // ESIZE):
        idx = rs.permutation(len(train_lb))
    mprint("Restored to iteration %d" % origiter)    
niter = origiter    
    

avg_loss = 0.; avg_acc = 0.
while niter < dset.MAXITER+1:
    lr = dset.get_lr(niter)
    
    if niter % dset.VALITER == 0:
        vacc = 0.; vloss = 0.; viter = 0
        for b in range(0,len(val_lb)-BSZ+1,BSZ):
            acc,loss = g.forward(val_im[validx[b:b+BSZ,...]],val_lb[validx[b:b+BSZ],...])
            viter = viter + 1;vacc = vacc + acc;vloss = vloss + loss
        vloss = vloss / viter; vacc = vacc / viter
        mprint("[%09d] Val_Loss = %.3e, Val_Acc = %.4f" % (niter,vloss,vacc))

    if niter == dset.MAXITER:
        break

    if niter % ESIZE == 0:
        idx = rs.permutation(len(train_lb))

    b = batches[niter % ESIZE]

    inp_b = dset.augment(train_im[idx[b:b+BSZ],...])
        
    acc,loss = g.forward(inp_b,train_lb[idx[b:b+BSZ],...])
    g.backward(lr)
    niter = niter+1
    
    avg_loss = avg_loss + loss; avg_acc = avg_acc + acc;
    if niter % DISPITER == 0:
        avg_loss = avg_loss / DISPITER; avg_acc = avg_acc / DISPITER
        mprint("[%09d] lr=%.2e, Train_Loss = %.3e, Train_Acc = %.4f" % (niter,lr,avg_loss,avg_acc))
        avg_loss = 0.; avg_acc = 0.;
        if ctrlc.stop:
            break

if niter > origiter:
    g.save(saveloc,niter)
    mprint("Saved model.")
