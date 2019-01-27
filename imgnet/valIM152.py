#!/usr/bin/env python3
#--Ayan Chakrabarti <ayan@wustl.edu>
import numpy as np
import ctrlc
import time
import sys
import os
import importlib
import argparse

BSZ=100

def mprint(s):
    sys.stdout.write(time.strftime("%Y-%m-%d %H:%M:%S ")+s+"\n")
    sys.stdout.flush()
    
parser = argparse.ArgumentParser()
parser.add_argument('-save',help='Path to model file name',required=True)
opts = parser.parse_args()
saveloc = opts.save

##########################################################################################
# Load data

BDIR='/scratch/data/ImageNet/clsloc/'
val = [f.rstrip().split(',') for f in open(BDIR+'val.txt').readlines()]

val_im = [BDIR+f[0] for f in val]
val_lb = [int(f[1]) for f in val]

##########################################################################################

from im10crop152 import Model
g = Model(BSZ,4,0.)


VSIZE = len(val_im)//BSZ

_ = g.load(saveloc)
vacc1 = 0.; vacc5 = 0.; viter = 0
for b in range(VSIZE):
    b_im = [val_im[i] for i in range(b*BSZ,(b+1)*BSZ)]
    b_lb = [val_lb[i] for i in range(b*BSZ,(b+1)*BSZ)]

    b_lb = np.reshape(np.int32(b_lb),[-1])

    pred = 0.
    for cid in range(10):
        pred += g.forward([cid,b_im],None)

    pred = np.reshape(pred,[-1,1000])

    pred = np.fliplr(np.argsort(pred))
    v0 = np.mean(np.float32(pred[:,0] == b_lb))
    vacc1 = vacc1+v0
    for i in range(1,5):
        v0 = v0 + np.mean(np.float32(pred[:,i] == b_lb))
    vacc5 = vacc5+v0
    viter = viter + 1;
    mprint("%d of %d: Acc.1 = %.5f, Acc.5 = %.5f" % (b+1,VSIZE,vacc1/viter, vacc5/viter))
