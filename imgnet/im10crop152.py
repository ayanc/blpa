#--Ayan Chakrabarti <ayan@wustl.edu>
import numpy as np
import tensorflow as tf
import blpa.graph as gp

SCALE=256.

def rsz_img(img,scale):
    in_s = tf.to_float(tf.shape(img)[:2])
    min_s = tf.minimum(in_s[0],in_s[1])
    new_s = tf.to_int32(tf.round((scale/min_s)*in_s))
    return tf.image.resize_images(img,[new_s[0],new_s[1]])

def crop(img,shp,cid):
    ishp = tf.shape(img)
    yr = (ishp[0]-shp[0])
    xr = (ishp[1]-shp[1])

    xs = tf.stack([0,0,xr,xr,xr//2])
    ys = tf.stack([0,yr,0,yr,yr//2])

    ys = ys[cid]
    xs = xs[cid]
    
    return tf.slice(img,tf.stack([ys,xs,0]),[shp[0],shp[1],3])

# Data layers
class IMData:
    def __init__(self,bsz):
        self.back = False
        self.oSz = [bsz,224,224,3]

        self.isflip = tf.placeholder(tf.bool)
        self.cid = tf.placeholder(tf.int32)
        
        self.fns = []
        imgs = []
        for i in range(bsz):
            fn = tf.placeholder(tf.string)
            self.fns += [fn]

            img = tf.read_file(self.fns[i])
            code = tf.decode_raw(img,tf.uint8)[0]
            img = tf.cond(tf.equal(code,137),
                          lambda: tf.image.decode_png(img,channels=3),
                          lambda: tf.image.decode_jpeg(img,channels=3))



            img = rsz_img(img,SCALE)
            img = crop(img,[224,224],self.cid)
            img = tf.cond(self.isflip,
                          lambda: tf.image.flip_left_right(img),
                          lambda: img)
            
            img = tf.to_float(img)/255.0
            imgs += [img]

        img = tf.stack(imgs,0)
        img = (img - [0.485,0.456,0.406])/[0.229,0.224,0.225]
        self.ftop = img
        
    def fd(self,batch):
        isflip = batch[0] >= 5
        cid = batch[0]%5
        fd = {self.isflip: isflip, self.cid: cid}
        for i in range(len(self.fns)):
            fd[self.fns[i]] = batch[1][i]
            
        return fd

def Model(bsz,qtype=0,WD=1e-4,multi=False):
        
    g = gp.Graph(IMData(bsz),gp.LogLoss())
    
    ch = [64,64,128,256,512]
    ng = [ 3,  8,  36,  3]
    
    g.add(7,ch[0],2,'SAME',gp.NLDef(False,False,False,None,False))

    skip = gp.NLDef(True,True,False,None,False)
    join = gp.NLDef(True,True,True,1,False)
    
    g.add(1,ch[1],1,'SAME',gp.NLDef(True,True,False,2,False,True))
    g.add(3,ch[1],1,'SAME',skip)
    g.add(1,ch[1]*4,1,'SAME',skip)
    for i in range(ng[0]-1):
        g.add(1,ch[1],1,'SAME',join)
        g.add(3,ch[1],1,'SAME',skip)
        g.add(1,ch[1]*4,1,'SAME',skip)

    for j in range(1,len(ng)):    
        g.add(1,ch[j+1],2,'SAME',join)
        g.add(3,ch[j+1],1,'SAME',skip)
        g.add(1,ch[j+1]*4,1,'SAME',skip)
        for i in range(ng[j]-1):
            g.add(1,ch[j+1],1,'SAME',join)
            g.add(3,ch[j+1],1,'SAME',skip)
            g.add(1,ch[j+1]*4,1,'SAME',skip)

    g.add(1,1000,1,'VALID',gp.NLDef(True,True,True,None,True))
    g.close(qtype,WD,multi)

    return g
