#--Ayan Chakrabarti <ayan@wustl.edu>
import tensorflow as tf
import numpy as np
import blpa.graph as gp

NLDef = gp.NLDef
LogLoss = gp.LogLoss
SimpleData = gp.SimpleData

def dsplit(d,ifl,ngpu):
    newd = []
    for j in range(ngpu):
        if ifl:
            dj = []
            for i in range(len(d)):
                if hasattr(d[i],'__iter__'):
                    dj.append(d[i][j::ngpu])
                else:
                    dj.append(d[i])
        else:
            dj = d[j::ngpu]
        newd.append(dj)
        
    return newd

class Graph:
    def __init__(self,dlayer,llayer,ngpu):
        self.graphs = []
        self.ngpu = ngpu
        self.dL = []
        self.lL = []
        for i in range(ngpu):
            dli = dlayer[0](*dlayer[1])
            lli = llayer[0](*llayer[1])
            self.dL.append(dli)
            self.lL.append(lli)
            self.graphs.append(gp.Graph(dli,lli))

    def add(self,*args):
        for i in range(len(self.graphs)):
            self.graphs[i].add(*args)

    def close(self,qtype=0,WD=1e-4,multi=False):
        if multi == 1:
            multi = False
        self.multi = multi
        
        for i in range(len(self.graphs)):
            with tf.device('/gpu:%d'%i):
                self.graphs[i].close(qtype,WD,False,True)

        self.weights = []
        self.grads = []

        self.wg2c = [] # Copy weights from gpu0 to cpu
        self.wc2g = [] # Copy weights from cpu to all gpus
        self.gg2c = [] # Aggregate gradients from all gpus to cpu
        
        g0 = self.graphs[0]
        for l in range(len(g0.weights)):
            shp = g0.weights[l].get_shape()
            with tf.device('/cpu:0'):
                wl = tf.Variable(tf.zeros(shp))
                gl = tf.Variable(tf.zeros(shp))
                
            self.weights.append(wl)
            self.grads.append(gl)

            self.wg2c.append(tf.assign(wl,g0.weights[l]))
            gsum = []
            for g in self.graphs:
                self.wc2g.append(tf.assign(g.weights[l],wl))
                gsum.append(g.grads[l])
            gsum = tf.add_n(gsum)
            if self.multi:
                gop = tf.assign_add(gl,gsum)
            else:
                gop = tf.assign(gl,gsum)
            self.gg2c.append(gop)

        if self.multi:
            self.gzero = tf.group(*[g.initializer for g in self.grads])
        self.wg2c = tf.group(*self.wg2c)
        self.wc2g = tf.group(*self.wc2g)
        self.gg2c = tf.group(*self.gg2c)

        fac = self.ngpu
        if self.multi:
            fac *= self.multi
        fac = 1.0 / np.float32(fac)
        gvpairs = [(fac*self.grads[i],self.weights[i]) for i in range(len(self.weights))]
        self.lr = tf.placeholder(tf.float32)
        self.opt = tf.train.MomentumOptimizer(self.lr,0.9)
        self.upd = self.opt.apply_gradients(gvpairs)

        self.fops = []
        self.bops = []
        for l in range(len(g0.layers)):
            self.fops.append([g.layers[l].fOp for g in self.graphs])
            self.bops.append([g.layers[l].bOp for g in self.graphs])


        self.acc, self.loss, self.loss_fb = [], [], []
        for g in self.graphs:
            a,l,o = g.lL.get_loss_ops()
            self.acc.append(a)
            self.loss.append(l)
            self.loss_fb.append(o)

        self.acc = tf.add_n(self.acc)/np.float32(len(self.acc))
        self.loss = tf.add_n(self.loss)/np.float32(len(self.loss))

        # Create Session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        try:
            sess.run(tf.global_variables_initializer())
        except:
            sess.run(tf.initialize_all_variables())
        sess.run(self.wg2c)
        sess.run(self.wc2g)
        self.sess = sess
        

    def forward(self,x,y,isxlist=False,isylist=False):
        x = dsplit(x,isxlist,self.ngpu)
        y = dsplit(y,isylist,self.ngpu)
        
        fd = {}
        for i in range(self.ngpu):
            fd.update(self.dL[i].fd(x[i]))
        
        self.sess.run(self.fops[0],feed_dict=fd)
        for k in range(1,len(self.fops)):
            self.sess.run(self.fops[k])

        fd = {}
        for i in range(self.ngpu):
            fd.update(self.lL[i].get_loss_fd(y[i]))
            
        acc,loss,_ = self.sess.run([self.acc,self.loss,self.loss_fb],fd)
        return acc, loss

    def backward(self,lr):
        for k in range(len(self.bops)-1,0,-1):
            self.sess.run(self.bops[k])
        self.sess.run(self.gg2c)
        self.sess.run(self.upd, feed_dict={self.lr: lr})
        self.sess.run(self.wc2g)

    def binit(self):
        if self.multi:
            self.sess.run(self.gzero)

    def hback(self):
        if not self.multi:
            return
        for k in range(len(self.bops)-1,0,-1):
            self.sess.run(self.bops[k])
        self.sess.run(self.gg2c)

    def cback(self,lr):
        if self.multi:
            self.sess.run(self.upd, feed_dict={self.lr: lr})
            self.sess.run(self.wc2g)
        else:
            self.backward(lr)
            
    def save(self,saveloc,niter):
        wts = {}
        for k in range(len(self.weights)):
            wts['%d'%k] = self.weights[k].eval(self.sess)
        wts['niter'] = niter
        np.savez(saveloc,**wts)

    def load(self,saveloc):
        wts = np.load(saveloc);
        for k in range(len(self.weights)):
            self.weights[k].load(wts['%d'%k],self.sess)
        self.sess.run(self.wc2g)
        return wts['niter']
        
