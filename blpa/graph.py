#--Ayan Chakrabarti <ayan@wustl.edu>
import numpy as np
import tensorflow as tf
import blpa.quant as q

from tensorflow.python.ops.gen_nn_ops import max_pool_grad

DT=tf.float32
BNEPS=1e-3

##################################################################################################        

class NLDef:
    def __init__(self,bn,relu,res_in,res_out,avpool,maxpool=False):
        self.bn = bn           # If BN (bool)
        self.relu = relu       # ReLU (bool)
        self.res_in = res_in   # add prev res connection (bool)
        self.res_out = res_out # Send out res connection: None, 1 (Before bn+relu), 2 (after bn+relu)

        self.avpool = avpool   # Average pool input (bool)
        self.maxpool = maxpool   # Average pool input (bool)
        
##################################################################################################        

# Weight+BN layer        
class Layer:
    def __init__(self,prev,kSz,outCh,stride,pad,nldef):
        inSz = prev.oSz
        self.prev = prev
        self.inSz = inSz
        self.ksz = kSz
        self.outCh = outCh
        self.stride = stride
        self.pad = pad
        self.nldef = nldef
        self.back = True

        self.inCh = inSz[-1]
        if self.nldef.avpool:
            self.oSz = [inSz[0],1,1,outCh]
            assert self.ksz == 1
            assert nldef.relu == True
        else:
            if self.nldef.maxpool:
                ht = inSz[2]//2
                wd = inSz[1]//2
            else:
                ht = inSz[2]
                wd = inSz[1]
                
            if self.pad == 'SAME':
                wd = wd//stride
                ht = ht//stride
            else:
                wd = (wd-self.ksz+1)//stride
                ht = (ht-self.ksz+1)//stride
            self.oSz = [inSz[0],wd,ht,outCh]
            
    def makeOps(self,graph):
        scratch, scratch2, rsz = graph.scratch, graph.scratch2, graph.rsz

        ##############################################################################
        ########################## Build forward ops
        ##############################################################################
        fsave = []
        out = self.prev.ftop

        ######### Handle res_in
        if self.nldef.res_in:
            assert self.prev.back
            rtop = tf.reshape(scratch2[:np.prod(rsz)],rsz)
            rchange = self.inSz == rsz
            if rchange:
                out = out + rtop
            else:
                rstride = rsz[1] // self.inSz[1]
                reskern = tf.ones([rstride,rstride,rsz[-1],1],dtype=DT)/np.float32(rstride**2)
                respad = (self.inCh-rsz[-1])//2
                rt = rtop
                if rstride > 1:
                    rt = tf.nn.depthwise_conv2d(rt,reskern,[1,rstride,rstride,1],'VALID')
                if respad > 0:
                    rt = tf.pad(rt,[[0,0],[0,0],[0,0],[respad,respad]])
                out = out + rt
        out0 = out

        ### BN
        if self.nldef.bn:
            assert self.prev.back
            mu,vr = tf.nn.moments(out,[0,1,2])
            bnfac = tf.sqrt(vr + BNEPS)
            out = (out - mu) / bnfac
            self.bnfac = tf.Variable(tf.zeros_like(bnfac))
            fsave += [tf.assign(self.bnfac,bnfac).op]

            scale = tf.Variable(tf.ones([self.inCh],dtype=DT))
            sgrad = tf.Variable(tf.zeros([self.inCh],dtype=DT))
            graph.weights.append(scale)
            graph.grads.append(sgrad)
            cscale = tf.maximum(1e-8,scale)
            out = out*cscale

        ### Bias
        if self.prev.back:
            bias = tf.Variable(tf.zeros([self.inCh],dtype=DT))
            bgrad = tf.Variable(tf.zeros([self.inCh],dtype=DT))
            graph.weights.append(bias)
            graph.grads.append(bgrad)
            out = out + bias
        
        ### Save + Handle ReLUs
        self.btop = None

        # If has max-pool
        if self.nldef.maxpool:
            assert self.nldef.res_out != 1
            var = tf.Variable(tf.zeros(out.get_shape(),dtype=DT))
            fsave += [tf.assign(var,out).op]
            self.btop0 = (var-bias)/cscale
            self.btop1 = self.btop0

            self.Rm = tf.cast(var > 0,dtype=DT)
            btop = tf.nn.relu(var)
            self.premp = btop
            self.btop = tf.nn.max_pool(btop,[1,3,3,1],[1,2,2,1],'SAME')

            out = tf.nn.relu(out)
            out = tf.nn.max_pool(out,[1,3,3,1],[1,2,2,1],'SAME')
            
        # Last layer
        elif self == graph.layers[-1]:
            fsave += [tf.assign(scratch2[:np.prod(self.inSz)],tf.reshape(out,[-1])).op]
            var = tf.reshape(scratch2[:np.prod(self.inSz)],out.get_shape())
                
        # Quantization
        elif self.nldef.bn and (graph.qtype == 4 or graph.qtype == 8):
            assert self.nldef.relu 

            sOp, outs, self.Rm = q.quant(graph.qtype,out/cscale,bias/cscale)
            fsave += sOp
            self.btop0 = outs-bias/cscale
            self.btop1 = self.btop0
            self.btop = tf.nn.relu(outs*cscale)
            out = tf.nn.relu(out)

            
        # No Quantization
        else:
            var = tf.Variable(tf.zeros(out.get_shape(),dtype=DT))
            fsave += [tf.assign(var,out).op]
                

        if self.btop is None:
            if self.nldef.bn:
                self.btop0 = (var-bias)/cscale
                self.btop1 = self.btop0
                
            if self.nldef.relu:
                self.btop = tf.nn.relu(var)
                self.Rm = tf.cast(var > 0,dtype=DT)
                out = tf.nn.relu(out)
            else:
                self.btop = var
        
        
        ######### Handle res_out
        if self.nldef.res_out is not None:
            graph.rsz = out.get_shape().as_list()
            sidx = np.prod(graph.rsz)
            if self.nldef.res_out == 1:
                fsave += [tf.assign(scratch2[:sidx],tf.reshape(out0,[-1])).op]
            else:
                fsave += [tf.assign(scratch2[:sidx],tf.reshape(out,[-1])).op]


        ########### Do the actual convolution
        kshp = [self.ksz,self.ksz,self.inCh,self.outCh]
        if self == graph.layers[-1]:
            sq = np.sqrt(1.0 / np.float32(self.ksz*self.ksz*self.inCh))
        else:
            sq = np.sqrt(2.0 / np.float32(self.ksz*self.ksz*self.inCh))
            
        kernel = tf.random_normal(kshp,stddev=sq,dtype=DT)
        kernel = tf.Variable(kernel)
        kgrad = tf.Variable(tf.zeros(kshp,dtype=DT))
        graph.weights.append(kernel)
        graph.grads.append(kgrad)

        if self.nldef.avpool == True:
            out = tf.reduce_mean(out,[1,2],True)
        out = tf.nn.conv2d(out,kernel,[1,self.stride,self.stride,1],self.pad)


        ########### Store output in scratch
        fsave += [tf.assign(scratch[:np.prod(self.oSz)],tf.reshape(out,[-1])).op]
        self.fOp = tf.group(*fsave)
        self.ftop = tf.reshape(scratch[:np.prod(self.oSz)],self.oSz)


        ##############################################################################
        ########################## Build Backward ops
        ##############################################################################
        ingrad = self.ftop # Same shape loading from scratch
        bsave = []
        
        inp = self.btop
        if self.nldef.avpool:
            inp = tf.reduce_mean(inp,[1,2],True)
            
        kg = tf.nn.conv2d_backprop_filter(inp,kshp,ingrad,[1,self.stride,self.stride,1],self.pad)
        kg += graph.WD*kernel
        bsave += [tf.assign(kgrad,kg).op]

        if not self.prev.back:
            self.bOp = tf.group(*bsave)
            return
        
        if self.nldef.avpool:
            ingrad = tf.nn.conv2d_backprop_input([self.inSz[0],1,1,self.inSz[3]],kernel,ingrad,
                                                 [1,1,1,1],'VALID') /  np.float32(self.inSz[1]*self.inSz[2])
        elif self.nldef.maxpool:
            ingrad = tf.nn.conv2d_backprop_input([self.inSz[0],self.inSz[1]//2,self.inSz[2]//2,self.inSz[3]],
                                                 kernel,ingrad,[1,self.stride,self.stride,1],self.pad)
        else:
            ingrad = tf.nn.conv2d_backprop_input(self.inSz,kernel,ingrad,
                                                 [1,self.stride,self.stride,1],self.pad)
        if self.nldef.res_out == 2:
            gshp = ingrad.get_shape().as_list()
            ingrad += tf.reshape(graph.scratch2[:np.prod(gshp)],gshp)

        if self.nldef.maxpool:
            ingrad = max_pool_grad(self.premp,self.btop,ingrad,[1,3,3,1],[1,2,2,1],'SAME')
            
        if self.nldef.relu:
            ingrad *= self.Rm
        bsave += [tf.assign(bgrad, tf.reduce_sum(ingrad,[0,1,2])).op]
        if self.nldef.bn:
            bsave += [tf.assign(sgrad, tf.reduce_sum(ingrad*self.btop0,[0,1,2])).op]
            ingrad = ingrad * cscale
            ingrad = ingrad - tf.reduce_mean(ingrad,[0,1,2])
            ingrad -= self.btop0 * tf.reduce_mean(ingrad*self.btop1,[0,1,2])
            ingrad /= self.bnfac
        if self.nldef.res_out == 1:
            ingrad += tf.reshape(graph.scratch2[:np.prod(self.inSz)],self.inSz)

        bsave += [tf.assign(graph.scratch[:np.prod(self.inSz)],tf.reshape(ingrad,[-1])).op]
        
        if self.nldef.res_in:
            if rchange:
                bsave += [tf.assign(graph.scratch2[:np.prod(self.inSz)],tf.reshape(ingrad,[-1])).op]
            else:
                if respad > 0:
                    ingrad = ingrad[:,:,:,respad:-respad]
                if rstride > 1:
                    ingrad = tf.nn.depthwise_conv2d_native_backprop_input(rsz,reskern,ingrad,[1,rstride,rstride,1],'VALID')
                bsave += [tf.assign(graph.scratch2[:np.prod(rsz)],tf.reshape(ingrad,[-1])).op]
        
        self.bOp = tf.group(*bsave)


# Loss layer        
class LogLoss:
    def __init__(self):
        pass

    def makeOps(self,graph,prev):
        xsz = prev.oSz
        self.ph = tf.placeholder(dtype=tf.int32,shape=[xsz[0],xsz[1],xsz[2],1])

        pred = prev.ftop
        bias = tf.Variable(tf.zeros([prev.oSz[-1]],dtype=DT))
        bgrad = tf.Variable(tf.zeros([prev.oSz[-1]],dtype=DT))
        graph.weights.append(bias)
        graph.grads.append(bgrad)
        pred = pred + bias

        self.pred = pred
        smx = pred - tf.reduce_logsumexp(pred,3,True)
        amx = tf.cast(tf.expand_dims(tf.argmax(pred,3),3),tf.int32)
        ph2 = tf.cast(tf.equal(tf.cast(tf.reshape(tf.range(xsz[3]),[xsz[3]]),tf.int32),self.ph),DT)
        
        self.acc = tf.reduce_mean(tf.cast(tf.equal(amx,self.ph),DT))
        self.loss = -tf.reduce_mean(ph2*smx)*np.float32(xsz[3])

        ingrad = (tf.exp(smx)-ph2)/np.float32(xsz[0])
        gOp = [tf.assign(bgrad, tf.reduce_sum(ingrad,[0,1,2])).op]
        gOp += [tf.assign(graph.scratch[:np.prod(xsz)],tf.reshape(ingrad,[-1])).op]
        self.gOp = tf.group(*gOp)

    def get_pred(self,sess):
        out = sess.run(self.pred)
        return out

    def get_loss_ops(self):
        return self.acc,self.loss,self.gOp

    def get_loss_fd(self,labels):
        return {self.ph: labels}

    def get_loss(self,labels,sess):
        out = sess.run([self.acc,self.loss,self.gOp],feed_dict={self.ph: labels})
        return out[0], out[1]
        

##################################################################################################        
        

# Graph class: currently only suppors feed-forward chains
class Graph:
    # Call with size of input
    def __init__(self,dlayer,llayer):
        self.dL = dlayer
        self.lL = llayer
        
        self.prev = dlayer
        
        self.layers = []
        self.weights = []
        self.grads = []


    # Add a conv layer
    def add(self, *args):
        l = Layer(self.prev, *args)
        self.layers.append(l)
        self.prev = l
        
    # Call this when done adding layers
    def close(self,qtype=0,WD=1e-4,multi=False,multigpu=False):
        if multi == 1:
            multi = False
        self.multi = multi
        
        # Compute max output size
        msz = np.prod(self.dL.oSz)
        msz2 = 0
        for l in self.layers:
            lsz = np.prod(l.oSz)
            if lsz > msz:
                msz = lsz
            if l.nldef.res_out is not None:
                lsz = np.prod(l.inSz)
                if l.nldef.maxpool:
                    lsz = lsz//4
                if lsz > msz2:
                    msz2 = lsz
                

        # Create scratch spaces
        self.scratch = tf.Variable(tf.zeros(shape=msz,dtype=DT))
        self.scratch2 = tf.Variable(tf.zeros(shape=msz2,dtype=DT))
        self.rsz = 0

        # Now create ops for layers
        self.WD=WD
        self.qtype=qtype
        for i in range(len(self.layers)):
            self.layers[i].makeOps(self)

        # Set up loss layer
        self.lL.makeOps(self,self.layers[-1])

        # Set up optimizer
        if self.multi:
            fac = 1.0/np.float32(self.multi)
            self.agrads = [tf.Variable(tf.zeros(self.grads[i].get_shape())) for i in range(len(self.grads))]
            self.aginit = tf.group(*[ag.initializer for ag in self.agrads])
            self.aupd = tf.group(*[tf.assign_add(self.agrads[i],self.grads[i]).op for i in range(len(self.grads))])
            gvpairs = [(fac*self.agrads[i],self.weights[i]) for i in range(len(self.weights))]
        else:
            gvpairs = [(tf.identity(self.grads[i]),self.weights[i]) for i in range(len(self.weights))]

        if not multigpu:
            self.lr = tf.placeholder(DT)
            self.opt = tf.train.MomentumOptimizer(self.lr,0.9)
            self.upd = self.opt.apply_gradients(gvpairs)

            # Create session
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            sess = tf.Session(config=config)
            try:
                sess.run(tf.global_variables_initializer())
            except:
                sess.run(tf.initialize_all_variables())
            self.sess = sess

        
    # Forward pass: returns accuracy and loss on batch
    def forward(self,x,y,_x=None,_y=None):
        fd = self.dL.fd(x)
        
        self.sess.run(self.layers[0].fOp,feed_dict=fd)
        for k in range(1,len(self.layers)):
            self.sess.run(self.layers[k].fOp)

        if y is None:
            return self.lL.get_pred(self.sess)
        acc,loss = self.lL.get_loss(y,self.sess)
        return acc, loss

    # Backward pass: computes gradient and does update
    def backward(self,lr):
        for k in range(len(self.layers)-1,0,-1):
            self.sess.run(self.layers[k].bOp)
        if lr is not None:    
            self.sess.run(self.upd, feed_dict={self.lr: lr})

    def binit(self):
        if self.multi:
            self.sess.run(self.aginit)

    def hback(self):
        if not self.multi:
            return
        for k in range(len(self.layers)-1,0,-1):
            self.sess.run(self.layers[k].bOp)
        self.sess.run(self.aupd)

    def cback(self,lr):
        if self.multi:
            self.sess.run(self.upd, feed_dict={self.lr: lr})
        else:
            self.backward(lr)

    def save(self,saveloc,niter):
        wts = {}
        for k in range(len(self.weights)):
            wts['%d'%k] = self.weights[k].eval(self.sess)
        wts['niter'] = niter
        np.savez(saveloc,**wts)

    def load(self,saveloc):
        wts = np.load(saveloc)
        for k in range(len(self.weights)):
            self.weights[k].load(wts['%d'%k],self.sess)
        return wts['niter']

##################################################################################################        

# Data layers
class SimpleData:
    def __init__(self,xsz):
        self.back = False
        self.ftop = tf.placeholder(shape=xsz,dtype=DT)
        self.oSz = xsz
        
    def fd(self,batch):
        return {self.ftop: batch}

        
