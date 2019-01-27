#--Ayan Chakrabarti <ayan@wustl.edu>
import tensorflow as tf
import os


sopath=os.path.abspath(os.path.dirname(__file__))+'/ops/quant.so'
try:
    mod = tf.load_op_library(sopath)
except:
    mod = None
    print("WARNING: COULD NOT LOAD CUDA LIBRARY")

def cu_quant(qtype,act,bias):
    assert qtype == 4 or qtype==8

    vshp = act.get_shape().as_list()
    if qtype==8:
        assert vshp[-1] >= 4 and vshp[-1] % 4 == 0
        vshp[-1] = vshp[-1]//4
    else:        
        assert vshp[-1] >= 8 and vshp[-1] % 8 == 0
        vshp[-1] = vshp[-1]//8

    var = tf.Variable(tf.zeros(vshp,dtype=tf.float32))
    sOp = [mod.save_act(var,act,bias,qtype).op]
    outs, Rm = mod.rest_act(var,bias,qtype)
    return sOp, outs, Rm


def tf_quant(qtype,act,bias):
    assert qtype==4 or qtype==8

    if qtype == 4:
        nbins = 16
    else:
        nbins = 256

    shift = tf.maximum(1.0,nbins//2-tf.floor(bias*nbins/6.0))

    outs = tf.floor((act-1e-6)*nbins/6.0)
    outs = tf.cast(tf.clip_by_value(outs+shift,0,nbins-1),tf.uint8)
            
    # Pack for 4-bit
    if nbins == 16:
        bsz = outs.get_shape()[0]
        o1 = tf.bitwise.left_shift(outs[:bsz//2,:,:,:],4)
        outs = tf.bitwise.bitwise_or(o1,outs[bsz//2:,:,:,:])

    var = tf.Variable(tf.zeros(outs.shape,dtype=tf.uint8))
    # Save, Restore    
    sOp = [tf.assign(var,outs).op]

    outs = var
    # Unpack for 4-bit
    if nbins == 16:
        o1 = tf.bitwise.right_shift(outs,4)
        o2 = tf.bitwise.bitwise_and(outs,15)
        outs = tf.concat([o1,o2],0)

    Rm = tf.cast(outs >= tf.cast(shift,tf.uint8),tf.float32)
    outs = (tf.cast(outs,tf.float32)+(0.5-shift))*6.0/nbins

    return sOp, outs, Rm 

if mod is not None:
    quant = cu_quant
else:
    quant = tf_quant


if __name__ == "__main__":
    import numpy as np

    bias = tf.range(16,dtype=tf.float32)/16.0-0.5
    act = tf.random_normal([20,30,30,16],dtype=tf.float32)

    s1,o1,R1 = tf_quant(8,act,bias)
    s2,o2,R2 = cu_quant(8,act,bias)

    s1b,o1b,R1b = tf_quant(4,act,bias)
    s2b,o2b,R2b = cu_quant(4,act,bias)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    print("8-bit Test")
    sess.run(s1+s2)
    o1,R1,o2,R2 = sess.run([o1,R1,o2,R2])

    print(np.mean(np.abs(o1-o2)))
    print(np.mean(np.abs(R1-R2)))

    print(o1.flatten())
    print(o2.flatten())

    print("4-bit Test")
    sess.run(s1b+s2b)
    o1,R1,o2,R2 = sess.run([o1b,R1b,o2b,R2b])

    print(np.mean(np.abs(o1-o2)))
    print(np.mean(np.abs(R1-R2)))

    print(o1.flatten())
    print(o2.flatten())
