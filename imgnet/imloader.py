#--Ayan Chakrabarti <ayan@wustl.edu>
import numpy as np
import tensorflow as tf

def rsz_img(img,scale):
    in_s = tf.to_float(tf.shape(img)[:2])
    min_s = tf.minimum(in_s[0],in_s[1])
    new_s = tf.to_int32(tf.round((scale/min_s)*in_s))
    return tf.image.resize_images(img,[new_s[0],new_s[1]])

def cent_crop(img,shp):
    ishp = tf.shape(img)
    ys = (ishp[0]-shp[0])//2
    xs = (ishp[1]-shp[1])//2

    return tf.slice(img,tf.stack([ys,xs,0]),[shp[0],shp[1],3])

def rand_color(img):
    img = tf.image.random_brightness(img, max_delta=32. / 255.)
    img = tf.image.random_saturation(img, lower=0.5, upper=1.5)
    img = tf.image.random_hue(img, max_delta=0.2)
    img = tf.image.random_contrast(img, lower=0.5, upper=1.5)
    return tf.clip_by_value(img,0.0,1.0)

# Data layers
class IMData:
    def __init__(self,bsz):
        self.back = False
        self.oSz = [bsz,224,224,3]

        self.isrand = tf.placeholder(tf.bool)
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


            scale = tf.cond(self.isrand,
                            lambda: tf.to_float(tf.random_uniform((),256,481,dtype=tf.int32)),
                            lambda: 368.0)

            img = rsz_img(img,scale)

            img = tf.cond(self.isrand,
                          lambda: tf.random_crop(img,[224,224,3]),
                          lambda: cent_crop(img,[224,224]))
            img = tf.cond(self.isrand,
                          lambda: tf.image.random_flip_left_right(img),
                          lambda: img)
            img = tf.to_float(img)/255.0
            img = tf.cond(self.isrand, lambda: rand_color(img), lambda: img)
            imgs += [img]

        img = tf.stack(imgs,0)
        img = (img - [0.485,0.456,0.406])/[0.229,0.224,0.225]
        self.ftop = img
        
    def fd(self,batch):
        fd = {self.isrand: batch[0]}
        for i in range(len(self.fns)):
            fd[self.fns[i]] = batch[1][i]
            
        return fd

        
