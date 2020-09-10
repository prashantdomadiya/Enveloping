

import numpy as np
import tensorflow as tf
import time

import os
from os.path import join
"""
def WriteObj(V,F,PoseN,InputPath):
    filepath=join(InputPath,'%05d.obj' % (PoseN))
    with open(filepath, 'w') as f:
        f.write("# OBJ file\n")
        for v in V:
            f.write("v %.4f %.4f %.4f\n" % (v[0],v[1],v[2]))
        for p in F:
            f.write("f")
            for i in p:
                f.write(" %d" % (i + 1))
            f.write("\n")
    return
"""

def ReadTxt(Name):
    Vec=[]
    fl = open(Name, 'r')
    NumLine=0
    for line in fl:
        words = line.split()
        l=len(words)
        tmp=[]
        for i in range(l):     
            tmp.append(int(words[i]))
        Vec.append(tmp)
        NumLine+=1
    return Vec


#######################################################
#                 Paths
#######################################################

path='/media/prashant/DATA/MyCodes/codeFiles/LBS/'
Rigname='50002'
path=path+Rigname+'/'
##############################################################
#                      Inputs
##############################################################

LBSVrt=np.loadtxt(path+Rigname+'_LBSVrts.txt',delimiter=',')
SFrm=np.loadtxt(path+Rigname+'_NlSFrm.txt',delimiter=',')

Fc=np.int32(ReadTxt(path+Rigname+'_facz.txt'))
Joints=ReadTxt(path+Rigname+'_Joints.txt')


#################################################################
#          DeepLBS
#################################################################

NFrm=len(SFrm.T)//4
Inn=np.zeros((1,12*NFrm))
for j in range(NFrm):
    Inn[:,12*j:12*j+9]=np.reshape(SFrm[:,4*j:4*j+3],9)
    Inn[:,12*j+9:12*j+12]=SFrm[:,4*j+3]


VrtSeg=ReadTxt(path+Rigname+'_VrtSeg.txt')
NV=0
for i in VrtSeg:
    NV+=len(i)

################################################################
# Network Parameters
num_h1=256
num_h2=256
t=0
Vrt=np.zeros((NV,3))
TotalTime=0
for v in VrtSeg:
    num_input=12*NFrm
    num_output=3*len(v)
    

    # tf Graph input (only pictures)
    X = tf.placeholder("float", [None,num_input])

    weights = {
        'l1': tf.Variable(tf.random_normal([num_input,num_h1])),
        'l2': tf.Variable(tf.random_normal([num_h1,num_h2])),
        'l3': tf.Variable(tf.random_normal([num_h2,num_output]))
    }
    biases = {
        'b1': tf.Variable(tf.random_normal([num_h1])),
        'b2': tf.Variable(tf.random_normal([num_h2])),
        'b3': tf.Variable(tf.random_normal([num_output]))
    }
    

    def DeepLBS(x):
        layer_1 = tf.tanh(tf.add(tf.matmul(x,weights['l1']),biases['b1']))
        layer_2 = tf.tanh(tf.add(tf.matmul(layer_1,weights['l2']),biases['b2']))
        layer_3 = tf.add(tf.matmul(layer_2,weights['l3']),biases['b3'])
        return layer_3

    Y=DeepLBS(X)
    Vid=np.reshape(3*np.array([v]*3)+[[0],[1],[2]],3*len(v),'F')
    
    
    saver = tf.train.Saver()
    
    with tf.Session() as sess:
        
        saver.restore(sess, "/media/prashant/DATA/3D_DATA/DeepSkiningWeights/"+Rigname+"_Bone_"+str(t)+".ckpt")
        tmpTime=time.time()
        tmp= sess.run(Y, feed_dict={X: Inn})
        TotalTime+=time.time()-tmpTime
        Vrt[v]=np.reshape(tmp,(len(v),3))
            
    t+=1
print('DeepLBS timming',TotalTime)
################################################################################
#             Output
################################################################################

np.savetxt(path+Rigname+'_DeepLBSVrts.txt',LBSVrt.T+0.001*Vrt,delimiter=',')


