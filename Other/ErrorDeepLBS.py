import numpy as np
import tensorflow as tf
import time

def WritteOBJ(vert,fc,filepath):
    filepath=filepath+'.obj'
    with open(filepath, 'w') as f:
        f.write("# OBJ file\n")
        for v in vert:
            f.write("v %.4f %.4f %.4f\n" % (v[0], v[1], v[2]))
        for p in fc:
            f.write("f")
            for i in p:
                f.write(" %d" % (i + 1))
            f.write("\n")

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

def WriteQuanMaric(Name,QuantValue,QuantName):
    Ln=len(QuantName)
    with open(Name, 'w') as fl:
        for i in range(Ln):
            fl.write(" %s" % QuantName[i])
            fl.write(" %f" % QuantValue[i])
            fl.write("\n")
            
###############################################################
#####  Input #################################################
path='/media/prashant/DATA/MyCodes/codeFiles/LBS/'
Rigname='50002'
path=path+Rigname+'/'
Phi=np.loadtxt(path+Rigname+'_Phi.txt',delimiter=',')
SFrm=np.loadtxt(path+'SkeletonFrame.txt',delimiter=',')

VrtLBS=SFrm.dot(Phi)

NPs,NV=np.shape(VrtLBS)
NPs=NPs//3
NFrm=len(SFrm.T)//4
d_hat=np.zeros((3*NV,NPs))
Inn=np.zeros((12*NFrm,NPs))
for i in range(NPs):
    d_hat[:,i]=np.reshape(VrtLBS[3*i:3*i+3],3*NV,'F')
    for j in range(NFrm):
        Inn[12*j:12*j+9,i]=np.reshape(SFrm[3*i:3*i+3,4*j:4*j+3],9)
        Inn[12*j+9:12*j+12,i]=SFrm[3*i:3*i+3,4*j+3]

VrtSeg=ReadTxt(path+Rigname+'_VrtSeg.txt')

d=np.loadtxt(path+'RefVrtErr.txt',delimiter=',')

#################################################################



# Network Parameters
num_h1=256
num_h2=256
t=0
VrtDeepLBS=np.zeros((3*NV,NPs))
strttime=time.time()
for v in VrtSeg:
    num_input=12*NFrm#12*len(Bns)#
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


    # Building the encoder
    def DeepLBS(x):
        layer_1 = tf.tanh(tf.add(tf.matmul(x,weights['l1']),biases['b1']))
        layer_2 = tf.tanh(tf.add(tf.matmul(layer_1,weights['l2']),biases['b2']))
        layer_3 = tf.add(tf.matmul(layer_2,weights['l3']),biases['b3'])
        return layer_3

    # Construct model
    Y=DeepLBS(X)
    
    Vid=np.reshape(3*np.array([v]*3)+[[0],[1],[2]],3*len(v),'F')

    ########################### Coputing Error  ####################################
    saver = tf.train.Saver()
    
    with tf.Session() as sess:
        
        saver.restore(sess, "/media/prashant/DATA/3D_DATA/DeepSkiningWeights/"+Rigname+"_Bone_"+str(t)+".ckpt")
        #tmp= sess.run(Y, feed_dict={X: Inn.T})
        VrtDeepLBS[Vid,:]=0.01*sess.run(Y, feed_dict={X: Inn.T}).T+d_hat[Vid]
    t+=1

#################################################################################
print("Computing Error ............")

VrtViceErr=np.zeros((NV,NPs))
print(NPs)
for i in range(NPs):
    Ref=d[3*i:3*i+3].T
    Est=np.reshape(VrtDeepLBS[:,i],(NV,3))
    VrtViceErr[:,i]=np.linalg.norm(Ref-Est,axis=1)
Err=np.mean(VrtViceErr,axis=1)*1000
QuantValue=[np.mean(Err),np.std(Err),np.max(Err),np.max(np.std(VrtViceErr,axis=1))*1000,
            np.argmax(Err),np.argmax(np.std(VrtViceErr,axis=1)),np.min(Err),np.min(np.std(VrtViceErr,axis=1))*1000,
            np.argmin(Err),np.argmin(np.std(VrtViceErr,axis=1))]
QuantName=['Mean','Standard deviation', 'Max Mean', 'Max Std','Max Mean Vert Id','Max Std Vert Id',
           'Min Mean', 'Min Std','Min Mean Vert Id','Min Std Vert Id']

OutPath='/media/prashant/DATA/MyCodes/codeFiles/DeepLBS/'
WriteQuanMaric(OutPath+Rigname+'_DeepLBSQuantitativeValues.txt',QuantValue,QuantName)
MaxErr=np.max(VrtViceErr,axis=1)*1000
np.savetxt(OutPath+Rigname+'_DeepLBSErr.txt',VrtViceErr,delimiter=',')

    
    


