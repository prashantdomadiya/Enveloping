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

###############################################################
#####  Input #################################################
path='/media/prashant/DATA/MyCodes/codeFiles/LBS/'
Rigname='50020'
path=path+Rigname+'/'
Phi=np.loadtxt(path+Rigname+'_Phi.txt',delimiter=',')
SFrm=np.loadtxt(path+Rigname+'_Trnsfrms.txt',delimiter=',')
F=ReadTxt(path+Rigname+'_facz.txt')

VrtLBS=SFrm.dot(Phi)


PoseNum=7
WritteOBJ(VrtLBS[3*PoseNum:3*PoseNum+3].T,F,path+Rigname+'LBS')

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



d=np.loadtxt(path+Rigname+'_Deepvertz.txt',delimiter=',')
WritteOBJ(np.reshape(d[:,PoseNum],(NV,3)),F,path+Rigname+'Ori')
VrtSeg=ReadTxt(path+Rigname+'_VrtSeg.txt')


#################################################################

# Training Parameters
learning_rate = 0.01
num_steps = 1000
batch_size = 256

display_step = 100
examples_to_show = 10

# Network Parameters
num_h1=256
num_h2=256
t=0
VrtTest=np.zeros((3*NV,3))
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
    print(t)
    Vid=np.reshape(3*np.array([v]*3)+[[0],[1],[2]],3*len(v),'F')
    
    loss = tf.reduce_sum(tf.pow(d[Vid].T-(d_hat[Vid].T+Y), 2))
    print("LBS Error....", np.sum((d[Vid].T-d_hat[Vid].T)**2))
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

    
    init = tf.initialize_all_variables()#tf.global_variables_initializer()
    saver = tf.train.Saver()

    
    
    with tf.Session() as sess:

        sess.run(init)
        for i in range(1, num_steps+1):
            _, l = sess.run([optimizer, loss], feed_dict={X: Inn.T})
            if i % display_step == 0 or i == 1:
                print('Step %i: Minibatch Loss: %f' % (i, l))
        tmp= sess.run(Y, feed_dict={X: np.reshape(Inn[:,PoseNum],(1,12*NFrm))})
        VrtTest[v,:]=np.reshape(d_hat[Vid,0]+tmp,(len(v),3))
        save_path = saver.save(sess, "/media/prashant/DATA/3D_DATA/DeepSkiningWeights/"+Rigname+"_Bone_"+str(t)+".ckpt")
        print("Model saved in path: %s" % save_path)
    t+=1

    
    ####################################################################################

    """
    
    saver = tf.train.Saver()
    
    with tf.Session() as sess:
        
        saver.restore(sess, "/media/prashant/DATA/3D_DATA/DeepSkiningWeights/"+Rigname+"_Bone_"+str(t)+".ckpt")
        tmp= sess.run(Y, feed_dict={X: np.reshape(Inn[:,PoseNum],(1,12*NFrm))})
        VrtTest[v,:]=np.reshape(d_hat[Vid,PoseNum]+tmp,(len(v),3))
            
    t+=1
    """
    
print(time.time()-strttime)
WritteOBJ(VrtTest,F,path+Rigname+'Test')
 


