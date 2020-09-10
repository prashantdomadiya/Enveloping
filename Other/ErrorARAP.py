import numpy as np
from scipy import sparse as sp
from scipy.sparse import linalg as sl

###############################################################
#               User Input Panel
######################################################
Rigname='50002'
path='/media/prashant/DATA/MyCodes/codeFiles/LBSARAP/'+Rigname+'/'
Lbspath='/media/prashant/DATA/MyCodes/codeFiles/LBS/'+Rigname+'/'
#############################################################

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
            
def load_sparse_csc(filename):
    loader = np.load(filename)
    return sp.csc_matrix((loader['data'], loader['indices'], loader['indptr']),
                      shape=loader['shape'])


def ARAP(PsCns,Lbsvrt,H,K):
    # replacements
    NoCnstrVrt=len(PsCns)
    LV=len(Lbsvrt)
    ############## Transformation constraints#############
    #Equation MeqV=Peq
   
    Peq=np.zeros((NoCnstrVrt,3))
    t=0
    for cv in PsCns:
        Peq[t]=Lbsvrt[cv]
        t+=1
    
    b=np.zeros((LV+NoCnstrVrt,3))
    b[LV:]=Peq
    for itr in range(10):
        S=K.dot(Lbsvrt)
        RT=np.zeros(np.shape(S))
        for i in range(len(S)//3):
            U,Lmda,V=np.linalg.svd(S[3*i:3*i+3])
            RT[3*i:3*i+3]=U.dot(V)
        b[0:LV]=(K.transpose()).dot(RT)
        tmp=sl.spsolve(H,sp.csc_matrix(b))
        Lbsvrt=tmp[0:LV].toarray()
        
    return Lbsvrt


Phi=(np.loadtxt(Lbspath+Rigname+'_Phi.txt',delimiter=','))
SFrm=np.loadtxt(Lbspath+'SkeletonFrame.txt',delimiter=',')
LbsVrt=SFrm.dot(Phi)

H=load_sparse_csc(path+Rigname+'_H.npz')
K=load_sparse_csc(path+Rigname+'_K.npz')

d=np.loadtxt(Lbspath+'RefVrtErr.txt',delimiter=',')

NPs,NV=np.shape(d)
NPs=NPs//3

Fc=np.int32(ReadTxt(Lbspath+Rigname+'_facz.txt'))
Joints=ReadTxt(Lbspath+Rigname+'_Joints.txt')
PsCns=[]
for j in Joints:
    PsCns=PsCns+j

ArapVrt=np.zeros((NV,3*NPs))
for i in range(NPs):
    ArapVrt[:,3*i:3*i+3]=ARAP(PsCns,LbsVrt[3*i:3*i+3].T,H,K)

print("Computing Error ............")
VrtViceErr=np.zeros((NV,NPs))
for i in range(NPs):
    Ref=d[3*i:3*i+3].T
    Est=ArapVrt[:,3*i:3*i+3]
    VrtViceErr[:,i]=np.linalg.norm(Ref-Est,axis=1)
Err=np.mean(VrtViceErr,axis=1)*1000
Err=np.mean(VrtViceErr,axis=1)*1000
QuantValue=[np.mean(Err),np.std(Err),np.max(Err),np.max(np.std(VrtViceErr,axis=1))*1000,
            np.argmax(Err),np.argmax(np.std(VrtViceErr,axis=1)),np.min(Err),np.min(np.std(VrtViceErr,axis=1))*1000,
            np.argmin(Err),np.argmin(np.std(VrtViceErr,axis=1))]
QuantName=['Mean','Standard deviation', 'Max Mean', 'Max Std','Max Mean Vert Id','Max Std Vert Id',
           'Min Mean', 'Min Std','Min Mean Vert Id','Min Std Vert Id']
WriteQuanMaric(path+Rigname+'_QuantitativeValues.txt',QuantValue,QuantName)
MaxErr=np.max(VrtViceErr,axis=1)*1000
np.savetxt(path+Rigname+'_ARAPLBSErr.txt',VrtViceErr,delimiter=',')
np.savetxt(path+Rigname+'_MaxARAPLBSErr.txt',MaxErr,delimiter=',')

