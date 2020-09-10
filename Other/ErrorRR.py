
bl_info = {
    "name": "Error RR",
    "author": "Prashant Domadiya",
    "version": (1, 3),
    "blender": (2, 71, 0),
    "location": "Python Console > Console > Run Script",
    "description": "Compute MSE for RR",
    "warning": "",
    "wiki_url": "",
    "tracker_url": "",
    "category": "Development"}

import sys
sys.path.append('/home/prashant/anaconda3/envs/Blender269/lib/python3.4/site-packages')

import bpy
import bmesh
import numpy as np

import os
from os.path import join

import os
from scipy import sparse as sp
from sksparse import cholmod as chmd
from scipy.sparse.linalg import inv

from functools import reduce,partial
from multiprocessing import Pool
import time
import itertools as itr
import shutil




#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#                   Display Mesh
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def CreateMesh(V,F,NPs):
    E=np.zeros(np.shape(V))
    F = [ [int(i) for i in thing] for thing in F]
    for i in range(NPs):
        E[:,3*i]=V[:,3*i]
        E[:,3*i+1]=-V[:,3*i+2]
        E[:,3*i+2]=V[:,3*i+1]
        me = bpy.data.meshes.new('MyMesh')
        ob = bpy.data.objects.new('Myobj', me)
        scn = bpy.context.scene
        scn.objects.link(ob)
        me.from_pydata(E[:,3*i:3*i+3], [], F)
        me.update()
        
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    #               Face Transformation
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def RotMatToAnglAxis(R):
    M=np.trace(R)-1
    if M>2:
        M=1.99
    elif M<-2:
        M=-1.99
    else:
        M=M/1
            
    angl=np.arccos(M/2)
    if np.sin(angl)==0:
        Axs=np.zeros(3)
    else:
        Axs=(1/(2*np.sin(angl)))*np.array([R[2,1]-R[1,2], R[0,2]-R[2,0], R[1,0]-R[0,1]])
    return Axs,angl

def AnglAxisToRotMat(D):
    Angl=np.linalg.norm(D)
    if Angl!=0.0:
        w=D/Angl
        C=np.cos(Angl)
        S=np.sin(Angl)     
        T=1-C
        R=np.array([[T*(w[0]**2)+C,T*w[0]*w[1]-S*w[2], T*w[0]*w[2]+S*w[1]],
                    [T*w[0]*w[1]+S*w[2],T*(w[1]**2)+C,T*w[1]*w[2]-S*w[0]],
                    [T*w[0]*w[2]-S*w[1],T*w[1]*w[2]+S*w[0],T*(w[2]**2)+C]])

    else:
        R=np.eye(3)
    return R

def WriteAsTxt(Name,Vec):
    with open(Name, 'w') as fl:
        for i in Vec:
            if str(type(i))=="<class 'list'>":
                for j in i:
                    fl.write(" %d" % j)
                fl.write("\n")
            else:
                fl.write(" %d" % i)

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
def WriteStringList(Name, Lst):
    with open(Name, 'w') as fl:
        for i in Lst:
            fl.write(" %s" % i)
            fl.write("\n")

def ReadStringList(Name):
    Lst=[]
    fl = open(Name, 'r')
    NumLine=0
    for line in fl:
        words = line.split()
        Lst=Lst+words
    return Lst



def WriteObj(V,F,PoseN):
    InputPath='/media/prashant/DATA/3D_DATA/SkinningData/'
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
##############################################################################################
#                      Global Variable
##############################################################################################
def LoadRigWeight(self, context):
    global  ActiveModel,path,Lbspath,L,W,H,Wgts,JntInd,Links,SFrm,RefFrm,AxsAngl,TwstAxsAngl
    global LbsSFrm,TwistLbsFrm,CnsVrt,FrVrt,K,LBSPhi
    if len(RigList)!=0:
        ActiveModel=RigList[context.scene.ModelErr]
        print('Loading Model '+ActiveModel)
        path=Mainpath+ActiveModel+'/'
        Lbspath=MainLbspath+ActiveModel+'/'
        L=np.loadtxt(path+ActiveModel+'_L.txt',delimiter=',')
        
        W=np.loadtxt(path+ActiveModel+'_RRWght.txt',delimiter=',')
        H=np.loadtxt(path+ActiveModel+'_HRWght.txt',delimiter=',')
        Wgts=np.loadtxt(path+ActiveModel+'_JointClsWeights.txt',delimiter=',')
        JntInd=ReadTxt(path+ActiveModel+'_JointClsRl.txt')
        Links=ReadTxt(path+ActiveModel+'_SkelEdgz.txt')
        RefFrm=np.loadtxt(Lbspath+ActiveModel+'_RefFrames.txt',delimiter=',')
        
        AxsAngl=np.zeros((len(RefFrm.T)//3,3))
        TwstAxsAngl=np.zeros((len(RefFrm.T)//3,3))

        ######  LBS  #################
        LbsSFrm=1*RefFrm
        TwistLbsFrm=1*LbsSFrm
        CnsVrt=ReadTxt(path+ActiveModel+'_LBSVrt.txt')[0]
        FrVrt=ReadTxt(path+ActiveModel+'_FrVrt.txt')[0]
        K=np.loadtxt(path+ActiveModel+'_K.txt',delimiter=',')
        LBSPhi=np.loadtxt(path+ActiveModel+'_LBSPhi.txt',delimiter=',')
    return

Mainpath='/media/prashant/DATA/MyCodes/codeFiles/RR/'
MainLbspath='/media/prashant/DATA/MyCodes/codeFiles/LBS/'

if os.path.isfile(Mainpath+'Riglist.txt'):
    RigList=ReadStringList(Mainpath+'Riglist.txt')
else:
    RigList=[]

if len(RigList)!=0:
    ActiveModel=RigList[0]
    path=Mainpath+ActiveModel+'/'
    Lbspath=MainLbspath+ActiveModel+'/'
    L=np.loadtxt(path+RigList[0]+'_L.txt',delimiter=',')
    W=np.loadtxt(path+RigList[0]+'_RRWght.txt',delimiter=',')
    H=np.loadtxt(path+RigList[0]+'_HRWght.txt',delimiter=',')
    Wgts=np.loadtxt(path+RigList[0]+'_JointClsWeights.txt',delimiter=',')
    JntInd=ReadTxt(path+RigList[0]+'_JointClsRl.txt')
    Links=ReadTxt(path+RigList[0]+'_SkelEdgz.txt')
    RefFrm=np.loadtxt(Lbspath+RigList[0]+'_RefFrames.txt',delimiter=',')
    AxsAngl=np.zeros((len(RefFrm.T)//3,3))
    TwstAxsAngl=np.zeros((len(RefFrm.T)//3,3))
    
    ######  LBS ##################
    LbsSFrm=1*RefFrm
    TwistLbsFrm=1*LbsSFrm
    CnsVrt=ReadTxt(path+RigList[0]+'_LBSVrt.txt')[0]
    FrVrt=ReadTxt(path+RigList[0]+'_FrVrt.txt')[0]
    K=np.loadtxt(path+RigList[0]+'_K.txt',delimiter=',')
    LBSPhi=np.loadtxt(path+RigList[0]+'_LBSPhi.txt',delimiter=',')
else:
    print("First Rig the Models")
##############################################################################################
#                                   Tools
##############################################################################################
    
class ToolsPanel(bpy.types.Panel):
    bl_label = "Error calculation Tools"
    bl_space_type = "VIEW_3D"
    bl_region_type = "TOOLS"
 
    def draw(self, context):
        
        self.layout.prop(context.scene,"ModelErr")

        self.layout.label("Animator")
        self.layout.operator("get.mesh",text='Mesh Seq').seqType="Mesh"
        self.layout.operator("animate.tool",text='Animate').seqType="animate"
        self.layout.operator("err.cmpt",text='Error').seqType="error"
        

class GetMeshSeq(bpy.types.Operator):
    bl_idname = "get.mesh"
    bl_label = "Load Skeleton"
    seqType = bpy.props.StringProperty()
 
    def execute(self, context):
        global path,Lbspath, ActiveModel,Links,JntInd
        
        Selected_Meshes=bpy.context.selected_objects
        obj = bpy.context.active_object

        Joints=ReadTxt(path+ActiveModel+'_Joints.txt')
        RefSFrm=np.loadtxt(Lbspath+ActiveModel+'_RefFrames.txt',delimiter=',')
        VecIndx=[]
        for ln in Links:
            VecIndx.append(ln)
       
        NJ=len(Joints)
        NFrm=len(VecIndx)
        NV=len(obj.data.vertices)
        NPs=len(Selected_Meshes)
        
        LbsSFrm=np.zeros([3*NPs,4*NFrm])
        Vrt=np.zeros((NV,3))
        RefVrtErr=np.zeros((3*NPs,NV))
        A=np.zeros((NJ,3))

        AxsAngl=np.zeros((NFrm,3))
        Ncl=len(JntInd)
        JntPrFace=len(JntInd[0])
        G=np.zeros((3*NPs,3*Ncl))
        Theta=np.ones(7)
        for p in range(NPs):
            bpy.context.scene.objects.active = Selected_Meshes[-p-1]
            obj = bpy.context.active_object
             
            t=0
            for v in obj.data.vertices:
                co_final= obj.matrix_world*v.co
                Vrt[t]=np.array([co_final.x,co_final.z,-co_final.y])
                t+=1
            RefVrtErr[3*p:3*p+3]=Vrt.T
            for r in range(NJ):
                A[r]=np.mean(Vrt[Joints[r]],axis=0)

            for r in range(NFrm):
                
                V=(A[VecIndx[r][0]]-A[VecIndx[r][1]])/np.linalg.norm(A[VecIndx[r][0]]-A[VecIndx[r][1]])
                tmp=Vrt[Joints[VecIndx[r][0]][0]]-A[VecIndx[r][0]]
                B=np.cross(V,tmp)/np.linalg.norm(np.cross(V,tmp))
                N=np.cross(B,V)/np.linalg.norm(np.cross(B,V))
                LbsSFrm[3*p:3*p+3,4*r]=V
                LbsSFrm[3*p:3*p+3,4*r+1]=B
                LbsSFrm[3*p:3*p+3,4*r+2]=N
                LbsSFrm[3*p:3*p+3,4*r+3]=A[VecIndx[r][0]]
                R=LbsSFrm[3*p:3*p+3,4*r:4*r+3].dot(np.linalg.inv(RefSFrm[:,4*r:4*r+3]))
                Axs,Angl=RotMatToAnglAxis(R)
                AxsAngl[r]=Angl*Axs
            
            for c in range(Ncl):
                Ind=JntInd[c]
                R=0
                for i in range(JntPrFace):
                    K=W[3*c:3*c+3,3*i:3*i+3].dot(AxsAngl[Ind[i]])
                    R=R+Wgts[c,i]*AnglAxisToRotMat(K)
                    Theta[3*i:3*i+3]=AxsAngl[Ind[i]]
                G[3*p:3*p+3,3*c:3*c+3]=R.dot(np.reshape(H[9*c:9*c+9].dot(Theta),(3,3)))
        
        np.savetxt(path+'RefVrtErr.txt',RefVrtErr,delimiter=',')
        np.savetxt(path+'G.txt',G,delimiter=',')
        np.savetxt(path+'LBS_SkeletonFrame.txt',LbsSFrm,delimiter=',')
        
        return{'FINISHED'}


class Animation(bpy.types.Operator):
    bl_idname = "animate.tool"
    bl_label = "Compute Animation"
    seqType = bpy.props.StringProperty()
 
    def execute(self, context):
        global L,LBSPhi,K, path, ActiveModel,CnsVrt,FrVrt
        G=np.loadtxt(path+'G.txt',delimiter=',')
        LbsSFrm=np.loadtxt(path+'LBS_SkeletonFrame.txt',delimiter=',')
        
        Fcs=ReadTxt(path+ActiveModel+'_facz.txt')
        NPs=len(LbsSFrm)//3
        strttime=time.time()
        X=np.zeros((3*NPs,len(CnsVrt)+len(FrVrt)))
        for i in range(NPs):
            X[3*i:3*i+3,CnsVrt]=LbsSFrm[3*i:3*i+3].dot(LBSPhi)
            X[3*i:3*i+3,FrVrt]=G[3*i:3*i+3].dot(L)-X[3*i:3*i+3,CnsVrt].dot(K)
        np.savetxt(path+'EstVrtErr.txt',X,delimiter=',')
        return{'FINISHED'}

class ErrCompute(bpy.types.Operator):
    bl_idname = "err.cmpt"
    bl_label = "Compute Animation"
    seqType = bpy.props.StringProperty()
 
    def execute(self, context):
        global path,ActiveModel

        Fcs=ReadTxt(path+ActiveModel+'_facz.txt')
        
        RefVrtErr=np.loadtxt(path+'RefVrtErr.txt',delimiter=',')
        EstVrtErr=np.loadtxt(path+'EstVrtErr.txt',delimiter=',')
        
        NPs,NVrt=np.shape(RefVrtErr)
        NPs=NPs//3
        VrtViceErr=np.zeros((NVrt,NPs))
        for i in range(NPs):
            Ref=RefVrtErr[3*i:3*i+3].T
            Est=EstVrtErr[3*i:3*i+3].T
            VrtViceErr[:,i]=np.linalg.norm(Ref-Est,axis=1)
        Err=np.mean(VrtViceErr,axis=1)*1000
        MaxErr=np.max(VrtViceErr,axis=1)*1000
        QuantValue=[np.mean(Err),np.std(Err),np.max(Err),np.max(np.std(VrtViceErr,axis=1))*1000,
                    np.argmax(Err),np.argmax(np.std(VrtViceErr,axis=1)),np.min(Err),np.min(np.std(VrtViceErr,axis=1))*1000,
                    np.argmin(Err),np.argmin(np.std(VrtViceErr,axis=1))]
        QuantName=['Mean','Standard deviation', 'Max Mean', 'Max Std','Max Mean Vert Id','Max Std Vert Id',
                   'Min Mean', 'Min Std','Min Mean Vert Id','Min Std Vert Id']
        WriteQuanMaric(path+ActiveModel+'_QuantitativeValues.txt',QuantValue,QuantName)
        np.savetxt(path+ActiveModel+'_Err.txt',VrtViceErr,delimiter=',')
        np.savetxt(path+ActiveModel+'_MaxErr.txt',MaxErr,delimiter=',')
        return{'FINISHED'}


def register():
    bpy.utils.register_class(ToolsPanel)
    bpy.types.Scene.ModelErr=bpy.props.IntProperty(name="Model", description="Select Rigged Model", default=0,
                                                min=0,max=(len(RigList)-1),options={'ANIMATABLE'}, update=LoadRigWeight)
    
    
    bpy.utils.register_class(GetMeshSeq)
    bpy.utils.register_class(ErrCompute)
    
    bpy.utils.register_class(Animation)
    
def unregister():
    bpy.utils.unregister_class(ToolsPanel)
    
    bpy.utils.unregister_class(GetMeshSeq)
    bpy.utils.unregister_class(ErrCompute)
    
    bpy.utils.unregister_class(Animation)
    del bpy.types.Scene.ModelErr
    
    
    
if __name__ == "__main__":  
    bpy.utils.register_module(__name__) 


