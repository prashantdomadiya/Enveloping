
bl_info = {
    "name": "Rigging Tool using Rotational Regression",
    "author": "Prashant Domadiya",
    "version": (1, 3),
    "blender": (2, 82, 0),
    "location": "Python Console > Console > Run Script",
    "description": "Rig the Animation using RR",
    "warning": "",
    "wiki_url": "",
    "tracker_url": "",
    "category": "Development"}

LibPath='/home/prashant/anaconda3/envs/Blender282/lib/python3.8/site-packages/'
FilePath='/media/prashant/DATA/MyCodes/codeFiles/RR/'

import sys
import os
from os.path import join
sys.path.append(LibPath)

import bpy
import bmesh as bm
import numpy as np

from scipy import sparse as sp
from sksparse import cholmod as chmd
from scipy.sparse.linalg import inv
import scipy.linalg as scla

from functools import reduce,partial
from multiprocessing import Pool
import time

import itertools as it
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
        scn = bpy.context.collection.objects.link(ob)
        me.from_pydata(E[:,3*i:3*i+3], [], F)
        me.update()
        
def UpdateMesh(Vertz):
    Selected_Meshes=bpy.context.selected_objects
    for i in range(len(Selected_Meshes)):
        bpy.context.scene.objects.active = Selected_Meshes[-i-1]
        obj = bpy.context.active_object
        j=0
        for v in obj.data.vertices:
            v.co=Vertz[j]#Vertz[3*j:3*j+3,i]
            j+=1

def ConnectionMatrices(fcs,NV,NF):
    A=sp.lil_matrix((NV,2*NF))
    t=0
    for f in fcs:
        A[f,2*t:2*t+2]=np.array([[-1,-1],[1,0],[0,1]])
        t+=1
    return A
            
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
    w=D/Angl
    C=np.cos(Angl)
    S=np.sin(Angl)     
    T=1-C
    R=np.array([[T*(w[0]**2)+C,T*w[0]*w[1]-S*w[2], T*w[0]*w[2]+S*w[1]],
                [T*w[0]*w[1]+S*w[2],T*(w[1]**2)+C,T*w[1]*w[2]-S*w[0]],
                [T*w[0]*w[2]-S*w[1],T*w[1]*w[2]+S*w[0],T*(w[2]**2)+C]])
    return R


def ComputeWeights(wghts,Vecs):
    Beta=2*0.1
    W=Beta*(wghts.T).dot(wghts)
    b=Beta*(wghts.T).dot(Vecs)
    Nw=len(wghts.T)
    Wt=np.array([0.5]*Nw)
    err=10000
    maxitr=100
    k=0
    while (err>1 and k<maxitr): #for k in range(10):#
        Wt=Wt-W.dot(Wt)+b
        for i in range(Nw):
            if Wt[i]<0:
                Wt[i]=0.0
        err=np.linalg.norm(Vecs-wghts.dot(Wt))
        k+=1
    return Wt
    
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

def Multiply(X,Y):
    if len(X)==np.size(X):
        X=np.reshape(X,(1,len(X)))
    
    if len(Y)==np.size(Y):
        Y=np.reshape(Y,(len(Y),1))
    
    Xl=len(X)
    Yl=len(Y.T)
    Z=np.zeros((Xl,Yl))
    for i in range(Xl):
        x=X[i]
        for j in range(Yl):
            y=Y[:,j]
            Z[i,j]=np.sum(x*y)
    if Yl==1:
        Z=Z[:,0]
    if Xl==1:
        Z=Z[0]
    return Z
            




##############################################################################################
#                      Global Variable
##############################################################################################
path=''
if os.path.isfile(FilePath+'/'+'Riglist.txt'):
    RigList=ReadStringList(FilePath+'/'+'Riglist.txt')
else:
    RigList=[]

    
##############################################################################################
#                                   Tools
##############################################################################################
    
class ToolsPanel(bpy.types.Panel):
    bl_label = "Animation Tool"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
 
    def draw(self, context):
        self.layout.label(text="Set Rig Parameters")
        self.layout.prop(context.scene,"RigName")
        self.layout.label(text="Assign Skeleton to Mesh")
        self.layout.operator("asgn.skel",text='Combine').seqType='combine'
        self.layout.operator("asgn.skel",text='Relate').seqType='relate'
        self.layout.label(text="Learn Weight")
        self.layout.operator("get.seq",text='Mesh Seq').seqType="mesh"
        self.layout.operator("rig.tool",text='Rig').seqType="rig"


#######################################################################################################
#                 Assign Skeleton to Mesh
#######################################################################################################
Len=[]
#SE=[]

class AssignSkeleton(bpy.types.Operator):
    bl_idname = "asgn.skel"
    bl_label = "Target Sequence"
    seqType : bpy.props.StringProperty()
 
    def execute(self, context):
        global path, E, Len, RigList,FilePath
        
        Rigname=context.scene.RigName
        if self.seqType=="combine":
            Selected_Meshes=bpy.context.selected_objects
            
            for i in range(2):
                bpy.context.view_layer.objects.active = Selected_Meshes[i]
                obj = bpy.context.active_object
                if len(obj.data.polygons)!=0:
                    MeshIndx=i
                    
                else:
                    SkelIndx=i
            V=[]
            E=[]
            F=[]
            SE=[]
            
            bpy.context.view_layer.objects.active = Selected_Meshes[MeshIndx]
            obj = bpy.context.active_object
            for v in obj.data.vertices:
                V.append(list(obj.matrix_world @ v.co))
            for e in obj.data.edges:
                E.append(list(e.vertices[:]))
            for f in obj.data.polygons:
                F.append(list(f.vertices[:]))
            Len.append(len(V))
            
            bpy.context.view_layer.objects.active = Selected_Meshes[SkelIndx]
            obj = bpy.context.active_object
            for v in obj.data.vertices:
                V.append(list(obj.matrix_world @ v.co))
            for e in obj.data.edges:
                tmp=[]
                for j in e.vertices[:]:
                    tmp.append(j+Len[0])
                SE.append(list(e.vertices[:]))
                E.append(tmp)
                     
            Len.append(len(V)-Len[0])     
            
            me = bpy.data.meshes.new('MyRig')
            ob = bpy.data.objects.new('Myrig', me)
            bpy.context.collection.objects.link(ob)
            me.from_pydata(V, E, F)
            me.update()
            if os.path.exists(FilePath+Rigname)==False:
                os.mkdir(FilePath+Rigname)
            path=FilePath+Rigname+'/'
            WriteAsTxt(path+Rigname+'_SkelEdgz.txt',SE)
            
        else:
            if (Rigname in RigList)==False:
                RigList.append(Rigname)
            print(len(RigList))
            
            NvrtM=Len[0]
            NvrtS=Len[1]
            Ne=len(E)
            obj = bpy.context.active_object
            Joints=[[]]*NvrtS
            i=0
            for e in obj.data.edges:
                if i>=Ne:
                    edg=list(e.vertices[:])
                    
                    if (edg[1]-NvrtM)>=0:
                        Joints[edg[1]-NvrtM]=Joints[edg[1]-NvrtM]+[edg[0]]
                i+=1
            
            WriteAsTxt(path+Rigname+'_Joints.txt',Joints)
            WriteStringList(FilePath+'Riglist.txt',RigList)
            
        return{'FINISHED'}

class GetMeshSeq(bpy.types.Operator):
    bl_idname = "get.seq"
    bl_label = "Target Sequence"
    seqType : bpy.props.StringProperty()
 
    def execute(self, context):
        global path,FilePath
        Rigname=context.scene.RigName
        if path[len(path)-len(Rigname)-1:len(path)-1]!=Rigname:
            path=FilePath+Rigname+'/'        
        Selected_Meshes=bpy.context.selected_objects
        obj = bpy.context.active_object
        
        F=[]
        for f in obj.data.polygons:
            F.append(list(f.vertices))
        
        V=np.zeros([3*len(Selected_Meshes),len(obj.data.vertices)])
        NF=len(F)
        Nrml=np.zeros([3*NF,len(Selected_Meshes)])
        
        NPs=len(Selected_Meshes)
        
        for i in range(len(Selected_Meshes)):
            bpy.context.view_layer.objects.active = Selected_Meshes[i]
            obj = bpy.context.active_object
             
            t=0
            for v in obj.data.vertices:
                co_final= obj.matrix_world @ v.co
                V[3*i:3*i+3,t]=np.array([co_final.x,co_final.z,-co_final.y])
                t+=1
            t=0
            for f in obj.data.polygons:
                TmpN=obj.matrix_world.to_3x3() @ f.normal    
                Nrml[t:t+3,i]=[TmpN[0],TmpN[2],-TmpN[1]]
                t+=3
                
           
        np.savetxt(path+Rigname+'_vertz.txt',V,delimiter=',')
        #np.savetxt(path+Rigname+'Normal.txt',np.array(Nrml),delimiter=',')
        WriteAsTxt(path+Rigname+'_facz.txt',F)
       
        return{'FINISHED'}


class RigAnimation(bpy.types.Operator):
    bl_idname = "rig.tool"
    bl_label = "Rig The Animattion"
    seqType : bpy.props.StringProperty()
 
    def execute(self, context):
        global Phi, path,FilePath
        ################# Input from user ###############
        Rigname=context.scene.RigName
        if path[len(path)-len(Rigname)-1:len(path)-1]!=Rigname:
            path=FilePath+Rigname+'/'
        ################################################
        Vrt=np.loadtxt(path+'/'+'vertz.txt',delimiter=',')
        Fcs=ReadTxt(path+'/'+Rigname+'_facz.txt')
        Nrml=np.loadtxt(path+'/'+'Normal.txt',delimiter=',')
        
        
        NPs,NV=np.shape(Vrt)
        NPs=NPs//3
        NF=len(Fcs)

        AA=ConnectionMatrices(Fcs,NV,NF)
        factor=chmd.cholesky_AAt(AA[:(NV-1),:].tocsc())
                
        ########################## Comute Skeleton Frames ########################################### 
        Joints=ReadTxt(path+'/'+Rigname+'_Joints.txt')
        Links=ReadTxt(path+'/'+Rigname+'_SkelEdgz.txt')
        
        VecIndx=[]
        for ln in Links:
            VecIndx.append(ln)
         
        print("Computing Skeleton Frames.....")
        NJ=len(Joints)
          
        
        NFrm=len(VecIndx)
        SFrm=np.zeros([3,3])
        SrFrm=np.zeros([3,3*NFrm])
        
        A=np.zeros([NJ,3])
        RS=np.zeros((3*NFrm,NPs-1))
        AS=np.zeros((NFrm,NPs-1))
        
        for ps in range(NPs):
            
            X=Vrt[3*ps:3*ps+3,:].T
            for r in range(NJ):
                A[r]=np.mean(X[Joints[r]],axis=0)      
            if ps==0:
                np.savetxt(path+'/'+Rigname+'_RefSkel.txt',A,delimiter=',')
            
            t=0
            for r in range(NFrm):
                
                V=(A[VecIndx[r][0]]-A[VecIndx[r][1]])/np.linalg.norm(A[VecIndx[r][0]]-A[VecIndx[r][1]])
                
                tmp=X[Joints[VecIndx[r][0]][0]]-A[VecIndx[r][0]]
                B=np.cross(V,tmp)/np.linalg.norm(np.cross(V,tmp))
                N=np.cross(B,V)/np.linalg.norm(np.cross(B,V))

                SFrm[:,0]=V
                SFrm[:,1]=B
                SFrm[:,2]=N
               
                if ps!=0:
                    Axs,Angl=RotMatToAnglAxis(np.dot(SFrm,SrFrm[:,3*r:3*r+3].T))
                    RS[3*r:3*r+3,ps-1]=Axs
                    AS[r,ps-1]=Angl
                else:
                    SrFrm[:,3*r:3*r+3]=SFrm
        np.savetxt(path+'/'+Rigname+'_RefFrames.txt',SrFrm,delimiter=',')

        
        ###################  Clustering ###################
        C=ReadTxt(path+'/'+'ClusterKmeans.txt')
        Ncl=max(C[0])+1
        Cls=[[]]*Ncl
        t=0
        for i in C[0]:
            Cls[i]=Cls[i]+[t]
            t+=1

        
        
        Vbar=np.zeros((6*(NPs-1),NF))
        Vref=np.zeros((6,NF))
        for i in range(NF):
            f=Fcs[i]
            for ps in range(NPs):
                tmp=Vrt[3*ps:3*ps+3,f[1:]].T-Vrt[3*ps:3*ps+3,f[0]]
                if ps!=0:
                    Vbar[6*(ps-1):6*(ps-1)+6,i]=np.ravel(tmp)
                else:
                    Vref[:,i]=np.ravel(tmp)
                
        ############ initialize Dl##################
        Dl=np.zeros((3*(NPs-1),3*Ncl))
        for j in range(Ncl):
            c=Cls[j]
            for ps in range(1,NPs):
                
                RefFrm=np.reshape(Vref[:,c],(3,2*len(c)),'F')                    
                DefFrm=np.reshape(Vbar[6*(ps-1):6*(ps-1)+6,c],(3,2*len(c)),'F')
            
                X=Multiply(RefFrm,RefFrm.T)
                Y=Multiply(DefFrm,RefFrm.T)
                Dl[3*(ps-1):3*(ps-1)+3,3*j:3*j+3]=Y.dot(np.linalg.inv(X))
        ###### Compute Weights #####################
        
       
        Q=np.zeros((3,2*(NPs-1)))
        Bt=np.zeros(NF)
        t=0
        for c in Cls:
            for f in c:
                for ps in range(NPs-1):
                    tmp=np.reshape(Vref[:,f],(3,2),'F')
                    Q[:,2*ps:2*ps+2]=Dl[3*ps:3*ps+3,3*t:3*t+3].dot(tmp)
                Vc=np.reshape(Vbar[:,f],(3,2*(NPs-1)),'F')
                Bt[f]=np.mean(np.linalg.norm(Vc,axis=0)/np.linalg.norm(Q,axis=0))
            t+=1
        ##################################################################################
        #                        Connect Meshes
        ##################################################################################
        print("Computing Mesh Frames.....")
        RM=np.zeros((3*Ncl,NPs-1))
        AM=np.zeros((Ncl,NPs-1))
        Sc=np.zeros((9*Ncl,NPs-1))
        
        #######################################
        JntPrFace=2
        W=np.zeros((3*Ncl,3*JntPrFace))
        H=np.zeros((9*Ncl,3*JntPrFace+1))
        JntInd=[]
        Wgts=np.zeros((Ncl,JntPrFace))
        #######################################
        Wtmp=np.zeros((3,3*NFrm))
        Theta=np.ones((3*JntPrFace+1,NPs-1))
        #Dl=np.zeros((3*(NPs-1),3*Ncl))
        #Bt=np.zeros(NF)
        
        for c in range(Ncl):
            for ps in range(NPs-1):
                Q=Dl[3*ps:3*ps+3,3*c:3*c+3]
                R,S=scla.polar(Q)
                Axs,Angl=RotMatToAnglAxis(R)
                    
                RM[3*c:3*c+3,ps]=Axs
                AM[c,ps]=Angl
                Sc[9*c:9*c+9,ps]=np.ravel(S)
            
            Err=[] 
            for r in range(NFrm):
                U,Lmda,V=np.linalg.svd((RS[3*r:3*r+3]).dot(RM[3*c:3*c+3].T))
                Wtmp[:,3*r:3*r+3]=(V.T).dot(U.T)
                Err.append(np.linalg.norm((Wtmp[:,3*r:3*r+3].dot(RS[3*r:3*r+3]))-RM[3*c:3*c+3]))
            
             
            IndSort=np.argsort(Err)
            JntInd.append(list(IndSort[0:JntPrFace]))
        
            TotalErr=sum([(1/Err[i]) for i in IndSort[0:JntPrFace]])
            for i in range(JntPrFace):
                r=IndSort[i]
                u=0
                l=0
                for j in range(NPs-1):
                    if AS[r,j]!=0:
                        u+=AM[c,j]/AS[r,j]
                        l+=1
                u=u/l
                W[3*c:3*c+3,3*i:3*i+3]=u*Wtmp[:,3*r:3*r+3]
                Wgts[c,i]=1/(Err[r]*TotalErr)
                Theta[3*i:3*i+3]=RS[3*r:3*r+3]*AS[r]
            H[9*c:9*c+9]=Sc[9*c:9*c+9].dot(np.linalg.pinv(Theta))

        ################################################################
        #         Forming matrix for Editing
        ################################################################
        B=sp.lil_matrix((3*Ncl,2*NF))
        for c in range(Ncl):
            for f in Cls[c]:
                B[3*c:3*c+3,2*f:2*f+2]=Bt[f]*np.reshape(Vref[:,f],(3,2),'F')
        
        L=(factor((AA[:(NV-1),:].dot(B.transpose())).tocsc())).transpose()
        np.savetxt(path+Rigname+'_L.txt',L.toarray(),delimiter=',')
        np.savetxt(path+Rigname+'_RRWght.txt',W,delimiter=',')
        np.savetxt(path+Rigname+'_HRWght.txt',H,delimiter=',')
        np.savetxt(path+Rigname+'_JointClsWeights.txt',Wgts,delimiter=',')
        WriteAsTxt(path+Rigname+'_JointClsRl.txt',JntInd)
        
        return{'FINISHED'}


def register():
    bpy.utils.register_class(ToolsPanel)
    bpy.utils.register_class(AssignSkeleton)
    bpy.types.Scene.RigName=bpy.props.StringProperty(name="RigName", description="", default="Default")
    bpy.utils.register_class(GetMeshSeq)
    bpy.utils.register_class(RigAnimation)
    
    
def unregister():
    bpy.utils.unregister_class(ToolsPanel)
    bpy.utils.unregister_class(AssignSkeleton)
    bpy.utils.unregister_class(GetMeshSeq)
    bpy.utils.unregister_class(RigAnimation)
    del bpy.types.Scene.RigName
    
    
if __name__ == "__main__":  
    bpy.utils.register_module(__name__) 


