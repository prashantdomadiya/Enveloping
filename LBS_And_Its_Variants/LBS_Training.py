
bl_info = {
    "name": "LBS",
    "author": "Prashant Domadiya",
    "version": (1, 1),
    "blender": (2, 82, 0),
    "location": "Python Console > Console > Run Script",
    "description": "Rig tool",
    "warning": "",
    "wiki_url": "",
    "tracker_url": "",
    "category": "Development"}

LibPath='/home/prashant/anaconda3/envs/Blender282/lib/python3.8/site-packages/'
FilePath='/media/prashant/DATA/MyCodes/codeFiles/LBS/'

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

from functools import reduce,partial
from multiprocessing import Pool
import time

import itertools as it
import shutil


def compute_mesh_laplacian(verts, tris,Cls, weight_type='cotangent'):
    n = len(verts)
    NE=3*len(tris)
    # we consider the triangle P, Q, R
    iP = tris[:, 0]
    iQ = tris[:, 1]
    iR = tris[:, 2]
    # edges forming the triangle
    PQ = verts[iP] - verts[iQ] # P--Q
    QR = verts[iQ] - verts[iR] # Q--R
    RP = verts[iR] - verts[iP] # R--P
    if weight_type == 'cotangent':
        # compute cotangent at all 3 points in triangle PQR
        cotP = -1 * (PQ * RP).sum(axis=1) / veclen(np.cross(PQ, RP)) # angle at vertex P
        cotQ = -1 * (QR * PQ).sum(axis=1) / veclen(np.cross(QR, PQ)) # angle at vertex Q
        cotR = -1 * (RP * QR).sum(axis=1) / veclen(np.cross(RP, QR)) # angle at vertex R

    # compute weights and indices
    if weight_type == 'cotangent':
        I =       np.concatenate((  iP,   iR,    iP,   iQ,    iQ,   iR))
        J =       np.concatenate((  iR,   iP,    iQ,   iP,    iR,   iQ))
        W = 0.5 * np.concatenate((cotQ, cotQ,  cotR, cotR,  cotP, cotP))
    
    elif weight_type == 'mean_value':
        # TODO: I didn't check this code yet
        PQlen = 1 / veclen(PQ)
        QRlen = 1 / veclen(QR)
        RPlen = 1 / veclen(RP)
        PQn = PQ * PQlen[:,np.newaxis] # normalized
        QRn = QR * QRlen[:,np.newaxis]
        RPn = RP * RPlen[:,np.newaxis]
        # TODO pretty sure there is a simpler solution to those 3 formulas
        tP = np.tan(0.5 * np.arccos((PQn * -RPn).sum(axis=1)))
        tQ = np.tan(0.5 * np.arccos((-PQn * QRn).sum(axis=1)))
        tR = np.tan(0.5 * np.arccos((RPn * -QRn).sum(axis=1)))
        I = np.concatenate((      iP,       iP,       iQ,       iQ,       iR,       iR))
        J = np.concatenate((      iQ,       iR,       iP,       iR,       iP,       iQ))
        W = np.concatenate((tP*PQlen, tP*RPlen, tQ*PQlen, tQ*QRlen, tR*RPlen, tR*QRlen))

    elif weight_type == 'uniform':
        
        I = np.concatenate((iP, iQ,  iQ, iR,  iR, iP))
        J = np.concatenate((iQ, iP,  iR, iQ,  iP, iR))
        W = np.ones(len(tris) * 6)

    
    L = sp.csr_matrix((W, (I, J)), shape=(n, n))
    if weight_type == 'uniform':
        
        L.data[:] = 1
    
    
    
    K=sp.lil_matrix((3*len(Cls),n))
    X=sp.lil_matrix((n,n))
    u=0
    for c in Cls:
        Ak=sp.lil_matrix((n,3*len(c)))
        
        #C=np.zeros((3*len(c),3*len(c)))
        t=0
        for i in c:
            f=tris[i]
            Ak[[f[0],f[1]],3*t]=[1.0,-1.0]
            Ak[[f[1],f[2]],3*t+1]=[1.0,-1.0]
            Ak[[f[2],f[0]],3*t+2]=[1.0,-1.0]
            #C[3*t,3*t]=L[f[0],f[1]]
            #C[3*t+1,3*t+1]=L[f[1],f[2]]
            #C[3*t+2,3*t+2]=L[f[2],f[0]]
            t+=1
        #X=X+(Ak.dot(C)).dot(Ak.T)
        X=X+(Ak.dot(Ak.transpose()))
        
        
        #K[3*u:3*u+3]=(Vrt.T).dot((Ak.dot(C)).dot(Ak.T))
        tmp=Ak.dot(Ak.transpose())
        K[3*u:3*u+3]=((tmp).dot(verts)).T
        u+=1
    return X,K
    
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



##############################################################################################
#                      Global Variable
##############################################################################################

path='/media/prashant/DATA/MyCodes/codeFiles/LBS/'

if os.path.isfile(path+'Riglist.txt'):
    RigList=ReadStringList(path+'Riglist.txt')
else:
    RigList=[]

    
##############################################################################################
#                                   Tools
##############################################################################################
    
class ToolsPanel(bpy.types.Panel):
    bl_label = "DT Tools Panel"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
 
    def draw(self, context):
        self.layout.label(text="Assign Skeleton to Mesh")
        self.layout.prop(context.scene,"RigNameLBS")
        self.layout.operator("asgn.skel",text='Combine').seqType='combine'
        self.layout.operator("asgn.skel",text='Relate').seqType='relate'

        self.layout.label(text="Learn Weight")
        self.layout.operator("get.seq",text='Mesh Seq').seqType="mesh"
        self.layout.operator("rig.tool",text='Rig').seqType="rig"
        self.layout.operator("rig.arap",text='RigARAP').seqType="rigarap"


#######################################################################################################
#                 Assign Skeleton to Mesh
#######################################################################################################
Len=[]


class AssignSkeleton(bpy.types.Operator):
    bl_idname = "asgn.skel"
    bl_label = "Target Sequence"
    seqType : bpy.props.StringProperty()
 
    def execute(self, context):
        global path, E, Len, RigList,FilePath
        
        Rigname=context.scene.RigNameLBS
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
        Rigname=context.scene.RigNameLBS
        if path[len(path)-len(Rigname)-1:len(path)-1]!=Rigname:
            path=FilePath+Rigname+'/'
        Selected_Meshes=bpy.context.selected_objects
        obj = bpy.context.active_object
        
        F=[]
        for f in obj.data.polygons:
            F.append(list(f.vertices))
        WriteAsTxt(path+Rigname+'_facz.txt',F)
        
        V=np.zeros([3*len(Selected_Meshes),len(obj.data.vertices)])
        #ddddddddddddddd DeepLBSdddddddddddddddd
        VDeep=np.zeros([3*len(obj.data.vertices),len(Selected_Meshes)-1])
        #ddddddddddddddddddddddddddddddddddddddd
        for i in range(len(Selected_Meshes)):
            bpy.context.view_layer.objects.active = Selected_Meshes[i]
            obj = bpy.context.active_object
             
            t=0
            for v in obj.data.vertices:
                co_final= obj.matrix_world @ v.co
                V[3*i:3*i+3,t]=np.array([co_final.x,co_final.z,-co_final.y])
                if i!=0:
                    VDeep[3*t:3*t+3,i-1]=np.array([co_final.x,co_final.z,-co_final.y])
                t+=1
        print(path)
        np.savetxt(path+Rigname+'vertz.txt',V,delimiter=',')
        #ddddddddddddd DeepLBS ddddddddddddddddddddddddd
        np.savetxt(path+Rigname+'_Deepvertz.txt',VDeep,delimiter=',')
        #ddddddddddddddddddddddddddddddddddddddddddddddd
        
        return{'FINISHED'}

class RigAnimation(bpy.types.Operator):
    bl_idname = "rig.tool"
    bl_label = "Rig The Animattion"
    seqType : bpy.props.StringProperty()
 
    def execute(self, context):
        global path, FilePath
        Rigname=context.scene.RigNameLBS
        #Rigname=context.scene.RigName
        if path[len(path)-len(Rigname)-1:len(path)-1]!=Rigname:
            path=FilePath+Rigname+'/' 
        Vrt=np.loadtxt(path+Rigname+'vertz.txt',delimiter=',')
        np.savetxt(path+Rigname+'_RefMesh.txt',Vrt[0:3,:].T,delimiter=',')
        
        NPs,NV=np.shape(Vrt)
        NPs=NPs//3
        
        ########################## Comute Skeleton Frames ########################################### 
        Joints=ReadTxt(path+Rigname+'_Joints.txt')
        Links=ReadTxt(path+Rigname+'_SkelEdgz.txt')
        
        print("Computing Skeleton Frames.....")
        NJ=len(Joints)

        VecIndx=[]
        for ln in Links:
            VecIndx.append(ln)
          
        
        NFrm=len(VecIndx)
        SFrm=np.eye(4)
        SrFrm=np.zeros([4,4*NFrm])
        
        A=np.zeros([NJ,3])
        RS=np.zeros((3*(NPs-1),4*NFrm))
        WC=np.array([[1.0,0,0,0],[0,1.0,0,0],[0,0,1.0,0]])
        strtTime=time.time()
        for ps in range(NPs):
            
            X=Vrt[3*ps:3*ps+3,:].T
            for r in range(NJ):
                A[r]=np.mean(X[Joints[r]],axis=0)
                  
            if ps==0:
                np.savetxt(path+Rigname+'_RefSkel.txt',A,delimiter=',')
            
            t=0
            for r in range(NFrm):
                V=(A[VecIndx[r][0]]-A[VecIndx[r][1]])/np.linalg.norm(A[VecIndx[r][0]]-A[VecIndx[r][1]])
                
                tmp=X[Joints[VecIndx[r][0]][0]]-A[VecIndx[r][0]]
                B=np.cross(V,tmp)/np.linalg.norm(np.cross(V,tmp))
                N=np.cross(B,V)/np.linalg.norm(np.cross(B,V))

                SFrm[0:3,0]=V
                SFrm[0:3,1]=B
                SFrm[0:3,2]=N
                SFrm[0:3,3]=A[VecIndx[r][0]]
                if ps!=0:
                    Xd=SFrm[0:3,:]#Xd
                    Xr=SrFrm[:,4*r:4*r+4]
                    RS[3*(ps-1):3*(ps-1)+3,4*r:4*r+4]=Xd.dot(np.linalg.inv(Xr))
                else:
                    SrFrm[:,4*r:4*r+4]=SFrm # Xr
        np.savetxt(path+Rigname+'_RefFrames.txt',SrFrm[0:3],delimiter=',')
        
        ########################## Comute Skeleton Frames ########################################### 
        print("Computing Mesh Frames.....")
        Phi=np.zeros((4*NFrm,NV))
        for v in range(NV):
            v0=np.append(Vrt[0:3,v],[1])
            Err=[]
            H=np.zeros((3*(NPs-1),NFrm))
            for f in range(NFrm):
                for p in range(NPs-1):
                    H[3*p:3*p+3,f]=RS[3*p:3*p+3,4*f:4*f+4].dot(v0)
                Err.append(np.linalg.norm(Vrt[3:,v]-H[:,f]))
               
            num_JntPrVrt=3
            IndSort=np.argsort(Err)
            SmlH=H[:,IndSort[0:num_JntPrVrt]]
            
            wghts=np.linalg.inv(np.dot(SmlH.T,SmlH)).dot(np.dot(SmlH.T,Vrt[3:,v]))
            
            for i in range(num_JntPrVrt):
                if wghts[i]<0.0:
                    wghts[i]=0.0
            wghts=wghts/np.sum(wghts)
            
            for i in range(num_JntPrVrt):
                r=IndSort[i]
                Xr=SrFrm[:,4*r:4*r+4]
                Phi[4*r:4*r+4,v]=wghts[i]*(np.linalg.inv(Xr).dot(v0))
        print(time.time()-strtTime)
        np.savetxt(path+Rigname+'_Phi.txt',Phi,delimiter=',')
        
        return{'FINISHED'}

class RigArap(bpy.types.Operator):
    bl_idname = "rig.arap"
    bl_label = "Rig The Animattion"
    seqType : bpy.props.StringProperty()
 
    def execute(self, context):
        global path, FilePath
        Rigname=context.scene.RigNameLBS
        if path[len(path)-len(Rigname)-1:len(path)-1]!=Rigname:
            path=FilePath+Rigname+'/'
            
        Fc=np.int32(ReadTxt(path+Rigname+'_facz.txt'))
        Joints=ReadTxt(path+Rigname+'_Joints.txt')
        tmp=ReadTxt(path+Rigname+'_ClusterKmeans.txt')
        Phi=sp.csc_matrix(np.loadtxt(path+Rigname+'_Phi.txt',delimiter=','))
        t=0
        Cls=[[]]*(max(tmp[0])+1)
        for c in tmp:
            for i in c:
                Cls[i]=Cls[i]+[t]
                t+=1

        InVrt=np.loadtxt(path+Rigname+'_RefMesh.txt',delimiter=',')
        strttime=time.time()
        L,K=compute_mesh_laplacian(InVrt,Fc,Cls,weight_type='uniform')

        PsCns=[]
        for j in Joints:
            PsCns=PsCns+j

        # replacements
        NoCnstrVrt=len(PsCns)
        LV=len(InVrt)
        ############## Transformation constraints#############
        #Equation MeqV=Peq
        Meq=sp.csc_matrix((NoCnstrVrt,LV))
        t=0
        for cv in PsCns:
            Meq[t,cv]=1.0
            t+=1
        #######################################################

        N1=sp.hstack([L,Meq.transpose()])
        N2=sp.hstack([Meq,sp.csc_matrix((NoCnstrVrt,NoCnstrVrt))])
        H=sp.vstack([N1,N2])
        print("ARAP Rigging time", time.time()-strttime)
        print("Saving Matrices ....")
        sp.save_npz(path+Rigname+'_H.npz', H.tocsc())
        sp.save_npz(path+Rigname+'_K.npz', K.tocsc())
                
        
        return{'FINISHED'}


def register():
    bpy.utils.register_class(ToolsPanel)
    bpy.utils.register_class(AssignSkeleton)
    bpy.types.Scene.RigNameLBS=bpy.props.StringProperty(name="RigName", description="", default="Default")
    bpy.utils.register_class(GetMeshSeq)
    bpy.utils.register_class(RigAnimation)
    bpy.utils.register_class(RigArap)
    
    
def unregister():
    bpy.utils.unregister_class(ToolsPanel)
    bpy.utils.unregister_class(AssignSkeleton)
    bpy.utils.unregister_class(GetMeshSeq)
    bpy.utils.unregister_class(RigAnimation)
    bpy.utils.unregister_class(RigArap)
    del bpy.types.Scene.RigNameLBS
    
if __name__ == "__main__":  
    bpy.utils.register_module(__name__) 


