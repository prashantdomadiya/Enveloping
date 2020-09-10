
bl_info = {
    "name": "Rigging Tool",
    "author": "Prashant Domadiya",
    "version": (1, 3),
    "blender": (2, 82, 0),
    "location": "Python Console > Console > Run Script",
    "description": "Rig the Animation",
    "warning": "",
    "wiki_url": "",
    "tracker_url": "",
    "category": "Development"}

LibPath='/home/prashant/anaconda3/envs/Blender282/lib/python3.7/site-packages/'
FilePath='/media/prashant/DATA/MyCodes/codeFiles/OurSkinning/'


import sys
import os
from os.path import join
sys.path.append(LibPath)
if os.path.isdir(FilePath)==False:
    os.mkdir(FilePath)

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
        bpy.context.collection.objects.link(ob)
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
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    #               Face Transformation
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

def ConnectionMatrices(fcs,NV,NVec,CnsVrt,FrVrt):
    
    NCvrt=len(CnsVrt)
    AB=sp.lil_matrix((NV-NCvrt,NVec),dtype=float)
    ABc=sp.lil_matrix((NCvrt,NVec),dtype=float)
    tt=0
    for f in fcs:
        NVF=len(f)
        tmp=np.eye(NVF)-np.ones(NVF)/NVF
        j=0
        for i in f:
            if i in CnsVrt:
                ABc[CnsVrt.index(i),tt:tt+NVF]=tmp[j]
            else:
                AB[FrVrt.index(i),tt:tt+NVF]=tmp[j]
            j+=1
        tt+=NVF
    return AB,ABc

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


def ComuteReducedFaceList(F):
    bpy.ops.object.mode_set(mode="EDIT")
    obj = bpy.context.active_object.data
    m=bm.from_edit_mesh(obj)
    B=set()
    for vrt in m.verts:
        X=[]
        ASF=[]
        FV=set()
        Num_link_face=0
        for fc in vrt.link_faces:
            Num_link_face+=1
            if fc.index in B:
                ASF.append(fc.index)
                FV.update(F[fc.index])
            else:
                X.append(fc.index)
        if Num_link_face>2:
            for f in X:
                for t in it.combinations(F[f],2):
                    if len(set(list(t)).intersection(FV))==0:
                        ASF.append(f)
                        FV.update(F[f])      
                        break
            B.update(ASF)
        else:
            B.update(X)
                    
    bpy.ops.object.mode_set(mode ="OBJECT")
    return B

##############################################################################################
#                      Global Variable
##############################################################################################
path=''
if os.path.isfile(FilePath+'Riglist.txt'):
    RigList=ReadStringList(FilePath+'Riglist.txt')
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
        self.layout.prop(context.scene,"JointPrVec")
        self.layout.prop(context.scene,"FrmPrBone")
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

        WriteAsTxt(path+Rigname+'_facz.txt',F)
        ################# Reduce Number of Faces #################            
        FcList=ComuteReducedFaceList(F)
        
        
        ##########################################################
        
        V=np.zeros([3*len(Selected_Meshes),len(obj.data.vertices)])
        NVec=0
        for f in FcList:
            NVec+=len(F[f])  
        F=[]
        Nrml=np.zeros([3*NVec,len(Selected_Meshes)])
        
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
            k=0
            for f in obj.data.polygons:
                if k in FcList:
                    if i==0:
                        F.append(list(f.vertices))
                    NVF=len(list(f.vertices))
                    TmpN=obj.matrix_world.to_3x3() @ f.normal    
                    Nrml[t:t+3*NVF,i]=np.array([[1,0,0],[0,0,1],[0,-1,0]]*NVF).dot(TmpN)
                    t+=3*NVF
                k+=1

        #Vrt=Vrt[0:3,FcList[0]]      
        np.savetxt(path+Rigname+'_vertz.txt',V,delimiter=',')
        np.savetxt(path+Rigname+'_Normal.txt',np.array(Nrml),delimiter=',')
        WriteAsTxt(path+Rigname+'_Halffacz.txt',F)
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
        num_JntPrVec=context.scene.JointPrVec
        FrmPrBn=context.scene.FrmPrBone
        ################################################
        Vrt=np.loadtxt(path+Rigname+'_vertz.txt',delimiter=',')
        Fcs=ReadTxt(path+Rigname+'_Halffacz.txt')
        Nrml=np.loadtxt(path+Rigname+'_Normal.txt',delimiter=',')
        Joints=ReadTxt(path+Rigname+'_Joints.txt')
        Links=ReadTxt(path+Rigname+'_SkelEdgz.txt')
        
        NPs,NV=np.shape(Vrt)
        NPs=NPs//3
        NVec=sum([len(f) for f in Fcs])
        
        CnsVrt=ReadTxt(path+Rigname+'_LBSVrt.txt')[0]
        FrVrt=[]
        for i in range(NV):
            if i not in CnsVrt: 
                FrVrt.append(i)
        WriteAsTxt(path+Rigname+'_FrVrt.txt',FrVrt)
        np.savetxt(path+Rigname+'_RefMesh.txt',Vrt[0:3,:].T,delimiter=',')       
        ########################## Commpute Incident Matrices ###########################################
        print("Computing Incident Matrices.....")
        strtTime=time.time()
        AB,ABc=ConnectionMatrices(Fcs,NV,NVec,CnsVrt,FrVrt)
        Vecs=((AB.transpose()).dot(Vrt[:,FrVrt].T)).T+((ABc.transpose()).dot(Vrt[:,CnsVrt].T)).T
        factor=chmd.cholesky_AAt(AB.tocsc())
        
        ########################## Comute Skeleton Frames ########################################### 
        
        
        VecIndx=[]
        for ln in Links:
            if FrmPrBn==1:
                VecIndx.append(ln)
            else:
                VecIndx.append(ln)
                VecIndx.append([ln[1],ln[0]])
         
        print("Computing Skeleton Frames.....")
        NJ=len(Joints)
          
        
        NFrm=len(VecIndx)
        SFrm=np.zeros([3,3])
        SrFrm=np.zeros([3,4*NFrm])

        SFrmLbs=np.eye(4)
        SrFrmLbs=np.zeros([4,4*NFrm])
        RSLbs=np.zeros((3*(NPs-1),4*NFrm))
        
        A=np.zeros([NJ,3])
        RS=np.zeros((3*(NPs-1),3*NFrm))
        
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

                SFrm[:,0]=V
                SFrm[:,1]=B
                SFrm[:,2]=N
                SFrmLbs[0:3,0]=V
                SFrmLbs[0:3,1]=B
                SFrmLbs[0:3,2]=N
                SFrmLbs[0:3,3]=A[VecIndx[r][0]]
                if ps!=0:
                    RS[3*(ps-1):3*(ps-1)+3,3*r:3*r+3]=np.dot(SFrm,SrFrm[:,4*r:4*r+3].T)
                    RSLbs[3*(ps-1):3*(ps-1)+3,4*r:4*r+4]=SFrmLbs[0:3,:].dot(np.linalg.inv(SrFrmLbs[:,4*r:4*r+4]))
                else:
                    SrFrm[:,4*r:4*r+3]=SFrm
                    SrFrm[:,4*r+3]=A[VecIndx[r][0]]
                    SrFrmLbs[:,4*r:4*r+4]=SFrmLbs
        np.savetxt(path+Rigname+'_RefFrames.txt',SrFrm,delimiter=',')
        ############################## LBS on Constrain Vertices ##########################################

        print("Computing Mesh Frames for Constraint vertices .....")
        LBSPhi=np.zeros((4*NFrm,len(CnsVrt)))
        Vn=0
        for v in CnsVrt:
            v0=np.append(Vrt[0:3,v],[1])
            Err=[]
            H=np.zeros((3*(NPs-1),NFrm))
            for f in range(NFrm):
                for p in range(NPs-1):
                    H[3*p:3*p+3,f]=RSLbs[3*p:3*p+3,4*f:4*f+4].dot(v0)
                Err.append(np.linalg.norm(Vrt[3:,v]-H[:,f]))
    
            num_JntPrVrt=3
            IndSort=np.argsort(Err)
            SmlH=H[:,IndSort[0:num_JntPrVrt]]
            wghts=np.linalg.inv(np.dot(SmlH.T,SmlH)).dot(np.dot(SmlH.T,Vrt[3:,v]))
            #print(wghts)
            for i in range(num_JntPrVrt):
                if wghts[i]<0.0:
                    wghts[i]=0.0
            wghts=wghts/np.sum(wghts)
            
            for i in range(num_JntPrVrt):
                r=IndSort[i]
                Xr=SrFrmLbs[:,4*r:4*r+4]
                LBSPhi[4*r:4*r+4,Vn]=wghts[i]*(np.linalg.inv(Xr).dot(v0))
            Vn+=1

        del Vrt, Fcs,A,H
        
        ########################## Compute Mesh Frames ########################################### 
        print("Computing Mesh Frames.....")
        RM=np.zeros((3*(NPs-1),3))
        W=sp.lil_matrix((4*NFrm,NVec))
        Wtmp=np.zeros([3,3*NFrm])
        
        t=0
        MRFrm=np.zeros((3,3))
        MFrm=np.zeros((3,3))

        WghtMtrx=np.zeros((3*(NPs-1),num_JntPrVec))
        for vc in range(NVec):
            Err=[]
            MRFrm[:,2]=Nrml[3*vc:3*vc+3,0]    
            MRFrm[:,0]=Vecs[0:3,vc]
            MRFrm[:,1]=np.cross(MRFrm[:,0],MRFrm[:,2])
            MRFrm=MRFrm/np.linalg.norm(MRFrm,axis=0)
            
            
            for ps in range(1,NPs):
                 
                MFrm[:,2]=Nrml[3*vc:3*vc+3,ps]
                MFrm[:,0]=Vecs[3*ps:3*ps+3,vc]
                MFrm[:,1]=np.cross(MFrm[:,0],MFrm[:,2])
                MFrm=MFrm/np.linalg.norm(MFrm,axis=0)
                RM[3*(ps-1):3*(ps-1)+3]=MFrm.dot(MRFrm.T)
                
              
            for r in range(NFrm):
                U,Lmda,V=np.linalg.svd((RM.T).dot(RS[:,3*r:3*r+3]))
                Wtmp[:,3*r:3*r+3]=(V.T).dot(U.T)
                tmp=(RS[:,3*r:3*r+3].dot(Wtmp[:,3*r:3*r+3]))-RM
                Err.append(np.linalg.norm(tmp))
                
            IndSort=np.argsort(Err)
            TotalErr=sum([(1/Err[i]) for i in IndSort[0:num_JntPrVec]])
            
            for ps in range(NPs-1):
                for i in range(num_JntPrVec):
                    r=IndSort[i]
                    WghtMtrx[3*ps:3*ps+3,i]=(RS[3*ps:3*ps+3,3*r:3*r+3].dot(Wtmp[:,3*r:3*r+3])).dot(Vecs[0:3,vc])
            wghts=ComputeWeights(WghtMtrx,Vecs[3:,vc])        
            for i in range(num_JntPrVec):
                r=IndSort[i]             
                tmp=wghts[i]*((SrFrm[:,3*r:3*r+3].T).dot(Wtmp[:,3*r:3*r+3]))
                W[4*r:4*r+3,vc]=np.reshape(tmp.dot(Vecs[0:3,vc]),(3,1))
                
        print("Computing Matrix.....")
        
        del RM, RS, SrFrm, Vecs, Nrml
        
        ##### New
        Phi=(factor((AB.dot(W.transpose())).tocsc())).transpose()
        K=(factor((AB.dot(ABc.transpose())).tocsc())).transpose()
        print(time.time()-strtTime)
        del W
        tmp=Phi.toarray()-LBSPhi.dot(K.toarray())
        Phi=np.zeros((4*NFrm,NV))
        t=0
        for i in range(NV):
            if i in CnsVrt:
                Phi[:,i]=LBSPhi[:,t]
                t+=1
            else:
                Phi[:,i]=tmp[:,FrVrt.index(i)]
        np.savetxt(path+Rigname+'_PhiSingle.txt',Phi,delimiter=',')
        
        return{'FINISHED'}


def register():
    bpy.utils.register_class(ToolsPanel)
    bpy.utils.register_class(AssignSkeleton)
    bpy.types.Scene.RigName=bpy.props.StringProperty(name="RigName", description="", default="Default")
    bpy.types.Scene.JointPrVec=bpy.props.IntProperty(name="Joint Per Vector", description="Assign num of joint to vec", default=1,
                                                min=1,options={'ANIMATABLE'})
    bpy.types.Scene.FrmPrBone=bpy.props.IntProperty(name="Set Frames Per bone", description="Assign num of frames to bone", default=1,
                                                min=1,max=2,options={'ANIMATABLE'})
    bpy.utils.register_class(GetMeshSeq)
    bpy.utils.register_class(RigAnimation)
    
    
def unregister():
    bpy.utils.unregister_class(ToolsPanel)
    bpy.utils.unregister_class(AssignSkeleton)
    bpy.utils.unregister_class(GetMeshSeq)
    bpy.utils.unregister_class(RigAnimation)
    del bpy.types.Scene.RigName
    del bpy.types.Scene.JointPrVec
    del bpy.types.Scene.FrmPrBone
    
if __name__ == "__main__":  
    bpy.utils.register_module(__name__) 


