
bl_info = {
    "name": "LBS Animation Tool",
    "author": "Prashant Domadiya",
    "version": (1, 1),
    "blender": (2, 82, 0),
    "location": "Python Console > Console > Run Script",
    "description": "Animate Rigged Models",
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
import bmesh
import numpy as np


from scipy import sparse as sp
from sksparse import cholmod as chmd
from scipy.sparse.linalg import inv
from scipy.sparse import linalg as sl
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
        ob = bpy.data.objects.new('LBS', me)
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
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    #               Face Transformation
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



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

def Twist(self, context):
    global SrFrm, Links,CrntRot,SFrm
    ########################## Selection ########################
    
    bpy.data.objects['MySkelPose'].select_set(True)
    bpy.context.view_layer.objects.active = bpy.data.objects['MySkelPose']
    obj=bpy.context.object
    bm=bmesh.from_edit_mesh(obj.data)
    SelVrt=[]
    for vs in bm.verts:
        if vs.select:
            SelVrt.append(vs.index)
    Id=0
    for j in Links:
        if SelVrt == j:
            
            theta=(context.scene.TwistAngleLBS1)*(np.pi/180)
        
            C=np.cos(theta)
            S=np.sin(theta)
            T=1-C
                
            w=SrFrm[:,4*Id]
            R=np.array([[T*(w[0]**2)+C,T*w[0]*w[1]-S*w[2], T*w[0]*w[2]+S*w[1]],
                        [T*w[0]*w[1]+S*w[2],T*(w[1]**2)+C,T*w[1]*w[2]-S*w[0]],
                        [T*w[0]*w[2]-S*w[1],T*w[1]*w[2]+S*w[0],T*(w[2]**2)+C]])
                
            SFrm[:,4*Id:4*Id+3]=CrntRot[:,3*Id:3*Id+3].dot(R.dot(SrFrm[:,4*Id:4*Id+3]))
            Id+=1
            break

        Id+=1
        
    X=(SFrm).dot(Phi)
    
    bpy.data.objects['MyMeshPose'].select_set(True)
    bpy.context.view_layer.objects.active = bpy.data.objects['MyMeshPose']
    obj = bpy.context.active_object
    j=0
    for v in obj.data.vertices:
        v.co=[X[0,j],-X[2,j],X[1,j]]
        j+=1
            
    bpy.data.objects['MySkelPose'].select_set(True)
    bpy.context.view_layer.objects.active = bpy.data.objects['MySkelPose']

    return

def ComputeSkelFrame():
    global SrFrm,CrntRot
    SFrm=np.zeros((3,len(SrFrm.T)))#np.zeros(np.shape(SrFrm))
    

    bpy.data.objects['MySkelPose'].select_set(True)
    bpy.context.view_layer.objects.active = bpy.data.objects['MySkelPose']
    obj=bpy.context.active_object
    Vs=[]
    for vs in obj.data.vertices:
        co_final= obj.matrix_world @ vs.co
        Vs.append([co_final.x,co_final.z,-co_final.y])
    Vs=np.array(Vs)
    
    t=0
    for e in obj.data.edges:
        ed=e.vertices[:]
        
        Vec=(Vs[ed[0]]-Vs[ed[1]])/np.linalg.norm(Vs[ed[0]]-Vs[ed[1]])
        w=np.cross(SrFrm[:,4*t],Vec)
        C=np.dot(Vec,SrFrm[:,4*t])
        if C>1.0:
            C=1.0
        elif C<-1.0:
            C=-1.0
        else:
            C=C
        if np.linalg.norm(w)!=0:
            w=w/np.linalg.norm(w)
            S=np.sin(np.arccos(C))
            T=1-C
            R=np.array([[T*(w[0]**2)+C,T*w[0]*w[1]-S*w[2], T*w[0]*w[2]+S*w[1]],
                        [T*w[0]*w[1]+S*w[2],T*(w[1]**2)+C,T*w[1]*w[2]-S*w[0]],
                        [T*w[0]*w[2]-S*w[1],T*w[1]*w[2]+S*w[0],T*(w[2]**2)+C]])
        else:
            R=np.eye(3)
        SFrm[:,4*t:4*t+3]=R.dot(SrFrm[:,4*t:4*t+3])
        SFrm[:,4*t+3]=Vs[ed[0]]  
        CrntRot[:,3*t:3*t+3]=R
        t+=1
    return SFrm

def WriteObj(V,F,PoseN):
    InputPath='/media/prashant/DATA/3D_DATA/SkinningData/'
    filepath=join(InputPath,'%05d.obj' % (PoseN))
    with open(filepath, 'w') as f:
        f.write("# OBJ file\n")
        for v in V:
            print(v)
            f.write("v %.4f %.4f %.4f\n" % (v[0],v[1],v[2]))
        for p in F:
            f.write("f")
            for i in p:
                f.write(" %d" % (i + 1))
            f.write("\n")
    return

def load_sparse_csc(filename):
    loader = np.load(filename)
    return sp.csc_matrix((loader['data'], loader['indices'], loader['indptr']),shape=loader['shape'])

##############################################################################################
#                      Global Variable
##############################################################################################
def LoadRigWeight(self, context):
    global ActiveModel,FilePath,path,Phi,SrFrm,SFrm,CrntRot,Links,H,K
    if len(RigList)!=0:
        ActiveModel=RigList[context.scene.ModelLBS]
        path=FilePath+ActiveModel+'/'
        print('Loading Model '+ActiveModel)
        Phi=np.loadtxt(path+ActiveModel+'_Phi.txt',delimiter=',')
        SrFrm=np.loadtxt(path+ActiveModel+'_RefFrames.txt',delimiter=',')
        SFrm=1*SrFrm
        CrntRot=np.array([[1.0,0,0]*(len(SrFrm.T)//4),[0,1.0,0]*(len(SrFrm.T)//4),[0,0,1]*(len(SrFrm.T)//4)])
        Links=ReadTxt(path+ActiveModel+'_SkelEdgz.txt')
        VrtArap=0
        H=load_sparse_csc(path+ActiveModel+'_H.npz')
        K=load_sparse_csc(path+ActiveModel+'_K.npz')
        Joints=ReadTxt(path+ActiveModel+'_Joints.txt')
        


if os.path.isfile(FilePath+'Riglist.txt'):
    RigList=ReadStringList(FilePath+'Riglist.txt')
else:
    RigList=[]

if len(RigList)!=0:
    ActiveModel=RigList[0]
    path=FilePath+ActiveModel+'/'
    Phi=np.loadtxt(path+RigList[0]+'_Phi.txt',delimiter=',')
    SrFrm=np.loadtxt(path+RigList[0]+'_RefFrames.txt',delimiter=',')
    CrntRot=np.array([[1.0,0,0]*(len(SrFrm.T)//4),[0,1.0,0]*(len(SrFrm.T)//4),[0,0,1]*(len(SrFrm.T)//4)])
    SFrm=1*SrFrm
    Links=ReadTxt(path+RigList[0]+'_SkelEdgz.txt')
    VrtArap=0
    
    H=sp.load_npz(path+ActiveModel+'_H.npz')
    K=sp.load_npz(path+ActiveModel+'_K.npz')
    Joints=ReadTxt(path+ActiveModel+'_Joints.txt')
else:
    print("First Rig the Models")
    ActiveModel=''
    Phi=np.zeros([1,1])
    SrFrm=np.zeros([1,1])
    SFrm=1*SrFrm
    CrntRot=1*SrFrm
    Links=[]
    
##############################################################################################
#                                   Tools
##############################################################################################
    
class ToolsPanel(bpy.types.Panel):
    bl_label = "LBS Tools Panel"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
 
    def draw(self, context):
        
        self.layout.prop(context.scene,"ModelLBS")

        self.layout.label(text="Poser")
        self.layout.operator("load.skel",text='Load Ref').seqType="loadref"
        self.layout.operator("pose.edtr",text='Edit Pose').seqType="editpose"
        self.layout.prop(context.scene,"TwistAngleLBS1")
        self.layout.operator("apply.arap",text='ARAP').seqType="applyarap"

        self.layout.label(text="Animator")
        self.layout.operator("get.skel",text='Skel Seq').seqType="skeleton"
        self.layout.operator("get.mesh",text='Mesh Seq').seqType="Mesh"
        self.layout.operator("animate.tool",text='Animate').seqType="animate"

        self.layout.label(text="DeepLBS")
        self.layout.operator("deeplbs.edtr",text='Deep Input').seqType="nllbs"
        self.layout.operator("deeplbs.edtr",text='DeepLBS').seqType="showdeeplbs"
        
class LoadReferenceSkel(bpy.types.Operator):
    bl_idname = "load.skel"
    bl_label = "Load Skeleton"
    seqType : bpy.props.StringProperty()
 
    def execute(self, context):
        global path, ActiveModel,Links
        
        VS=np.loadtxt(path+ActiveModel+'_RefSkel.txt',delimiter=',')
        
        E=np.zeros(np.shape(VS))
        E[:,0]=VS[:,0]
        E[:,1]=-VS[:,2]
        E[:,2]=VS[:,1]
        me = bpy.data.meshes.new('MySkel')
        ob = bpy.data.objects.new('MySkelPose', me)
        scn = bpy.context.collection.objects.link(ob)
        me.from_pydata(E, Links,[])
        me.update()


        Fcs=ReadTxt(path+ActiveModel+'_facz.txt')
        VM=np.loadtxt(path+ActiveModel+'_RefMesh.txt',delimiter=',')
        
        E=np.zeros(np.shape(VM))
        E[:,0]=VM[:,0]
        E[:,1]=-VM[:,2]
        E[:,2]=VM[:,1]
        me = bpy.data.meshes.new('MyMesh')
        ob = bpy.data.objects.new('MyMeshPose', me)
        scn = bpy.context.collection.objects.link(ob)
        me.from_pydata(E, [],Fcs)
        me.update()
        
        return{'FINISHED'}

class PoseEditor(bpy.types.Operator):
    bl_idname = "pose.edtr"
    bl_label = "Edit Pose"
    seqType = bpy.props.StringProperty()
 
    def execute(self, context):
        global Phi,SrFrm,SFrm,path, VrtArap
        strttime=time.time()
        SFrm=ComputeSkelFrame()
        
        X=(SFrm).dot(Phi)
        print(time.time()-strttime)
        VrtArap=1*X
        np.savetxt(path+ActiveModel+'_LBSVrts.txt',X,delimiter=',')
        bpy.data.objects['MyMeshPose'].select_set(True)
        bpy.context.view_layer.objects.active = bpy.data.objects['MyMeshPose']
        obj = bpy.context.active_object
        j=0
        for v in obj.data.vertices:
            v.co=[X[0,j],-X[2,j],X[1,j]]
            j+=1
            
        bpy.data.objects['MySkelPose'].select_set(True)
        bpy.context.view_layer.objects.active = bpy.data.objects['MySkelPose']
        
        return{'FINISHED'}
    
class ApplyARAP(bpy.types.Operator):
    bl_idname = "apply.arap"
    bl_label = "Apply ARAP"
    seqType : bpy.props.StringProperty()
 
    def execute(self, context):
        global VrtArap,H,K,Joints
        PsCns=[]
        for j in Joints:
            PsCns=PsCns+j
        X=ARAP(PsCns,VrtArap.T,H,K)
        
        bpy.data.objects['MyMeshPose'].select_set(True)
        bpy.context.view_layer.objects.active = bpy.data.objects['MyMeshPose']
        obj = bpy.context.active_object
        j=0
        for v in obj.data.vertices:
            v.co=[X[j,0],-X[j,2],X[j,1]]
            j+=1
            
        bpy.data.objects['MySkelPose'].select_set(True)
        bpy.context.view_layer.objects.active = bpy.data.objects['MySkelPose']
        
        return{'FINISHED'}

    
class DeepLBSEditor(bpy.types.Operator):
    bl_idname = "deeplbs.edtr"
    bl_label = "DeepLBS Pose Editor"
    seqType = bpy.props.StringProperty()
 
    def execute(self, context):
        global LBSVrt,ActiveModel,path
        if self.seqType=='nllbs':
            #bpy.data.objects['MyMeshPose'].select = True
            #bpy.context.scene.objects.active = bpy.data.objects['MyMeshPose']
            obj = bpy.context.active_object
            Vrt=[]
            for v in obj.data.vertices:
                co_final= obj.matrix_world @ v.co
                Vrt.append([co_final.x,co_final.z,-co_final.y])
            Vrt=np.array(Vrt)
            np.savetxt(path+ActiveModel+'_LBSVrts.txt',Vrt.T,delimiter=',')
            Joints=ReadTxt(path+ActiveModel+'_Joints.txt')
            Links=ReadTxt(path+ActiveModel+'_SkelEdgz.txt')
            NJ=len(Joints)

            VecIndx=[]
            for ln in Links:
                VecIndx.append(ln)
                
            NFrm=len(VecIndx)
            A=np.zeros((NJ,3))
            NlSFrm=np.zeros(np.shape(SFrm))
            for r in range(NJ):
                A[r]=np.mean(Vrt[Joints[r]],axis=0)
                
            for r in range(NFrm):
                
                V=(A[VecIndx[r][0]]-A[VecIndx[r][1]])/np.linalg.norm(A[VecIndx[r][0]]-A[VecIndx[r][1]])
                
                tmp=Vrt[Joints[VecIndx[r][0]][0]]-A[VecIndx[r][0]]
                B=np.cross(V,tmp)/np.linalg.norm(np.cross(V,tmp))
                N=np.cross(B,V)/np.linalg.norm(np.cross(B,V))

                NlSFrm[:,4*r]=V
                NlSFrm[:,4*r+1]=B
                NlSFrm[:,4*r+2]=N
                NlSFrm[:,4*r+3]=A[VecIndx[r][0]]
                
            np.savetxt(path+ActiveModel+'_NlSFrm.txt',NlSFrm,delimiter=',')
        
        else:
            X=np.loadtxt(path+ActiveModel+'_DeepLBSVrts.txt',delimiter=',')
            Fcs=ReadTxt(path+ActiveModel+'_facz.txt')
            E=np.zeros(np.shape(X))
            
            E[:,0]=X[:,0]
            E[:,1]=-X[:,2]
            E[:,2]=X[:,1]
            me = bpy.data.meshes.new('DeepLBS')
            ob = bpy.data.objects.new('DeepLBSPose', me)
            scn = bpy.context.collection.objects.link(ob)
            me.from_pydata(E, [],Fcs)
            me.update()
        
        return{'FINISHED'}
    
class GetMeshSeq(bpy.types.Operator):
    bl_idname = "get.mesh"
    bl_label = "Load Skeleton"
    seqType = bpy.props.StringProperty()
 
    def execute(self, context):
        global path, ActiveModel,Links
        
        Selected_Meshes=bpy.context.selected_objects
        obj = bpy.context.active_object

        Joints=ReadTxt(path+ActiveModel+'_Joints.txt')
        VecIndx=[]
        for ln in Links:
            VecIndx.append(ln)
       
        NJ=len(Joints)
        NFrm=len(VecIndx)
        NV=len(obj.data.vertices)
        NPs=len(Selected_Meshes)
        
        #SFrm=np.zeros([3*NPs,3*NFrm])
        SFrm=np.eye(4)
        Vrt=np.zeros((NV,3))
        A=np.zeros((NJ,3))
        WC=np.array([[1.0,0,0,0],[0,1.0,0,0],[0,0,1.0,0]])
        RS=np.zeros((3*NPs,4*NFrm))
        for i in range(NPs):
            bpy.context.view_layer.objects.active = Selected_Meshes[i]
            obj = bpy.context.active_object
             
            t=0
            for v in obj.data.vertices:
                co_final= obj.matrix_world @ v.co
                Vrt[t]=np.array([co_final.x,co_final.z,-co_final.y])
                t+=1

            for r in range(NJ):
                A[r]=np.mean(Vrt[Joints[r]],axis=0)

            #WriteObj(A,Links,i)    
            #CreateMesh(A,Links,1)
                
            for r in range(NFrm):
                
                V=(A[VecIndx[r][0]]-A[VecIndx[r][1]])/np.linalg.norm(A[VecIndx[r][0]]-A[VecIndx[r][1]])
                tmp=Vrt[Joints[VecIndx[r][0]][0]]-A[VecIndx[r][0]]
                B=np.cross(V,tmp)/np.linalg.norm(np.cross(V,tmp))
                N=np.cross(B,V)/np.linalg.norm(np.cross(B,V))
                
                SFrm[0:3,0]=V
                SFrm[0:3,1]=B
                SFrm[0:3,2]=N
                SFrm[0:3,3]=A[VecIndx[r][0]]#A[r]
                RS[3*i:3*i+3,4*r:4*r+4]=SFrm[0:3]#WC.dot(np.linalg.inv(SFrm))
        np.savetxt(path+'SkeletonFrame.txt',RS,delimiter=',')
        
        return{'FINISHED'}

class GetSkeletonSeq(bpy.types.Operator):
    bl_idname = "get.skel"
    bl_label = "Load Skeleton"
    seqType = bpy.props.StringProperty()
 
    def execute(self, context):
        global path, ActiveModel,SrFrm,Links
        
        Selected_Meshes=bpy.context.selected_objects
        obj = bpy.context.active_object

        NPs=len(Selected_Meshes)
        NFrm=len(SrFrm.T)//3
        
        SFrm=np.zeros([3*NPs,3*NFrm])
        
        for ps in range(NPs):
            bpy.context.view_layer.objects.active = Selected_Meshes[ps]
            obj = bpy.context.active_object

            Vs=[]
            for vs in obj.data.vertices:
                co_final= obj.matrix_world @ vs.co
                Vs.append([co_final.x,co_final.z,-co_final.y])
            Vs=np.array(Vs)
            t=0
            for ed in Links:
                Vec=(Vs[ed[0]]-Vs[ed[1]])/np.linalg.norm(Vs[ed[0]]-Vs[ed[1]])
                w=np.cross(SrFrm[:,3*t],Vec)
                C=np.dot(Vec,SrFrm[:,3*t])
                if C>1.0:
                    C=1.0
                elif C<-1.0:
                    C=-1.0
                else:
                    C=C
                if np.linalg.norm(w)!=0:
                    w=w/np.linalg.norm(w)
                    S=np.sin(np.arccos(C))
                    T=1-C
                    R=np.array([[T*(w[0]**2)+C,T*w[0]*w[1]-S*w[2], T*w[0]*w[2]+S*w[1]],
                                [T*w[0]*w[1]+S*w[2],T*(w[1]**2)+C,T*w[1]*w[2]-S*w[0]],
                                [T*w[0]*w[2]-S*w[1],T*w[1]*w[2]+S*w[0],T*(w[2]**2)+C]])
                else:
                    R=np.eye(3)
                SFrm[3*ps:3*ps+3,3*t:3*t+3]=R.dot(SrFrm[:,3*t:3*t+3])
                    
                t+=1    
        np.savetxt(path+'SkeletonFrame.txt',SFrm,delimiter=',')
        
        return{'FINISHED'}



class Animation(bpy.types.Operator):
    bl_idname = "animate.tool"
    bl_label = "Compute Animation"
    seqType = bpy.props.StringProperty()
 
    def execute(self, context):
        global Phi, path, ActiveModel
        
        SFrm=np.loadtxt(path+'SkeletonFrame.txt',delimiter=',')
        NPs=len(SFrm)//3
        Fcs=ReadTxt(path+ActiveModel+'_facz.txt')
        strttime=time.time()
        X=(SFrm).dot(Phi)
        print(time.time()-strttime)
        
        for i in range(NPs):
            CreateMesh(X[3*i:3*i+3].T,Fcs,1)
            
        
        return{'FINISHED'}


def register():
    bpy.utils.register_class(ToolsPanel)
    bpy.types.Scene.ModelLBS=bpy.props.IntProperty(name="Model", description="Select Rigged Model", default=0,
                                                min=0,max=(len(RigList)-1),options={'ANIMATABLE'}, update=LoadRigWeight)
    bpy.types.Scene.TwistAngleLBS1=bpy.props.IntProperty(name="Angle1", description="Select Rigged Model", default=0,
                                                min=-180,max=180,options={'ANIMATABLE'}, update=Twist)
    bpy.types.Scene.TwistAngleLBS2=bpy.props.IntProperty(name="Angle1", description="Select Rigged Model", default=0,
                                                min=-180,max=180,options={'ANIMATABLE'}, update=Twist)
    
    bpy.utils.register_class(LoadReferenceSkel)
    bpy.utils.register_class(PoseEditor)
    bpy.utils.register_class(ApplyARAP)
    bpy.utils.register_class(GetMeshSeq)
    bpy.utils.register_class(GetSkeletonSeq)
    bpy.utils.register_class(Animation)
    bpy.utils.register_class(DeepLBSEditor)
    
def unregister():
    bpy.utils.unregister_class(ToolsPanel)
    bpy.utils.unregister_class(LoadReferenceSkel)
    bpy.utils.unregister_class(PoseEditor)
    bpy.utils.unregister_class(ApplyARAP)
    bpy.utils.unregister_class(GetMeshSeq)
    bpy.utils.unregister_class(GetSkeletonSeq)
    bpy.utils.unregister_class(Animation)
    bpy.utils.unregister_class(DeepLBSEditor)
    del bpy.types.Scene.ModelLBS
    del bpy.types.Scene.TwistAngleLBS1
    del bpy.types.Scene.TwistAngleLBS2
    
    
if __name__ == "__main__":  
    bpy.utils.register_module(__name__) 


