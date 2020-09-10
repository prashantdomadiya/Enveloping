
bl_info = {
    "name": "Error Calculation",
    "author": "Prashant Domadiya",
    "version": (1, 3),
    "blender": (2, 81, 0),
    "location": "Python Console > Console > Run Script",
    "description": "Compute MSE",
    "warning": "",
    "wiki_url": "",
    "tracker_url": "",
    "category": "Development"}

import sys
sys.path.append('/home/student/anaconda3/envs/Blender281/lib/python3.7/site-packages/')

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
        bpy.context.collection.objects.link(ob)
        me.from_pydata(E[:,3*i:3*i+3], [], F)
        me.update()
        


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
"""
def LoadRigWeight(self, context):
    global ActiveModel,path,Lbspath, Phi,TwistFrm, Links,FrmPrBn
    global SFrm,K,LBSPhi,CnsVrt,FrVrt,Id
    if len(RigList)!=0:
        ActiveModel=RigList[context.scene.ModelErr]
        path=Mainpath+ActiveModel+'/'
        Lbspath=MainLbspath+ActiveModel+'/'
        print('Loading Model '+ActiveModel)
        Phi=np.loadtxt(path+ActiveModel+'_PhiSingle.txt',delimiter=',')
        Links=ReadTxt(path+ActiveModel+'_SkelEdgz.txt')
        ############## LBS
        SFrm=np.loadtxt(Lbspath+ActiveModel+'_RefFrames.txt',delimiter=',')
        K=np.loadtxt(path+ActiveModel+'_LBSK.txt',delimiter=',')
        LBSPhi=np.loadtxt(path+ActiveModel+'_LBSPhi.txt',delimiter=',')
        CnsVrt=ReadTxt(path+ActiveModel+'_LBSVrt.txt')[0]
        FrVrt=ReadTxt(path+ActiveModel+'_FrVrt.txt')[0]
        Id=[i for i in range(len(SFrm[0])) if ((i+1)%4)!=0]
        TwistFrm=1*SFrm
        FrmPrBn=len(SFrm[0])//(4*len(Links))
        ##############
    return

Mainpath='/media/prashant/DATA/MyCodes/codeFiles/OurSkinning/'
MainLbspath='/media/prashant/DATA/MyCodes/codeFiles/LBS/'

if os.path.isfile(Mainpath+'Riglist.txt'):
    RigList=ReadStringList(Mainpath+'Riglist.txt')
else:
    RigList=[]

if len(RigList)!=0:
    ActiveModel=RigList[0]
    path=Mainpath+ActiveModel+'/'
    Lbspath=MainLbspath+ActiveModel+'/'
    Phi=np.loadtxt(path+RigList[0]+'_Phi.txt',delimiter=',')
    Links=ReadTxt(path+RigList[0]+'_SkelEdgz.txt')
    ############## LBS
    SFrm=np.loadtxt(path+RigList[0]+'_RefFrames.txt',delimiter=',')
    K=np.loadtxt(path+RigList[0]+'_LBSK.txt',delimiter=',')
    LBSPhi=np.loadtxt(path+RigList[0]+'_LBSPhi.txt',delimiter=',')
    CnsVrt=ReadTxt(path+RigList[0]+'_LBSVrt.txt')[0]
    FrVrt=ReadTxt(path+RigList[0]+'_FrVrt.txt')[0]
    Id=[i for i in range(len(SFrm[0])) if ((i+1)%4)!=0]
    TwistFrm=1*SFrm
    FrmPrBn=len(SFrm[0])//(4*len(Links))
    #############
else:
    print("First Rig the Models")
    
"""
def LoadRigWeight(self, context):
    global ActiveModel,path,Lbspath, Phi,TwistFrm, Links,FrmPrBn
    global SFrm 
    if len(RigList)!=0:
        ActiveModel=RigList[context.scene.ModelErr]
        path=Mainpath+ActiveModel+'/'
        Lbspath=MainLbspath+ActiveModel+'/'
        print('Loading Model '+ActiveModel)
        Phi=np.loadtxt(path+ActiveModel+'_PhiSingle.txt',delimiter=',')
        Links=ReadTxt(path+ActiveModel+'_SkelEdgz.txt')
        SFrm=np.loadtxt(path+ActiveModel+'_RefFrames.txt',delimiter=',')
        TwistFrm=1*SFrm
        FrmPrBn=len(SFrm[0])//(4*len(Links))
        
    return

Mainpath='/home/student/Documents/Skinning/OurSkinning/'
MainLbspath='/home/student/Documents/Skinning/LBS/'

if os.path.isfile(Mainpath+'Riglist.txt'):
    RigList=ReadStringList(Mainpath+'Riglist.txt')
else:
    RigList=[]

if len(RigList)!=0:
    ActiveModel=RigList[0]
    path=Mainpath+ActiveModel+'/'
    Lbspath=MainLbspath+ActiveModel+'/'
    Phi=np.loadtxt(path+RigList[0]+'_PhiSingle.txt',delimiter=',')
    Links=ReadTxt(path+RigList[0]+'_SkelEdgz.txt')
    SFrm=np.loadtxt(path+RigList[0]+'_RefFrames.txt',delimiter=',')
    TwistFrm=1*SFrm
    FrmPrBn=len(SFrm[0])//(4*len(Links))
    #############
else:
    print("First Rig the Models")
    
    
##############################################################################################
#                                   Tools
##############################################################################################
    
class ToolsPanel(bpy.types.Panel):
    bl_label = "Error calculation Tools"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
 
    def draw(self, context):
        
        self.layout.prop(context.scene,"ModelErr")

        self.layout.label(text="Animator")
        self.layout.operator("get.mesh",text='Mesh Seq').seqType="Mesh"
        self.layout.operator("animate.tool",text='Animate').seqType="animate"
        self.layout.operator("err.cmpt",text='Error').seqType="error"
        

class GetMeshSeq(bpy.types.Operator):
    bl_idname = "get.mesh"
    bl_label = "Load Skeleton"
    seqType : bpy.props.StringProperty()
 
    def execute(self, context):
        global path, ActiveModel,Links,FrmPrBn
        
        Selected_Meshes=bpy.context.selected_objects
        obj = bpy.context.active_object

        Joints=ReadTxt(path+'/'+ActiveModel+'_Joints.txt')
        #Links=ReadTxt(path+'/'+ActiveModel+'_SkelEdgz.txt')
        VecIndx=[]
        for ln in Links:
            if FrmPrBn==1:
                VecIndx.append(ln)
            else:
                VecIndx.append(ln)
                VecIndx.append([ln[1],ln[0]])
       
        NJ=len(Joints)
        NFrm=len(VecIndx)
        NV=len(obj.data.vertices)
        NPs=len(Selected_Meshes)
        
        SFrm=np.zeros([3*NPs,4*NFrm])
        Vrt=np.zeros((NV,3))
        RefVrtErr=np.zeros((3*NPs,NV))
        A=np.zeros((NJ,3))
        for i in range(NPs):
            bpy.context.scene.objects.active = Selected_Meshes[-i-1]
            obj = bpy.context.active_object
             
            t=0
            for v in obj.data.vertices:
                co_final= obj.matrix_world*v.co
                Vrt[t]=np.array([co_final.x,co_final.z,-co_final.y])
                t+=1
            RefVrtErr[3*i:3*i+3]=Vrt.T
            for r in range(NJ):
                A[r]=np.mean(Vrt[Joints[r]],axis=0)

            for r in range(NFrm):
                
                V=(A[VecIndx[r][0]]-A[VecIndx[r][1]])/np.linalg.norm(A[VecIndx[r][0]]-A[VecIndx[r][1]])
                tmp=Vrt[Joints[VecIndx[r][0]][0]]-A[VecIndx[r][0]]
                B=np.cross(V,tmp)/np.linalg.norm(np.cross(V,tmp))
                N=np.cross(B,V)/np.linalg.norm(np.cross(B,V))
                SFrm[3*i:3*i+3,4*r]=V
                SFrm[3*i:3*i+3,4*r+1]=B
                SFrm[3*i:3*i+3,4*r+2]=N
                SFrm[3*i:3*i+3,4*r+3]=A[VecIndx[r][0]]
        
        np.savetxt(path+'RefVrtErr.txt',RefVrtErr,delimiter=',')
        np.savetxt(path+'SkeletonFrame.txt',SFrm,delimiter=',')
        
        return{'FINISHED'}

class Animation(bpy.types.Operator):
    bl_idname = "animate.tool"
    bl_label = "Compute Animation"
    seqType:bpy.props.StringProperty()
 
    def execute(self, context):
        global Phi, path, ActiveModel
        
        SFrm=np.loadtxt(path+'SkeletonFrame.txt',delimiter=',')
        
        Fcs=ReadTxt(path+ActiveModel+'_facz.txt')
        NPs=len(SFrm)//3
    
        strttime=time.time()
        X=SFrm[0:3].dot(Phi)
        for i in range(1,NPs):
            tmp=SFrm[3*i:3*i+3].dot(Phi)
            X=np.append(X,tmp,axis=0)
            
        np.savetxt(path+'/'+'EstVrtErr.txt',X,delimiter=',')
        return{'FINISHED'}

class ErrCompute(bpy.types.Operator):
    bl_idname = "err.cmpt"
    bl_label = "Compute Animation"
    seqType:bpy.props.StringProperty()
 
    def execute(self, context):
        global path,ActiveModel

        Fcs=ReadTxt(path+'/'+ActiveModel+'_facz.txt')
        
        RefVrtErr=np.loadtxt(path+'/'+'RefVrtErr.txt',delimiter=',')
        EstVrtErr=np.loadtxt(path+'/'+'EstVrtErr.txt',delimiter=',')
        
        NPs,NVrt=np.shape(RefVrtErr)
        NPs=NPs//3
        VrtViceErr=np.zeros((NVrt,NPs))
        for i in range(NPs):
            Ref=RefVrtErr[3*i:3*i+3].T
            Est=EstVrtErr[3*i:3*i+3].T
            VrtViceErr[:,i]=np.linalg.norm(Ref-Est,axis=1)
        Err=np.mean(VrtViceErr,axis=1)*1000
        print('Mean=',np.mean(Err))
        print('Standard deviation=',np.std(Err))
        print('Max mean Error', np.max(Err))
        print('Max std Error', np.max(np.std(VrtViceErr,axis=1)))
        
        MaxErr=np.max(VrtViceErr,axis=1)*1000
        np.savetxt(path+ActiveModel+'_Err.txt',Err,delimiter=',')
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


