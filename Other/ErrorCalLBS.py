
bl_info = {
    "name": "LBS Error Calculation",
    "author": "Prashant Domadiya",
    "version": (1, 3),
    "blender": (2, 71, 0),
    "location": "Python Console > Console > Run Script",
    "description": "Compute MSE for LBS",
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
    global ActiveModel,path,Phi,SrFrm,SFrm,CrntRot,Links
    if len(RigList)!=0:
        ActiveModel=RigList[context.scene.ModelLBS]
        path=Mainpath+ActiveModel+'/'
        print('Loading Model '+ActiveModel)
        Phi=np.loadtxt(path+ActiveModel+'_Phi.txt',delimiter=',')
        SrFrm=np.loadtxt(path+ActiveModel+'_RefFrames.txt',delimiter=',')
        SFrm=1*SrFrm
        CrntRot=np.array([[1.0,0,0]*(len(SrFrm.T)//4),[0,1.0,0]*(len(SrFrm.T)//4),[0,0,1]*(len(SrFrm.T)//4)])
        Links=ReadTxt(path+ActiveModel+'_SkelEdgz.txt')
        
Mainpath='/media/prashant/DATA/MyCodes/codeFiles/LBS/'

if os.path.isfile(Mainpath+'Riglist.txt'):
    RigList=ReadStringList(Mainpath+'Riglist.txt')
else:
    RigList=[]

if len(RigList)!=0:
    ActiveModel=RigList[0]
    path=Mainpath+ActiveModel+'/'
    Phi=np.loadtxt(path+RigList[0]+'_Phi.txt',delimiter=',')
    SrFrm=np.loadtxt(path+RigList[0]+'_RefFrames.txt',delimiter=',')
    CrntRot=np.array([[1.0,0,0]*(len(SrFrm.T)//4),[0,1.0,0]*(len(SrFrm.T)//4),[0,0,1]*(len(SrFrm.T)//4)])
    SFrm=1*SrFrm
    Links=ReadTxt(path+RigList[0]+'_SkelEdgz.txt')
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
    bl_label = "LBS Error calculation Tools"
    bl_space_type = "VIEW_3D"
    bl_region_type = "TOOLS"
 
    def draw(self, context):
        
        self.layout.prop(context.scene,"ModelLBS")

        self.layout.label("Animator")
        self.layout.operator("get.mesh",text='Mesh Seq').seqType="Mesh"
        self.layout.operator("animate.tool",text='Animate').seqType="animate"
        self.layout.operator("err.cmpt",text='Error').seqType="error"
        

    

class GetMeshSeq(bpy.types.Operator):
    bl_idname = "get.mesh"
    bl_label = "Load Skeleton"
    seqType = bpy.props.StringProperty()
 
    def execute(self, context):
        global path, ActiveModel,Links
        
        Selected_Meshes=bpy.context.selected_objects
        obj = bpy.context.active_object

        Joints=ReadTxt(path+'/'+ActiveModel+'_Joints.txt')
        
       
        NJ=len(Joints)
        NFrm=len(Links)
        NV=len(obj.data.vertices)
        NPs=len(Selected_Meshes)
        
        #SFrm=np.zeros([3*NPs,3*NFrm])
        SFrm=np.eye(4)
        Vrt=np.zeros((NV,3))
        RefVrtErr=np.zeros((3*NPs,NV))
        A=np.zeros((NJ,3))
        WC=np.array([[1.0,0,0,0],[0,1.0,0,0],[0,0,1.0,0]])
        RS=np.zeros((3*NPs,4*NFrm))
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

            #WriteObj(A,Links,i)    
            #CreateMesh(A,Links,1)
                
            for r in range(NFrm):
                
                V=(A[Links[r][0]]-A[Links[r][1]])/np.linalg.norm(A[Links[r][0]]-A[Links[r][1]])
                
                tmp=Vrt[Joints[Links[r][0]][0]]-A[Links[r][0]]
                B=np.cross(V,tmp)/np.linalg.norm(np.cross(V,tmp))
                N=np.cross(B,V)/np.linalg.norm(np.cross(B,V))
                
                SFrm[0:3,0]=V
                SFrm[0:3,1]=B
                SFrm[0:3,2]=N
                SFrm[0:3,3]=A[Links[r][0]]#A[r]
                RS[3*i:3*i+3,4*r:4*r+4]=SFrm[0:3]#WC.dot(np.linalg.inv(SFrm))    
        np.savetxt(path+'SkeletonFrame.txt',RS,delimiter=',')
        np.savetxt(path+'/'+'RefVrtErr.txt',RefVrtErr,delimiter=',')
        return{'FINISHED'}

class Animation(bpy.types.Operator):
    bl_idname = "animate.tool"
    bl_label = "Compute Animation"
    seqType = bpy.props.StringProperty()
 
    def execute(self, context):
        global Phi, path, ActiveModel
        
        SFrm=np.loadtxt(path+'SkeletonFrame.txt',delimiter=',')
        
        Fcs=ReadTxt(path+'/'+ActiveModel+'_facz.txt')
        NPs=len(SFrm)//3

        X=(SFrm).dot(Phi)
        np.savetxt(path+'/'+'EstVrtErr.txt',X,delimiter=',')
        
        
        #for i in range(NPs):
            #CreateMesh(X[3*i:3*i+3].T,Fcs,1)
        
        return{'FINISHED'}
class ErrCompute(bpy.types.Operator):
    bl_idname = "err.cmpt"
    bl_label = "Compute Animation"
    seqType = bpy.props.StringProperty()
 
    def execute(self, context):
        global path,ActiveModel,CnsVrt

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
        QuantValue=[np.mean(Err),np.std(Err),np.max(Err),np.max(np.std(VrtViceErr,axis=1))*1000,
                    np.argmax(Err),np.argmax(np.std(VrtViceErr,axis=1)),np.min(Err),np.min(np.std(VrtViceErr,axis=1))*1000,
                    np.argmin(Err),np.argmin(np.std(VrtViceErr,axis=1))]
        QuantName=['Mean','Standard deviation', 'Max Mean', 'Max Std','Max Mean Vert Id','Max Std Vert Id',
                   'Min Mean', 'Min Std','Min Mean Vert Id','Min Std Vert Id']
        WriteQuanMaric(path+ActiveModel+'_QuantitativeValues.txt',QuantValue,QuantName)
        MaxErr=np.max(VrtViceErr,axis=1)*1000
        np.savetxt(path+ActiveModel+'_Err.txt',VrtViceErr,delimiter=',')
        np.savetxt(path+ActiveModel+'_MaxErr.txt',MaxErr,delimiter=',')
        return{'FINISHED'}


def register():
    bpy.utils.register_class(ToolsPanel)
    bpy.types.Scene.ModelLBS=bpy.props.IntProperty(name="Model", description="Select Rigged Model", default=0,
                                                min=0,max=(len(RigList)-1),options={'ANIMATABLE'}, update=LoadRigWeight)
    
    
    bpy.utils.register_class(GetMeshSeq)
    bpy.utils.register_class(ErrCompute)
    
    bpy.utils.register_class(Animation)
    
def unregister():
    bpy.utils.unregister_class(ToolsPanel)
    
    bpy.utils.unregister_class(GetMeshSeq)
    bpy.utils.unregister_class(ErrCompute)
    
    bpy.utils.unregister_class(Animation)
    del bpy.types.Scene.ModelLBS
    
    
    
if __name__ == "__main__":  
    bpy.utils.register_module(__name__) 


