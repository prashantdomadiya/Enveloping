
bl_info = {
    "name": "Animation Tool",
    "author": "Prashant Domadiya",
    "version": (1, 0),
    "blender": (2, 81, 0),
    "location": "Python Console > Console > Run Script",
    "description": "Animate Rigged Models",
    "warning": "",
    "wiki_url": "",
    "tracker_url": "",
    "category": "Development"}

LibPath='/home/prashant/anaconda3/envs/mypy39/lib/python3.9/site-packages/'
#FilePath='/media/prashant/DATA/MyCodes/codeFiles/OurSkinning/'

import sys
import os
from os.path import join
sys.path.append(LibPath)



import bpy
import bmesh
import numpy as np
from scipy import sparse as sp
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
        ob = bpy.data.objects.new('VG', me)
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



def Twist(self, context):
    global TwistFrm,Links,SFrm,FrmPrBn,CnsVrt,FrVrt,Id
    ########################## Selection ########################
    bpy.data.objects['MySkelPose'].select_set(True)
    bpy.context.view_layer.objects.active = bpy.data.objects['MySkelPose']
    obj=bpy.context.object
    bm=bmesh.from_edit_mesh(obj.data)
    SelVrt=[]
    for vs in bm.verts:
        if vs.select:
            SelVrt.append(vs.index)
    
    t=0
    for j in Links:
        if SelVrt == j:
            
            theta=[(context.scene.TwistAngle1)*(np.pi/180)]
            if FrmPrBn>1:
                theta.append((context.scene.TwistAngle2)*(np.pi/180))
        
            for i in range(FrmPrBn):
                C=np.cos(theta[i])
                S=np.sin(theta[i])
                T=1-C
                
                w=SFrm[:,4*(t+i)]
                R=np.array([[T*(w[0]**2)+C,T*w[0]*w[1]-S*w[2], T*w[0]*w[2]+S*w[1]],
                            [T*w[0]*w[1]+S*w[2],T*(w[1]**2)+C,T*w[1]*w[2]-S*w[0]],
                            [T*w[0]*w[2]-S*w[1],T*w[1]*w[2]+S*w[0],T*(w[2]**2)+C]])
                TwistFrm[:,4*(t+i):4*(t+i)+3]=R.dot(SFrm[:,4*(t+i):4*(t+i)+3])
                
            break

        t+=FrmPrBn
    
    X=TwistFrm.dot(Phi)
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
    global FrmPrBn,SFrm   
    STmp=1*SFrm
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
        
        for i in range(FrmPrBn):
            if i==0:
                Vec=(Vs[ed[0]]-Vs[ed[1]])/np.linalg.norm(Vs[ed[0]]-Vs[ed[1]])
            else:
                Vec=(Vs[ed[1]]-Vs[ed[0]])/np.linalg.norm(Vs[ed[1]]-Vs[ed[0]])
            w=np.cross(SFrm[:,4*(t+i)],Vec)
            C=np.dot(Vec,SFrm[:,4*(t+i)])
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
            STmp[:,4*(t+i):4*(t+i)+3]=R.dot(SFrm[:,4*(t+i):4*(t+i)+3])
            STmp[:,4*(t+i)+3]=Vs[ed[0]]
        t+=FrmPrBn    
    return STmp

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
    global FilePath,path,ActiveModel, Phi,TwistFrm, Links,FrmPrBn,SFrm
    FilePath=context.scene.FilePath
    RigList=ReadStringList(FilePath+"/Riglist.txt")
    if len(RigList)!=0:
        ActiveModel=RigList[context.scene.Model]
        path=FilePath+"/"+ActiveModel+"/"
        print('Loading Model '+ActiveModel)
        Phi=np.loadtxt(path+ActiveModel+'_PhiSingle.txt',delimiter=',')
        Links=ReadTxt(path+ActiveModel+'_SkelEdgz.txt')
        SFrm=np.loadtxt(path+ActiveModel+'_RefFrames.txt',delimiter=',')
        TwistFrm=1*SFrm
        FrmPrBn=len(SFrm[0])//(4*len(Links))
    else:
        print("First Rig the Models")
        
    return




    
    
##############################################################################################
#                                   Tools
##############################################################################################
    
class ToolsPanel(bpy.types.Panel):
    bl_label = "VG Animation Tools Panel"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
 
    def draw(self, context):
        
        self.layout.prop(context.scene,"Model")
        self.layout.prop(context.scene,"FilePath")

        self.layout.label(text="Poser")
        self.layout.operator("load.skel",text='Load Ref').seqType="loadref"
        self.layout.operator("pose.edtr",text='Edit Pose').seqType="editpose"
        self.layout.operator("pose.edtr",text='Edit twist').seqType="edittwist"
        self.layout.prop(context.scene,"TwistAngle1")
        

        
        
class LoadReferenceSkel(bpy.types.Operator):
    bl_idname = "load.skel"
    bl_label = "Load Skeleton"
    seqType : bpy.props.StringProperty()
 
    def execute(self, context):
        global path, ActiveModel,Links
        
        VS=np.loadtxt(path+'/'+ActiveModel+'_RefSkel.txt',delimiter=',')
        
        E=np.zeros(np.shape(VS))
        E[:,0]=VS[:,0]
        E[:,1]=-VS[:,2]
        E[:,2]=VS[:,1]
        me = bpy.data.meshes.new('MySkel')
        ob = bpy.data.objects.new('MySkelPose', me)
        bpy.context.collection.objects.link(ob)
        me.from_pydata(E, Links,[])
        me.update()


        Fcs=ReadTxt(path+'/'+ActiveModel+'_facz.txt')
        VM=np.loadtxt(path+'/'+ActiveModel+'_RefMesh.txt',delimiter=',')
        
        E=np.zeros(np.shape(VM))
        E[:,0]=VM[:,0]
        E[:,1]=-VM[:,2]
        E[:,2]=VM[:,1]
        me = bpy.data.meshes.new('MyMesh')
        ob = bpy.data.objects.new('MyMeshPose', me)
        bpy.context.collection.objects.link(ob)
        me.from_pydata(E, [],Fcs)
        me.update()
        
        return{'FINISHED'}

class PoseEditor(bpy.types.Operator):
    bl_idname = "pose.edtr"
    bl_label = "Edit Pose"
    seqType : bpy.props.StringProperty()
 
    def execute(self, context):
        global Phi,SFrm,TwistFrm
        strttime=time.time()
        if self.seqType=='editpose':
            SFrm=ComputeSkelFrame()
            TwistFrm=1*SFrm
        else:
            SFrm=1*TwistFrm
        
        X=SFrm.dot(Phi)
        print(time.time()-strttime)
    
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
    


def register():
    bpy.utils.register_class(ToolsPanel)
    bpy.types.Scene.Model=bpy.props.IntProperty(name="Model", description="Select Rigged Model", default=0,
                                                min=0,max=100,options={'ANIMATABLE'}, update=LoadRigWeight)
    bpy.types.Scene.TwistAngle1=bpy.props.IntProperty(name="Angle1", description="Select Rigged Model", default=0,
                                                min=-180,max=180,options={'ANIMATABLE'}, update=Twist)
    bpy.types.Scene.FilePath=bpy.props.StringProperty(name="FilePath", description="", default="Default")
    bpy.utils.register_class(LoadReferenceSkel)
    bpy.utils.register_class(PoseEditor)
    
    
def unregister():
    bpy.utils.unregister_class(ToolsPanel)
    bpy.utils.unregister_class(LoadReferenceSkel)
    bpy.utils.unregister_class(PoseEditor)
    
    
    del bpy.types.Scene.Model
    del bpy.types.Scene.TwistAngle1
    del bpy.types.Scene.FilePath
    
    
if __name__ == "__main__":  
    bpy.utils.register_module(__name__) 


