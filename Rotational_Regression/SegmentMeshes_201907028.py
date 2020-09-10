
bl_info = {
    "name": "Find Clusters",
    "author": "Prashant Domadiya",
    "version": (1, 1),
    "blender": (2, 82, 0),
    "location": "Python Console > Console > Run Script",
    "description": "Segment Meshes",
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
import time
import scipy.linalg as scla
from sklearn.cluster import KMeans




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


def GetNeighbours(G):
    Nbr=[[]]*len(G)
    for i in range(len(G)):
        g=G[i]
        for j in range(i+1,len(G)):
            gn=G[j]
            if len(set(g).intersection(set(gn)))!=0:
                Nbr[i]=Nbr[i]+[j]
                Nbr[j]=Nbr[j]+[i]
    return Nbr

def FindProxyBones(v,G):
    Pb=[]
    for i in range(len(G)):
        if v in G[i]:
            Pb.append(i)
    return Pb
            
##############################################################################################
#                      Global Variable
##############################################################################################
def LoadRigWeight(self, context):
    global FilePath,ActiveModel,path
    if len(RigList)!=0:
        ActiveModel=RigList[context.scene.ModelLBS]
        print('Active Model is....',ActiveModel)
        path=FilePath+ActiveModel+'/'

path=''
if os.path.isfile(FilePath+'Riglist.txt'):
    RigList=ReadStringList(FilePath+'Riglist.txt')
else:
    RigList=[]

if len(RigList)!=0:
    ActiveModel=RigList[0]
    print('Active Model is....',ActiveModel)
    path=FilePath+ActiveModel+'/'
else:
    print('Please Rigg the model first.....')

    
##############################################################################################
#                                   Tools
##############################################################################################
    
class ToolsPanel(bpy.types.Panel):
    bl_label = "Animation Tool"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
 
    def draw(self, context):
        self.layout.prop(context.scene,"ModelLBS")
        self.layout.label(text="Segment the mesh")
        self.layout.operator("get.seq",text='Mesh Seq').seqType="mesh"
        self.layout.prop(context.scene,"NumOfCls")
        self.layout.operator("get.seg",text='Segment').seqType="segment"
        self.layout.operator("clr.mesh",text='Color').seqType="ColorMesh"

class GetMeshSeq(bpy.types.Operator):
    bl_idname = "get.seq"
    bl_label = "Target Sequence"
    seqType : bpy.props.StringProperty()
 
    def execute(self, context):
        global FilePath,ActiveModel   
        Selected_Meshes=bpy.context.selected_objects
        obj = bpy.context.active_object
        
        
        V=np.zeros([3*len(Selected_Meshes),len(obj.data.vertices)])
        F=[]
        
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
            if i==0:
                for f in obj.data.polygons:
                    F.append(list(f.vertices))
              
        np.savetxt(path+ActiveModel+'_vertz.txt',V,delimiter=',')
        WriteAsTxt(path+ActiveModel+'_facz.txt',F)
        return{'FINISHED'}
    
class GetSegments(bpy.types.Operator):
    bl_idname = "get.seg"
    bl_label = "Segment Meshes"
    seqType : bpy.props.StringProperty()
 
    def execute(self, context):
        global path,ActiveModel
        Vrt=np.loadtxt(path+ActiveModel+'_vertz.txt',delimiter=',')
        Fcs=ReadTxt(path+ActiveModel+'_facz.txt')
        NF=len(Fcs)
        NPs,NV=np.shape(Vrt)
        NPs=NPs//3
        RM=np.zeros((NF,9*(NPs-1)))
        RefFrm=np.zeros((3,3))
        DefFrm=np.zeros((3,3))
        strttime=time.time()
        t=0
        for f in Fcs:
            for ps in range(NPs):
                if ps==0:
                    RefFrm[0:2]=Vrt[3*ps:3*ps+3,f[1:]].T-Vrt[3*ps:3*ps+3,f[0]]
                    RefFrm[2]=np.cross(RefFrm[0],RefFrm[1])/np.linalg.norm(np.cross(RefFrm[0],RefFrm[1]))
                else:
                    DefFrm[0:2]=Vrt[3*ps:3*ps+3,f[1:]].T-Vrt[3*ps:3*ps+3,f[0]]
                    DefFrm[2]=np.cross(DefFrm[0],DefFrm[1])/np.linalg.norm(np.cross(DefFrm[0],DefFrm[1]))
                    Q=np.dot(DefFrm.T,np.linalg.inv(RefFrm.T))
                    
                    R,S=scla.polar(Q)
                    RM[t,9*(ps-1):9*(ps-1)+9]=np.ravel(R)    
            t+=1
        
        Ncl=context.scene.NumOfCls
        print('Classifying....',Ncl,'...classes')
        clustering = KMeans(n_clusters=Ncl).fit(RM)
        Y=list(clustering.labels_)
        
        Cls=[[]]*Ncl
        t=0
        for i in list(Y):            
            Cls[i]=Cls[i]+[t]
            t+=1
        
        print("Segmentation time ...", time.time()-strttime)
        WriteAsTxt(path+ActiveModel+"_ClusterKmeans.txt",Cls)
        #np.savetxt(path+ActiveModel+"_KmeansInput.txt",RM,delimiter=',')
        print('Computing time=', time.time()-strttime)
        return{'FINISHED'}
class ColorMesh(bpy.types.Operator):
    bl_idname = "clr.mesh"
    bl_label = "Color Segmentations"
    seqType : bpy.props.StringProperty()
 
    def execute(self, context):
        global path,ActiveModel
        
        C=ReadTxt(path+ActiveModel+'_ClusterKmeans.txt')
        Ncl=(max(C[0])+1)
        Cls=[[]]*Ncl
        t=0
        for i in C[0]:
            Cls[i]=Cls[i]+[t]
            t+=1
        
        Selected_Meshes=bpy.context.selected_objects
        bpy.context.view_layer.objects.active = Selected_Meshes[0]
        obj = bpy.context.active_object
        mesh=bpy.context.object.data
        t=0
        for c in Cls:
            Mtrl=bpy.data.materials.new('Material'+str(t))
            Mtrl.diffuse_color=np.random.uniform(0,1,4).tolist()
            mesh.materials.append(Mtrl)
            for j in c:
                obj.data.polygons[j].material_index=t
            t+=1
        
        return{'FINISHED'}


def register():
    bpy.utils.register_class(ToolsPanel)
    bpy.types.Scene.ModelLBS=bpy.props.IntProperty(name="Model", description="Select Rigged Model", default=0,
                                                min=0,max=(len(RigList)-1),options={'ANIMATABLE'}, update=LoadRigWeight)
    bpy.types.Scene.NumOfCls=bpy.props.IntProperty(name="classes", description="Set Number of segments", default=1,
                                                  min=1,options={'ANIMATABLE'})
    bpy.utils.register_class(GetMeshSeq)
    bpy.utils.register_class(GetSegments)
    bpy.utils.register_class(ColorMesh)
    
    
def unregister():
    bpy.utils.unregister_class(ToolsPanel)
    bpy.utils.unregister_class(GetMeshSeq)
    bpy.utils.unregister_class(GetSegments)
    bpy.utils.unregister_class(ColorMesh)
    del bpy.types.Scene.ModelLBS
    del bpy.types.Scene.NumOfCls
    
if __name__ == "__main__":  
    bpy.utils.register_module(__name__) 


