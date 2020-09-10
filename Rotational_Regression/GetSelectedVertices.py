
bl_info = {
    "name": "Get Indices of Selected Vertices",
    "author": "Prashant Domadiya",
    "version": (1, 1),
    "blender": (2, 82, 0),
    "location": "Python Console > Console > Run Script",
    "description": "Rig the Animation",
    "warning": "",
    "wiki_url": "",
    "tracker_url": "",
    "category": "Development"}

LibPath='/home/prashant/anaconda3/envs/Blender282/lib/python3.7/site-packages/'
FilePath='/media/prashant/DATA/MyCodes/codeFiles/RR/'

import sys
import os
from os.path import join
sys.path.append(LibPath)

import bpy
import bmesh as bm
import numpy as np



def WriteAsTxt(Name,Vec):
    with open(Name, 'w') as fl:
        for i in Vec:
            if str(type(i))=="<class 'list'>":
                for j in i:
                    fl.write(" %d" % j)
                fl.write("\n")
            else:
                fl.write(" %d" % i)


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
RRMainpath='/media/prashant/DATA/MyCodes/codeFiles/RR/'
def LoadRigWeight(self, context):
    global FilePath, ActiveModel,path,RRpath
    if len(RigList)!=0:
        ActiveModel=RigList[context.scene.Model]
        print('Active model=',ActiveModel)
        path=FilePath+ActiveModel+'/'
        RRpath=RRMainpath+ActiveModel+'/'

if os.path.isfile(FilePath+'Riglist.txt'):
    RigList=ReadStringList(FilePath+'Riglist.txt')
else:
    RigList=[]

if len(RigList)!=0:
    ActiveModel=RigList[0]
    path=FilePath+ActiveModel+'/'
    RRpath=RRMainpath+ActiveModel+'/'

    
##############################################################################################
#                                   Tools
##############################################################################################
    
class ToolsPanel(bpy.types.Panel):
    bl_label = "Animation Tool"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
 
    def draw(self, context):
        
        self.layout.prop(context.scene,"Model")
        self.layout.operator("get.selvrt",text='Get Vert').seqType="get"



class GetSelVrt(bpy.types.Operator):
    bl_idname = "get.selvrt"
    bl_label = "Target Sequence"
    seqType : bpy.props.StringProperty()
 
    def execute(self, context):
        global path,RRpath,ActiveModel
         
        Selected_Meshes=bpy.context.selected_objects
        obj = bpy.context.active_object
        SelVrt = [i.index for i in bpy.context.active_object.data.vertices if i.select]
        WriteAsTxt(path+ActiveModel+'_LBSVrt.txt',SelVrt)
        WriteAsTxt(RRpath+ActiveModel+'_LBSVrt.txt',SelVrt)
        return{'FINISHED'}


def register():
    bpy.utils.register_class(ToolsPanel)
    bpy.utils.register_class(GetSelVrt)
    bpy.types.Scene.Model=bpy.props.IntProperty(name="Model", description="Select Rigged Model", default=0,
                                                min=0,max=(len(RigList)-1),options={'ANIMATABLE'}, update=LoadRigWeight)
    
    
def unregister():
    bpy.utils.unregister_class(ToolsPanel)
    bpy.utils.unregister_class(GetSelVrt)
    del bpy.types.Scene.Model
    
if __name__ == "__main__":  
    bpy.utils.register_module(__name__) 


