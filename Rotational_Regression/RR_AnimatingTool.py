
bl_info = {
    "name": "Rotation Regration Animation Tool",
    "author": "Prashant Domadiya",
    "version": (1, 1),
    "blender": (2, 82, 0),
    "location": "Python Console > Console > Run Script",
    "description": "Animate Rigged Models using RR",
    "warning": "",
    "wiki_url": "",
    "tracker_url": "",
    "category": "Development"}

LibPath='/home/prashant/anaconda3/envs/Blender282/lib/python3.8/site-packages/'
Mainpath='/media/prashant/DATA/MyCodes/codeFiles/RR/'

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
        ob = bpy.data.objects.new('RR', me)
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
    global L,LBSPhi,K,Links,RefFrm,TwistLbsFrm,LbsSFrm,TwstAxsAngl,AxsAngl,CnsVrt,FrVrt
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
    NFrm=len(Links)
    #AxsAnglTmp=np.zeros((NFrm,3))
    for j in Links:
        if SelVrt == j:
            
            theta=(context.scene.TwistAngle1)*(np.pi/180)
            C=np.cos(theta)
            S=np.sin(theta)
            T=1-C
                
            #########  LBS  ######
            w=LbsSFrm[:,4*Id]
            R=np.array([[T*(w[0]**2)+C,T*w[0]*w[1]-S*w[2], T*w[0]*w[2]+S*w[1]],
                        [T*w[0]*w[1]+S*w[2],T*(w[1]**2)+C,T*w[1]*w[2]-S*w[0]],
                        [T*w[0]*w[2]-S*w[1],T*w[1]*w[2]+S*w[0],T*(w[2]**2)+C]])
            TwistLbsFrm[:,4*Id:4*Id+3]=R.dot(LbsSFrm[:,4*Id:4*Id+3])
            ######################
            w=RefFrm[:,4*Id]
            TwstAxsAngl[Id]=theta*w    
            Id+=1
            break

        Id+=1
    
    Ncl=len(JntInd)
    JntPrFace=len(JntInd[0])
    G=np.zeros((3,3*Ncl))
    Theta=np.ones(7)
    for c in range(Ncl):
        Ind=JntInd[c]
        R=0
        for i in range(JntPrFace):
            Y=W[3*c:3*c+3,3*i:3*i+3].dot(TwstAxsAngl[Ind[i]]+AxsAngl[Ind[i]])
            R=R+Wgts[c,i]*AnglAxisToRotMat(Y)
            Theta[3*i:3*i+3]=AxsAngl[Ind[i]]
        G[:,3*c:3*c+3]=R.dot(np.reshape(H[9*c:9*c+9].dot(Theta),(3,3)))

    X=np.zeros((3,len(CnsVrt)+len(FrVrt)))
    X[:,CnsVrt]=TwistLbsFrm.dot(LBSPhi)
    X[:,FrVrt]=G.dot(L)-X[:,CnsVrt].dot(K)
    
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
    global RefFrm,LbsSFrm,W,JntInd,Wgts,H,AxsAngl,TwstAxsAngl
    LbsSTmp=1*LbsSFrm
    bpy.data.objects['MySkelPose'].select_set(True)
    bpy.context.view_layer.objects.active = bpy.data.objects['MySkelPose']
    obj=bpy.context.active_object
    Vs=[]
    for vs in obj.data.vertices:
        co_final= obj.matrix_world @ vs.co
        Vs.append([co_final.x,co_final.z,-co_final.y])
    Vs=np.array(Vs)
    
    NFrm=len(LbsSFrm.T)//3
    t=0
    for e in obj.data.edges:
        ed=e.vertices[:]
        
        Vec=(Vs[ed[0]]-Vs[ed[1]])/np.linalg.norm(Vs[ed[0]]-Vs[ed[1]])
        #######  LBS  ############    
        w=np.cross(LbsSFrm[:,4*t],Vec)
        C=np.dot(Vec,LbsSFrm[:,4*t])
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
        LbsSTmp[:,4*t:4*t+3]=R.dot(LbsSFrm[:,4*t:4*t+3])
        LbsSTmp[:,4*t+3]=Vs[ed[0]]
        ##########################
        w=np.cross(RefFrm[:,4*t],Vec)
        if np.linalg.norm(w)!=0:
            w=w/np.linalg.norm(w)
            C=np.dot(Vec,RefFrm[:,4*t])
            if C>1.0:
                C=1.0
            elif C<-1.0:
                C=-1.0
            else:
                C=C
            Angl=np.arccos(C)
            AxsAngl[t]=Angl*w
        else:
            AxsAngl[t]=0
        
        t+=1

    
    Ncl=len(JntInd)
    JntPrFace=len(JntInd[0])
    G=np.zeros((3,3*Ncl))
    Theta=np.ones(7)
    for c in range(Ncl):
        Ind=JntInd[c]
        R=0
        for i in range(JntPrFace):
            Y=W[3*c:3*c+3,3*i:3*i+3].dot(TwstAxsAngl[Ind[i]]+AxsAngl[Ind[i]])
            R=R+Wgts[c,i]*AnglAxisToRotMat(Y)
            Theta[3*i:3*i+3]=AxsAngl[Ind[i]]
        G[:,3*c:3*c+3]=R.dot(np.reshape(H[9*c:9*c+9].dot(Theta),(3,3)))
    return G,LbsSTmp

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
    global ActiveModel,path,Lbspath, L,W,H,Wgts,JntInd,Links,RefFrm,AxsAngl,TwstAxsAngl
    global LbsSFrm,TwistLbsFrm,CnsVrt,FrVrt,K,LBSPhi
    if len(RigList)!=0:
        ActiveModel=RigList[context.scene.Model]
        print('Loading Model '+ActiveModel)
        path=Mainpath+ActiveModel+'/'
        Lbspath=MainLbspath+ActiveModel+'/'
        L=np.loadtxt(path+'/'+ActiveModel+'_L.txt',delimiter=',')
        
        W=np.loadtxt(path+'/'+ActiveModel+'_RRWght.txt',delimiter=',')
        H=np.loadtxt(path+'/'+ActiveModel+'_HRWght.txt',delimiter=',')
        Wgts=np.loadtxt(path+'/'+ActiveModel+'_JointClsWeights.txt',delimiter=',')
        JntInd=ReadTxt(path+ActiveModel+'_JointClsRl.txt')
        Links=ReadTxt(path+'/'+ActiveModel+'_SkelEdgz.txt')
        RefFrm=np.loadtxt(Lbspath+'/'+ActiveModel+'_RefFrames.txt',delimiter=',')
        
        AxsAngl=np.zeros((len(RefFrm.T)//3,3))
        TwstAxsAngl=np.zeros((len(RefFrm.T)//3,3))

        ######  LBS  #################
        LbsSFrm=1*RefFrm
        TwistLbsFrm=1*LbsSFrm
        CnsVrt=ReadTxt(path+ActiveModel+'_LBSVrt.txt')[0]
        FrVrt=ReadTxt(path+ActiveModel+'_FrVrt.txt')[0]
        K=np.loadtxt(path+'/'+ActiveModel+'_K.txt',delimiter=',')
        LBSPhi=np.loadtxt(path+'/'+ActiveModel+'_LBSPhi.txt',delimiter=',')
    return


MainLbspath='/media/prashant/DATA/MyCodes/codeFiles/LBS/'

if os.path.isfile(Mainpath+'/'+'Riglist.txt'):
    RigList=ReadStringList(Mainpath+'/'+'Riglist.txt')
else:
    RigList=[]

if len(RigList)!=0:
    ActiveModel=RigList[0]
    path=Mainpath+ActiveModel+'/'
    Lbspath=MainLbspath+ActiveModel+'/'
    L=np.loadtxt(path+'/'+RigList[0]+'_L.txt',delimiter=',')
    W=np.loadtxt(path+'/'+RigList[0]+'_RRWght.txt',delimiter=',')
    H=np.loadtxt(path+'/'+RigList[0]+'_HRWght.txt',delimiter=',')
    Wgts=np.loadtxt(path+'/'+RigList[0]+'_JointClsWeights.txt',delimiter=',')
    JntInd=ReadTxt(path+RigList[0]+'_JointClsRl.txt')
    Links=ReadTxt(path+'/'+RigList[0]+'_SkelEdgz.txt')
    RefFrm=np.loadtxt(Lbspath+'/'+RigList[0]+'_RefFrames.txt',delimiter=',')
    AxsAngl=np.zeros((len(RefFrm.T)//3,3))
    TwstAxsAngl=np.zeros((len(RefFrm.T)//3,3))
    
    ######  LBS ##################
    LbsSFrm=1*RefFrm
    TwistLbsFrm=1*LbsSFrm
    CnsVrt=ReadTxt(path+RigList[0]+'_LBSVrt.txt')[0]
    FrVrt=ReadTxt(path+RigList[0]+'_FrVrt.txt')[0]
    K=np.loadtxt(path+'/'+RigList[0]+'_K.txt',delimiter=',')
    LBSPhi=np.loadtxt(path+'/'+RigList[0]+'_LBSPhi.txt',delimiter=',')
else:
    print("First Rig the Models")
    
    
##############################################################################################
#                                   Tools
##############################################################################################
    
class ToolsPanel(bpy.types.Panel):
    bl_label = "Animation Tools Panel"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
 
    def draw(self, context):
        
        self.layout.prop(context.scene,"Model")

        self.layout.label(text="Poser")
        self.layout.operator("load.skel",text='Load Ref').seqType="loadref"
        self.layout.operator("pose.edtr",text='Edit Pose').seqType="editpose"
        self.layout.operator("pose.edtr",text='Edit twist').seqType="edittwist"
        self.layout.prop(context.scene,"TwistAngle1")
        
        self.layout.label(text="Animator")
        self.layout.operator("get.mesh",text='Mesh Seq').seqType="Mesh"
        self.layout.operator("animate.tool",text='Animate').seqType="animate"
        
class LoadReferenceSkel(bpy.types.Operator):
    bl_idname = "load.skel"
    bl_label = "Load Skeleton"
    seqType : bpy.props.StringProperty()
 
    def execute(self, context):
        global path, ActiveModel,Links
        
        #Links=ReadTxt(path+'/'+ActiveModel+'_SkelEdgz.txt')
        VS=np.loadtxt(path+'/'+ActiveModel+'_RefSkel.txt',delimiter=',')
        
        E=np.zeros(np.shape(VS))
        E[:,0]=VS[:,0]
        E[:,1]=-VS[:,2]
        E[:,2]=VS[:,1]
        me = bpy.data.meshes.new('MySkel')
        ob = bpy.data.objects.new('MySkelPose', me)
        scn = bpy.context.collection.objects.link(ob)
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
        scn = bpy.context.collection.objects.link(ob)
        me.from_pydata(E, [],Fcs)
        me.update()
        
        return{'FINISHED'}

class PoseEditor(bpy.types.Operator):
    bl_idname = "pose.edtr"
    bl_label = "Edit Pose"
    seqType : bpy.props.StringProperty()
 
    def execute(self, context):
        global L,LBSPhi,K,CnsVrt,FrVrt,LbsSFrm,TwistLbsFrm
        strttime=time.time()
        
        G,LbsSFrm=ComputeSkelFrame()
        if self.seqType=='editpose':
            TwistLbsFrm=1*LbsSFrm
        else:
            LbsSFrm=1*TwistLbsFrm
        X=np.zeros((3,len(CnsVrt)+len(FrVrt)))
        X[:,CnsVrt]=LbsSFrm.dot(LBSPhi)
        X[:,FrVrt]=G.dot(L)-X[:,CnsVrt].dot(K)
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

class GetMeshSeq(bpy.types.Operator):
    bl_idname = "get.mesh"
    bl_label = "Load Skeleton"
    seqType : bpy.props.StringProperty()
 
    def execute(self, context):
        global path,Lbspath, ActiveModel,Links,JntInd
        
        
        Selected_Meshes=bpy.context.selected_objects
        obj = bpy.context.active_object

        Joints=ReadTxt(path+'/'+ActiveModel+'_Joints.txt')
        RefSFrm=np.loadtxt(Lbspath+'/'+ActiveModel+'_RefFrames.txt',delimiter=',')
        VecIndx=[]
        for ln in Links:
            VecIndx.append(ln)
            
       
        NJ=len(Joints)
        NFrm=len(VecIndx)
        NV=len(obj.data.vertices)
        
        LbsSFrm=np.zeros([3,4*NFrm])
        Vrt=np.zeros((NV,3))
        A=np.zeros((NJ,3))
        AxsAngl=np.zeros((NFrm,3))
        
        bpy.context.view_layer.objects.active = Selected_Meshes[0]
        obj = bpy.context.active_object
             
        t=0
        for v in obj.data.vertices:
            co_final= obj.matrix_world @ v.co
            Vrt[t]=np.array([co_final.x,co_final.z,-co_final.y])
            t+=1
            
        for r in range(NJ):
            A[r]=np.mean(Vrt[Joints[r]],axis=0)

        for r in range(NFrm):
                
            V=(A[VecIndx[r][0]]-A[VecIndx[r][1]])/np.linalg.norm(A[VecIndx[r][0]]-A[VecIndx[r][1]])
            tmp=Vrt[Joints[VecIndx[r][0]][0]]-A[VecIndx[r][0]]
            B=np.cross(V,tmp)/np.linalg.norm(np.cross(V,tmp))
            N=np.cross(B,V)/np.linalg.norm(np.cross(B,V))
            LbsSFrm[:,4*r]=V
            LbsSFrm[:,4*r+1]=B
            LbsSFrm[:,4*r+2]=N
            LbsSFrm[:,4*r+3]=A[VecIndx[r][0]]
            R=LbsSFrm[:,4*r:4*r+3].dot(np.linalg.inv(RefSFrm[:,4*r:4*r+3]))
            Axs,Angl=RotMatToAnglAxis(R)
            AxsAngl[r]=Angl*Axs

        Ncl=len(JntInd)
        JntPrFace=len(JntInd[0])
        G=np.zeros((3,3*Ncl))
        Theta=np.ones(7)
        for c in range(Ncl):
            Ind=JntInd[c]
            R=0
            for i in range(JntPrFace):
                K=W[3*c:3*c+3,3*i:3*i+3].dot(AxsAngl[Ind[i]])
                R=R+Wgts[c,i]*AnglAxisToRotMat(K)
                Theta[3*i:3*i+3]=AxsAngl[Ind[i]]
            G[:,3*c:3*c+3]=R.dot(np.reshape(H[9*c:9*c+9].dot(Theta),(3,3)))
        
        np.savetxt(path+'/'+'G.txt',G,delimiter=',')
        np.savetxt(path+'/'+'LBS_SkeletonFrame.txt',LbsSFrm,delimiter=',')
        
        return{'FINISHED'}


class Animation(bpy.types.Operator):
    bl_idname = "animate.tool"
    bl_label = "Compute Animation"
    seqType : bpy.props.StringProperty()
 
    def execute(self, context):
        global L,LBSPhi,K, path, ActiveModel,CnsVrt,FrVrt
        
        G=np.loadtxt(path+'/'+'G.txt',delimiter=',')
        LbsSFrm=np.loadtxt(path+'/'+'LBS_SkeletonFrame.txt',delimiter=',')
        
        Fcs=ReadTxt(path+'/'+ActiveModel+'_facz.txt')
        NPs=len(LbsSFrm)//3

        strttime=time.time()
        print(len(CnsVrt))
        X=np.zeros((3*NPs,len(CnsVrt)+len(FrVrt)))
        X[:,CnsVrt]=LbsSFrm.dot(LBSPhi)
        X[:,FrVrt]=G.dot(L)-X[:,CnsVrt].dot(K)
        for i in range(NPs):
            CreateMesh(X[3*i:3*i+3].T,Fcs,1)
        
        return{'FINISHED'}


def register():
    bpy.utils.register_class(ToolsPanel)
    bpy.types.Scene.Model=bpy.props.IntProperty(name="Model", description="Select Rigged Model", default=0,
                                                min=0,max=(len(RigList)-1),options={'ANIMATABLE'}, update=LoadRigWeight)
    bpy.types.Scene.TwistAngle1=bpy.props.IntProperty(name="Angle1", description="Select Rigged Model", default=0,
                                                min=-180,max=180,options={'ANIMATABLE'}, update=Twist)
   
    bpy.utils.register_class(LoadReferenceSkel)
    bpy.utils.register_class(PoseEditor)
    bpy.utils.register_class(GetMeshSeq)
    bpy.utils.register_class(Animation)
    
def unregister():
    bpy.utils.unregister_class(ToolsPanel)
    bpy.utils.unregister_class(LoadReferenceSkel)
    bpy.utils.unregister_class(PoseEditor)
    bpy.utils.unregister_class(GetMeshSeq)
    bpy.utils.unregister_class(Animation)
   
    del bpy.types.Scene.Model
    del bpy.types.Scene.TwistAngle1
    
    
    
if __name__ == "__main__":  
    bpy.utils.register_module(__name__) 


