import numpy as np
from os.path import join
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

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

SlVrt=ReadTxt('/media/prashant/DATA/MyCodes/ColorError/Selected_vertz.txt')[0]


Rigname='50002'
Lbspath='/media/prashant/DATA/MyCodes/codeFiles/LBS/'+Rigname+'/'
VGpath='/media/prashant/DATA/MyCodes/codeFiles/OurSkinning/'+Rigname+'/'
VGpath_TwoBones='/media/prashant/DATA/MyCodes/codeFiles/OurSkinning_TwoBones/'+Rigname+'/'
DeepLbspath='/media/prashant/DATA/MyCodes/codeFiles/DeepLBS/'
ARAPLbspath='/media/prashant/DATA/MyCodes/codeFiles/LBSARAP/'+Rigname+'/'
RRpath='/media/prashant/DATA/MyCodes/codeFiles/RR/'+Rigname+'/'

#ERR=np.loadtxt(Lbspath+Rigname+'_Err.txt',delimiter=',')
#ERR=np.loadtxt(VGpath+Rigname+'_Err.txt',delimiter=',')
#ERR=np.loadtxt(VGpath_TwoBones+Rigname+'_Err.txt',delimiter=',')
#ERR=np.loadtxt(DeepLbspath+Rigname+'_DeepLBSErr.txt',delimiter=',')
#ERR=np.loadtxt(ARAPLbspath+Rigname+'_ARAPLBSErr.txt',delimiter=',')
ERR=np.loadtxt(RRpath+Rigname+'_Err.txt',delimiter=',')


ERR=ERR*1000
SlErr=ERR[SlVrt]
mu=np.mean(ERR)
sd=np.std(ERR)


N,bins,ptchs=plt.hist(ERR,100,color='b',stacked=True,normed=True,histtype='stepfilled',alpha=0.4)
N,bins,ptchs=plt.hist(SlErr,50,color='r',stacked=True,normed=True,histtype='stepfilled',alpha=0.4)
N,bins,ptchs=plt.hist(ERR,100,color='b',stacked=True,normed=True,histtype='step')
N,bins,ptchs=plt.hist(SlErr,50,color='r',stacked=True,normed=True,histtype='step')

"""
for i in range(len(ptchs)):
    if (bins[i]>=min(SlErr) and bins[i]<=max(SlErr)):
        ptchs[i].set_facecolor('r')
    else:
        ptchs[i].set_facecolor('b')
"""
plt.xlim(0,50)
#plt.ylim(0,800)
plt.ylim(0,0.25)
plt.show()
"""


N,bins,ptchs=plt.hist(SlErr,100,color='r',stacked=True,normed=True)

plt.xlim(0,50)
#plt.ylim(0,6)
plt.show()
"""
