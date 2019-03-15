import sys
import os
import numpy as np
import skimage.color
#my classes
from Converter import *


def Huang2(data):
    #data: np.array s.t data[t]=number of point of intensity t (histogramm)
    first=-1
    for i in range(0,len(data)):
        if data[i]!=0:
            first=i
            break
    if first==-1:
        first=len(data)-1
    last=len(data)-1
    for i in np.arange(len(data)-1,first,-1):
        last=i
        if(data[i]!=0):
            break
    if(first==last):
        return 0
    
    S=[]
    W=[]
    for e in range(0,last+1):
        S+=[0]
        W+=[0]
    S[0]=data[0]
    for i in range(max(1,first),last+1):
        S[i]=S[i-1]+data[i]
        W[i]=W[i-1]+i*data[i]

    C=last-first
    Smu=np.zeros(C+1)
    mu=1./(1+np.array(range(1,C+1))/C)
    Smu[1:]=-mu*np.log(mu)-(1-mu)*np.log(1-mu)
    
    bestTresh=0
    bestEntropy=-1
    for t in range(first,last+1):
        entropy=0
        mu=int(W[t]/S[t])
        for i in range(first,t+1):
            entropy+=Smu[np.abs(i-mu)]*data[i]
        if(t==last or S[last]==S[t]):
            mu=1
        else:
            mu=int((W[last]-W[t])/(S[last]-S[t]))
        for i in range(t+1,last+1):
            try:
                entropy+=Smu[np.abs(i-mu)]*data[i]
            except:
                raise
        if(bestEntropy==-1 or bestEntropy>entropy):
            bestEntropy=entropy
            bestTresh=t
    return bestTresh

def build3DLHistogramm(stack):
    histogram=np.zeros(256)
    for s in stack:
        histogram[s[0]]+=1
    return histogram

def filter(stack,conv,isLUV=False):
    # conv is expected to be a MyFormatConverter instance
    # stack is expected to be a np.uint8 np array, stack is filtered
    '''This function gives back a booleanMask refering to which voxel should or should not be considered anymore

    '''
    assert(type(conv)==MyFormatConverter)
    if(not isLUV):
        luvStack=skimage.color.rgb2luv(stack)
    else:
        luvStack=stack
    print("building L histogram")
    data=build3DLHistogramm(np.array(luvStack[:,:,:,0],dtype=np.uint8))
    print("applying Huang2")
    bestTresh=Huang2(data)
    underTrsh=np.argwhere(luvStack[:,:,:,0]<bestTresh)
    booleanMask=np.zeros(stack.shape[:3],dtype=np.bool)
    booleanMask[underTrsh[:,0],underTrsh[:,1],underTrsh[:,2]]=1
    return 1-booleanMask


