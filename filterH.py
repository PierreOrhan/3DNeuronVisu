import sys
import os
import numpy as np
import skimage.color
#my classes
from Converter import *
from javaSkeleton import *
import filterL as filterer
import matplotlib.pyplot as plt
from umapReduction import myUmap
from nearestNeighbor import *
import hdbscan
import time

def getMask(smallStack,conv):
    booleanMask=filterer.filter(smallStack,conv)
    return booleanMask
def build3DHHistogramm(stack):
    histogram=np.zeros(256)
    for s in stack:
        histogram[s[0]]+=1
    totalH=sum(histogram)
    histogram=histogram/totalH
    return histogram
def HistogramSmoothing(histogram):
    pred=histogram[-1]
    alpha0=histogram[0]
    for i in range(0,len(histogram)-1):
        alpha=histogram[i]
        histogram[i]=(pred+histogram[i]+histogram[i+1])/3
        pred=alpha
    histogram[len(histogram)-1]=(pred+histogram[len(histogram)-1]+alpha0)/3
    return histogram

def filterH(smallStack,conv,isHSV=False,name="",tau=0.001,k=2,n_neighbors=15,min_dist=0.1,n_components=2):
    '''
        This function clusters voxel by applying multiple tresholding on the hue histogramm.
        We find the desired number of cluster by using a TDA approach (TOMATO algorithm, an extension to hill climbing), the tau function enable to filter between ignored and kept topological variation.
        K is the number of neighbor to which each hue value from the histogram compare itself. We recommend to use k=2.
    '''
    assert(smallStack.dtype==np.uint8)
    assert(type(conv)==MyFormatConverter)
    #We make sure we have the dir to save our results:
    dir=['histogram','result','clusters','diagram','clustersColor','skeleton','resultHDB']
    for d in dir:
        if d not in os.listdir():
            os.makedirs(d)
    #We start by masking with the Huang filter:
    booleanMask=getMask(smallStack,conv) #gives a 1D boolean Mask
    LfilteredArray=conv.from255toFloat(smallStack[booleanMask==True]) #
    booleanStackMask=np.stack((booleanMask,booleanMask,booleanMask),axis=-1)
    LfilteredStack=np.where(booleanStackMask,smallStack,np.zeros_like(smallStack))
    LfileredStackFloat=conv.from255toFloat(LfilteredStack)
    #After this mask is done we create our skeleton:
    booleanStackSum=np.sum(LfilteredStack,axis=3)
    booleanStack=np.array(np.array(booleanStackSum,dtype=np.bool),dtype=np.uint8)
    skeletonMask=createSkeleton(booleanStack)
    skeletonStackMask=np.stack((skeletonMask,skeletonMask,skeletonMask),axis=-1)
    #We use this skeleton to filter our stack
    skeletonStack=np.where(skeletonStackMask,LfilteredStack,np.zeros_like(LfilteredStack))
    skeletonArray=conv.from255toFloat(smallStack[skeletonMask==True])

    #UMAP algorithm:    
    print("starting umap on skeletonized")
    umap=myUmap(skeletonStack)
    #fitting the whole fitered stack on this umap
    #umap.transform(LfilteredStack)
    #newStack=umap.transformed
    print("starting transforming")
    #umap.transform(LfilteredStack)
    print("ended umap")
    print("starting hdbcsan")
    
    #HDBSCAN algorithm on the fitted stack:
    # sizeOfColored=np.where(LfilteredStack>0)[0].shape[0]
    # min_size=int(sizeOfColored/100)
    # cluster_labels=hdbFilter(newStack,min_size)
    # saveHDBfilter(conv,smallStack,booleanStackMask,cluster_labels,name="LstackFromSkeleton")
    #HDBSCAN on the skeleton  stack
    sizeOfColored=np.where(skeletonStack>0)[0].shape[0]
    min_size=int(sizeOfColored/100)
    cluster_labels2=hdbFilter(umap.transformedInit,min_size)
    saveHDBfilter(conv,smallStack,skeletonStackMask,cluster_labels2,name="skeleton")
    #Rather than re-doing HDBSCAN we can use nearest neighbor search
    # cluster_labels3=getSkeletonNeighbor(umap.transformedInit,umap.transformed,cluster_labels2,kneighbor=1)
    
    #we need to remove point of label -1:
    
    
    cluster_labels3=getFuzzySkeletonNeighborIn3DSpace(skeletonArray,LfilteredArray,cluster_labels2,15)
    # plotingHDB(umap.transformed,cluster_labels3)
    saveHDBfilter(conv,smallStack,booleanStackMask,cluster_labels3,name="nearestNeighbor")

    # umap2=myUmap(LfilteredStack)
    # sizeOfColored=np.where(LfilteredStack>0)[0].shape[0]
    # min_size=int(sizeOfColored/100)
    # cluster_labels=hdbFilter(umap2.transformedInit,min_size)
    # saveHDBfilter(conv,smallStack,booleanStackMask,cluster_labels,name="LstackFromUmap")
    
    
    print("creating skeleton tiff file")
    conv.createSmallTiff(LfilteredStack,"skeleton/initialStack")
    conv.createSmallTiff(skeletonStack,"skeleton/skeletonStack")
    print("ended tifile")    
    #stackList,label=Tomato(analyzedArray,smallStack)    
    return cluster_labels3

def plotingHDB(transformedInit,cluster_labels):
    min=np.min(cluster_labels)
    max=np.max(cluster_labels-min)
    fig=plt.figure()
    ax=fig.add_subplot(111)
    ax.scatter(transformedInit[:,0],transformedInit[:,1], c=(cluster_labels-min)/max, s=100)
    plt.show()

def hdbFilter(transformedInit,min_size):
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_size)
    cluster_labels = clusterer.fit_predict(transformedInit)
    plotingHDB(transformedInit,cluster_labels)
    return cluster_labels

def saveHDBfilter(conv,smallStack,skeletonStackMask,labels,name=""):
    label=np.unique(labels)
    myCarray=np.empty((len(label),3),dtype=np.float16)
    labelConv={}
    for l in range(0,len(label)):
        myCarray[l]=[np.random.randint(0,255)/255,np.random.randint(0,255)/255,np.random.randint(0,255)/255]
        labelConv[label[l]]=l
    clusteredStack=np.zeros(smallStack.shape)
    voxel=0
    for i in range(0,smallStack.shape[0]):
        for j in range(0,smallStack.shape[1]):
            for k in range(0,smallStack.shape[2]):
                if(skeletonStackMask[i,j,k,0]):
                    if(labels[voxel]!=-1):
                        clusteredStack[i,j,k]=myCarray[labelConv[labels[voxel]]]
                    voxel+=1
    conv.createSmallTiff(np.array(clusteredStack*255,dtype=np.uint8),"resultHDB/cluster"+name)
    conv.createSmallTiff(smallStack,"resultHDB/initial"+name)


def Tomato(analyzedArray,smallStack):
    #Which colors should be used within our skeleton? let us begin by only using the existing color
    hsvStack=skimage.color.rgb2hsv([skeletonArray])
    hArray=np.array(hsvStack[0,:]*255,dtype=np.uint8)
    print("computing hue histogram")
    data=build3DHHistogramm(hArray)
    plt.figure()
    plt.plot(range(256),data)
    plt.savefig("histogram/rawHistogram_"+name)
    plt.figure()
    data2=HistogramSmoothing(data)
    data2=HistogramSmoothing(data2)
    plt.plot(range(256),data2)
    plt.savefig("histogram/smoothedHistogram_"+name)
    #We now filter this hue histogram:
    print("applying Tomato")
    indexDic=np.argsort(data2)
    sortedDensity=data2[indexDic]
    parent=dict()
    diagramPoint=[]
    barCode=[]
    k=2
    tau=0.001
    #2nd steps
    for j in range(0,sortedDensity.shape[0]):
        i=sortedDensity.shape[0]-1-j
        if(sortedDensity[i]!=0):
            #create set of neighbor of higher density:
            neighborSet=[]
            argMax=-1
            dmax=0
            for t in np.arange(-k/2,k/2+1):
                if(t!=0):
                    neighbor=int(indexDic[i]+t)
                    if(neighbor>255):
                        neighbor=neighbor-256
                    if(neighbor<0):
                        neighbor=256+neighbor
                    d=data2[neighbor]
                    if(d>sortedDensity[i]):
                        neighborSet+=[neighbor]
                        if(d>dmax):
                            dmax=d
                            argMax=neighbor
            #merge:
            if(len(neighborSet)==0):
                parent[indexDic[i]]=indexDic[i]
            else:
                ei=parent[argMax]
                parent[indexDic[i]]=ei
                for neighbor in neighborSet:
                    e=parent[neighbor]
                    if(e!=ei):
                        roote=e
                        while(roote!=parent[roote]):
                            roote=parent[roote]
                        rootei=ei
                        while(rootei!=parent[rootei]):
                            rootei=parent[rootei]
                        diagramPoint+=[[np.min([data2[roote],data2[rootei]]),data2[indexDic[i]]]]
                        # if np.argmin([data2[roote],data2[rootei]])==1:
                        #     hueSpread=rootei
                        # else:
                        #     hueSpread=roote
                        # if((np.min([data2[roote],data2[rootei]])-data2[indexDic[i]])*(1+np.log(np.abs(hueSpread-indexDic[i])))<tau):
                        if((np.min([data2[roote],data2[rootei]])-data2[indexDic[i]])<tau):
                            if(data2[roote]>data2[rootei]):
                                parent[rootei]=roote
                                ei=e
                            else:
                                parent[roote]=rootei
        else:
            parent[indexDic[i]]=-1
    print("saving cluster")
    diagramPoint+=[[0,0]]
    labelDic=np.empty(len(list(parent.keys())))
    for p in parent.keys():
        if(parent[p]!=-1):
            papa=parent[p]
            while(parent[papa]!=papa):
                papa=parent[papa]
            labelDic[p]=papa
        else:
            labelDic[p]=labelDic[p-1]
    label=np.unique(labelDic)
    print(label)
    fig=plt.figure()
    plt.scatter(np.array(diagramPoint)[:,0],np.array(diagramPoint)[:,1])
    plt.savefig("diagram/"+name)

    myCarray=np.empty((len(label),3),dtype=np.float16)
    labelConv={}
    for l in range(0,len(label)):
        myCarray[l]=[np.random.randint(0,255)/255,np.random.randint(0,255)/255,np.random.randint(0,255)/255]
        labelConv[label[l]]=l
    plt.figure()
    for i in range(0,256):
        plt.scatter(i,data2[i],c=myCarray[labelConv[labelDic[i]]])
    plt.savefig("histogram/clusteredHisto_"+name)

    clusteredStack=np.empty(smallStack.shape)
    bigHSVStack=np.empty(smallStack.shape)
    for i in range(0,bigHSVStack.shape[0]):
        bigHSVStack[i,:]=skimage.color.rgb2hsv(smallStack[i])
    for i in range(0,bigHSVStack.shape[0]):
        for j in range(0,bigHSVStack.shape[1]):
            for k in range(0,bigHSVStack.shape[2]):
                if(booleanMask[i,j,k]):
                    h=int(bigHSVStack[i,j,k,0]*255)
                    clusteredStack[i,j,k]=myCarray[labelConv[labelDic[h]]]
    conv.createSmallTiff(np.array(clusteredStack*255,dtype=np.uint8),"result/cluster_"+name)
    conv.createSmallTiff(smallStack,"result/initial_"+name)

    stackList=[]
    for lbl in label:
        clusteredStack=np.zeros(smallStack.shape,dtype=np.uint8)
        for i in range(0,bigHSVStack.shape[0]):
            for j in range(0,bigHSVStack.shape[1]):
                for k in range(0,bigHSVStack.shape[2]):
                    if(booleanMask[i,j,k]):
                        h=int(bigHSVStack[i,j,k,0]*255)
                        if lbl==labelDic[h]:
                            clusteredStack[i,j,k]=smallStack[i,j,k]
        conv.createSmallTiff(clusteredStack,"clustersColor/"+str(int(lbl))+"_"+name)
        stackList.append(clusteredStack)
    return stackList,label