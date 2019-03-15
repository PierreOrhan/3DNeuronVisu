from Converter import *

def compareIndex(i1,i2):
    equal=True
    for e in range(0,len(i2)):
        if(i1[e]!=i2[e]):
            equal=False
    return equal

def easyIndex(a):
    return (a[0],a[1],a[2])

def easyIndex2(a,i):
    return (a[0],a[1],a[2],i)

def getVoxelNeighbor(coord,k,limit):
    if(k!=6):
        raise Exception("k should be 6 for neighbor of 3D voxel")
    neighbor=[]
    for e in range(0,3):
        if(coord[e]!=0):
            b=[coord[0],coord[1],coord[2]]
            b[e]=b[e]-1
            neighbor+=[b]
        if(coord[e]!=limit[e]):
            b=[coord[0],coord[1],coord[2]]
            b[e]=b[e]+1
            neighbor+=[b]
    return neighbor

def computeKNeighborhood(gridDim):
    neighborDict=dict()
    for i in range(0,gridDim[0]):
        for j in range(0,gridDim[1]):
            for k in range(0,gridDim[2]):
                neighborDict[i,j,k]=getVoxelNeighbor([i,j,k],6,[gridDim[0]-1,gridDim[1]-1,gridDim[2]-1])
    return neighborDict

def generateMaskedTiff(booleanMask,smallStack,name):
    conv=MyFormatConverter()
    displayStack=np.empty(smallStack.shape,dtype=np.uint8)
    for i in range(0,smallStack.shape[0]):
        for j in range(0,smallStack.shape[1]):
            for k in range(0,smallStack.shape[2]):
                if(booleanMask[i,j,k]):
                    displayStack[i,j,k]=smallStack[i,j,k]
                else:
                    displayStack[i,j,k]=[0,0,0]
    conv.createSmallTiff(displayStack,name)