import numba
import umap
import numpy as np
import time
import matplotlib.pyplot as plt
gamma=0.01
gammargb=0.5
from tqdm import tqdm

@numba.njit()
def hue(r, g, b):
    cmax = max(r, g, b)
    cmin = min(r, g, b)
    delta = cmax - cmin
    if delta == 0:
        return 0
    elif cmax == r:
        return ((g - b) / delta) % 6
    elif cmax == g:
        return ((b - r) / delta) + 2
    else:
        return ((r - g) / delta) + 4

@numba.njit()
def my_dist(a, b):
    ''' Define the distance between two vectors
    :param a: should be of shape (x,y,z,r,g,b)
    :return: the distance: gamma*euclid(a-b)+(1-gamma)*hueDist(a-b)
    '''
    diff = (hue(a[3], a[4], a[5]) - hue(b[3], b[4], b[5])) % 6
    if diff < 0:
        huediff = diff + 6
    else:
        huediff = diff
    euclidDist=(a[0]-b[0])**2+(a[1]-b[1])**2+(a[2]-b[2])**2
    return gamma*euclidDist+(1-gamma)*huediff

@numba.njit()
def my_dist2(a, b):
    rgbDist=(a[3]-b[3])**2+(a[4]-b[4])**2+(a[5]-b[5])**2
    euclidDist=(a[0]-b[0])**2+(a[1]-b[1])**2+(a[2]-b[2])**2
    return gammargb*euclidDist+(1-gammargb)*rgbDist


class myUmap:
    def __init__(self,mystack,n_neighbors=15,min_dist=0.1,n_components=2):
        '''
        The class cretaing the umap
        :param mystack: our RGB stack
        :param n_neighbors:
        :param min_dist:
        :param n_components:
        :return:
        '''
        assert(mystack.dtype==np.uint8)
        self.fit = umap.UMAP(
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            n_components=n_components,
            metric=my_dist
        )
        #Build in a unefficient way the coordinate matrix:
        newStack=[]
        for i in range(mystack.shape[0]):
            for j in range(mystack.shape[1]):
                for k in range(mystack.shape[2]):
                    if((mystack[i,j,k]!=0).all()):
                        newStack+=[list(np.concatenate(([i,j,k],mystack[i,j,k])))]
        #arrayStack=np.reshape(newStack,newshape=(newStack.shape[0]*newStack.shape[1]*newStack.shape[2],6))
        arrayStack=np.array(newStack)
        print("started fitting")
        t0=time.time()
        u=self.fit.fit_transform(arrayStack)
        print("finished fitting in ",time.time()-t0," for size",len(newStack))
        fig = plt.figure()
        if n_components == 1:
            ax = fig.add_subplot(111)
            ax.scatter(u[:,0], range(len(u)),c=arrayStack[:,3:]/255) #c=newStack[:,:,:,3:]
        if n_components == 2:
            ax = fig.add_subplot(111)
            ax.scatter(u[:,0], u[:,1],c=arrayStack[:,3:]/255) # c=newStack[:,:,:,3:]
        if n_components == 3:
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(u[:,0], u[:,1], u[:,2], c=arrayStack[:,3:]/255, s=100)
        plt.title("", fontsize=18)
        plt.show()
        self.transformedInit=u
    def transform(self,X):
        newStack=[]
        for i in tqdm(range(X.shape[0])):
            for j in range(X.shape[1]):
                for k in range(X.shape[2]):
                    if((X[i,j,k]!=0).all()):
                        newStack+=[list(np.concatenate(([i,j,k],X[i,j,k])))]
        arrayStack=np.array(newStack)
        print("starting transform of remaining stack, of size",len(newStack))
        u=self.fit.transform(arrayStack)
        self.transformed=u
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.scatter(u[:,0], u[:,1],c=arrayStack[:,3:]/255)
        plt.show()
        