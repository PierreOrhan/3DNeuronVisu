from scipy.spatial import cKDTree
import numpy as np
from tqdm import tqdm

def getSkeletonNeighbor(skeletonEmbedding,transformed_remaining,label,kneighbor=1):
    kneighbor=1 #We fix kneighbor at 1!
    mykdtree=cKDTree(skeletonEmbedding,leafsize=10000)
    newlabels=[]
    for i in tqdm(range(0,transformed_remaining.shape[0])):
        _,result=mykdtree.query(transformed_remaining[i],kneighbor)
        newlabels+=[label[result]]
    return newlabels
    
def softmax(dists,dist):
    '''
     Compute proba on the distance to each neighbors
    '''
    proba=np.exp(-1*dist)/np.sum(np.exp(-1*dists))
    return proba
    
def getFuzzySkeletonNeighborIn3DSpace(skeletonArray,LfilteredArray,label,kneighbor):
    '''
        We look for k neighbors voxel from skeletons and build probabilty measures on the assign label with a softmax measure of distances, the proba for each label is computed as the sum of proba for each neighbor having this label
    '''
    labelDic={}
    uniqueLabel=np.unique(label)
    for idx,lbl in enumerate(uniqueLabel):
        labelDic[lbl]=idx
    mykdtree=cKDTree(skeletonArray,leafsize=10000) #Change leafsize???
    newlabels=[]
    for i in tqdm(range(0,LfilteredArray.shape[0])):
        dists,results=mykdtree.query(LfilteredArray[i],kneighbor)
        probas=np.zeros(len(uniqueLabel))
        for idx,result in enumerate(results):
            probas[labelDic[label[result]]]+=softmax(dists,dists[idx])
        newlabels+=[np.argmax(probas)]
    return newlabels