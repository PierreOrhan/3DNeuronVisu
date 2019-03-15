from GaussianFilter import *
from filterH import filterH
import os
import numpy as np
import itertools
import math
from tqdm import tqdm
from Converter import *
'''
    In this file we create a large number of tests case for our various scenarios!
    We will build differents test at different test scales
    
'''

#initialShape:(481, 3, 4865, 9683)

def listPossibleTile(scale):
    initialShape=[4865,9683,481]
    list=[]
    for i in range(0,int(initialShape[0]/scale[0])):
        for j in range(0,int(initialShape[1]/scale[1])):
            for h in range(0,int(initialShape[2]/scale[2])):
                list+=[{'x':[i*scale[0],(i+1)*scale[0]],'y':[j*scale[1],(j+1)*scale[1]],'z':[h*scale[2],(h+1)*scale[2]]}]
    return list

scales={#'1000':[1000,1000,300],
    '500':[500,500,170],
    '750':[750,750,250],
    '100':[100,100,30],
    '250':[250,250,80]}

scales={'100':[100,100,30]}

#os.chdir("costValidation")
gf=GaussianFilter("..//StackVecteur_185.tif")

for s in [list(scales.keys())[0]]:
    if s not in os.listdir(): os.makedirs(s)
    os.chdir(s)
    list=listPossibleTile(scales[s])
    print(s,len(list))
    nbrOfTest=1
    test=[]
    while(len(np.unique(test))<1):
        test=np.random.random_integers(0,len(list),nbrOfTest)
    print(len(test))
    for i in tqdm(range(0,len(test))):
        posDic=list[test[i]]
        posDic={'x': [2000, 2100], 'y': [3150, 3250], 'z': [20,50]}
        #posDic={'x':[2000,2050],'y':[3000,3050],'z':[20,35]}
        test[i]=150
        print("")
        print("creating GaussianStack")
        gaussStack=gf.gaussianFilter(posDic)
        print("created GausianStack")
        conv=MyFormatConverter(False)
        label=filterH(gaussStack,conv,name=str(test[i]),n_neighbors=100)
        
        print("ended  clustering")
    os.chdir('..')
# depth=[50,50,50]
# x=70
# y=75
# z=24
# posDic={'x': [2000, 2100], 'y': [3150, 3250], 'z': [20,50]}
# gaussStack=gf.gaussianFilter(posDic)
# conv=MyFormatConverter(False)
# conv.createSmallTiff(gaussStack,"2000_3000_450_10.tif")


