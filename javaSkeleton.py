import os.path,subprocess
from subprocess import STDOUT,PIPE
import pandas
import numpy as np

def compile_java(java_file):
    subprocess.check_call(['javac', java_file])

def execute_java(java_file, mystdin):
    #java_class,ext = os.path.splitext(java_file)
    cmd = ['java', java_file] + mystdin
    proc = subprocess.Popen(cmd, stdin=PIPE,stdout=PIPE,stderr=STDOUT)
    stdout,stderr = proc.communicate()
    print(stdout)

def createSkeleton(imageStack):
    #We should have a stack of shape (x,y,z,1)
    #We open our java src file
    mydir=os.path.abspath('.')
    os.chdir("../../java/src/")
    #We save our files
    my_stack=str(imageStack.shape[1])+","+str(imageStack.shape[2])+"\n"
    for stack in imageStack:
        l=[e for s in stack for e in list(s)] #we reshape the y-z axis
        first=True
        for car in l:
            if not first:
                my_stack+=","
            first=False
            my_stack+=str(car)
        my_stack+="\n"
    fileWrite=open("..//data/forJavaSkeleton.txt","w")
    fileWrite.write(my_stack)
    fileWrite.close()

    compile_java("Skeletonize/ImageStack.java")
    compile_java("Skeletonize3d.java")
    print("compiled Java File")
    sub=subprocess.Popen(['java','Skeletonize3D','..//data/forJavaSkeleton.txt','..//data/outSkeleton.txt'])
    sub.wait()
    print("it looks as if the subprocess has finished")
    sub.wait()
    file=open("..//data/outSkeleton.txt")
    myline=file.read()
    docs=myline.split('\n')[:]
    array2D=[]
    for line in docs:
        array2D+=[line.split(',')]

    array=np.zeros(imageStack.shape)
    for x,line in enumerate(array2D[:-1]):
        z=0
        y=0
        for i in range(len(line)):
            array[x,y,z]=int(line[i])
            if(i%imageStack.shape[2]==imageStack.shape[2]-1):
                y+=1
                z=-1
            z+=1
    os.chdir(mydir)
    print("finished saving")
    return np.array(array,dtype=np.uint8)


