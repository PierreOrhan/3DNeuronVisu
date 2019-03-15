import numpy as np
import cv2
import tiffile
import sys
class MyFormatConverter:
    def __init__(self,to255=True):
        self.to255=to255

    def displayFromFloatColorArray(self,imageIn,minColor=np.array([0.,0.,0.]),maxColor=np.array([2000.,2000.,2000.])):
        #this function find the range of intersted colors (beetween min and max for each channel) and set to min all inferior
        # and to max all superior pixel.
        image=np.copy(imageIn)
        for i in range(0,image.shape[0]):
            for e in range(0,3):
                image[i,np.argwhere((image[i,:,:]>maxColor)[:,e]),e]=maxColor[e]
                image[i,np.argwhere((image[i,:,:]<minColor)[:,e]),e]=minColor[e]
        if(not((maxColor==minColor).all())):
            beta=minColor/(maxColor-minColor)
            alpha=1./(maxColor-minColor)
            for e in range(0,3):
                image[:,:,e]=image[:,:,e]*alpha[e]-beta[e]
        else:
            for e in range(0,3):
                image[:,:,e]=0
        if self.to255:
            image=image*255
        return image

    def convert(self,image):
        maxTbl,minTbl=self.getMaxMinRGB(image)
        image=self.displayFromFloatColorArray(image,np.array(minTbl),np.array(maxTbl))
        return image

    def convertRawStack(self,StackImageIni):
        #create a stack of converted images
        StackImage=np.empty(StackImageIni.shape)
        for i in range(0,StackImageIni.shape[2]):
            #we stack our images in a big file, changing the axis and converting to a known range of work
            StackImage[:,:,i]=self.convert(StackImageIni[:,:,i])
        if(self.to255):
            return np.array(StackImage,dtype=np.uint8)
        else:
            return np.array(StackImage)

    def getMaxMinRGB(self,img):
        maxTbl=[]
        minTbl=[]
        for i in range(0,img.shape[2]):
            minChannel,maxChannel,_,_=cv2.minMaxLoc(img[:,:,i])
            maxTbl+=[maxChannel]
            minTbl+=[minChannel]
        return maxTbl,minTbl

    def changeChannelValue(self,image,minChannel,maxChannel):
        for i in range(0,image.shape[0]):
            image[i,np.argwhere((image[i,:]>maxChannel))]=maxChannel
            image[i,np.argwhere((image[i,:]<minChannel))]=minChannel
        if(not(maxChannel==minChannel)):
            beta=minChannel/(maxChannel-minChannel)
            alpha=1./(maxChannel-minChannel)
            image=image*alpha-beta
        else:
            image=image*0
        if self.to255:
            image=image*255
        return image

    def convertOneChannel(self,image):
        minChannel,maxChannel,_,_=cv2.minMaxLoc(image)
        image=self.changeChannelValue(image,minChannel,maxChannel)
        return image


    def convertRawStackOneChannel(self,StackImageIni):
        StackImage=np.empty(StackImageIni.shape)
        for i in range(0,StackImageIni.shape[2]):
            #we stack our images in a big file, changing the axis and converting to a known range of work
            StackImage[:,:,i]=self.convertOneChannel(StackImageIni[:,:,i])
        if(self.to255):
            return np.array(StackImage,dtype=np.uint8)
        else:
            return np.array(StackImage)

    def renormaliser(self,imageFinal):
        #make sure our highest color is at the maximum of its channel intensity.
        maxTbl=[]
        for i in range(0,3):
            _,maxChannel,_,_=cv2.minMaxLoc(imageFinal[:,:,i])
            maxTbl+=[maxChannel]
        mymax=max(maxTbl)
        if mymax==0:
            return imageFinal
        if self.to255:
            return imageFinal*int(255/mymax)
        return imageFinal*(1/mymax)

    def createSmallTiff(self,StackImageIni,name="test/PartialVecteur"):
        writer=tiffile.TiffWriter(name+".tif")
        writer.save(StackImageIni)
        writer.close()
        # for e in range(0,3):
        #     writer=tiffile.TiffWriter(sys.path[0]+name+str(e)+".tif")
        #     writer.save(StackImageIni[:,:,:,e])
        #     writer.close()

    def from255toFloat(self,stack):
        newStack=stack.astype(dtype=np.float)
        return newStack/255

