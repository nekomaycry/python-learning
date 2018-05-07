from numpy import *
import operator
from varibles import *

def createDataSet():
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels

def classify0(inX, dataSet, labels, k):
    m = dataSet.shape[0]
    diff=tile(inX, (m,1))-dataSet
    sqDiff = diff**2
    sqDistance = sqDiff.sum(axis=1)
    distance=sqDistance**0.5
    sortedDistIndicies=distance.argsort()#motherfucker
    classCount={};
    for i in range(0,k):
        label=labels[sortedDistIndicies[i]]
        classCount[label]=classCount.get(label,0)+1#motherfucker
    sortedClassCount=sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)#sorted  you motherfucker
    return sortedClassCount[0][0]
    
def file2matrix(filename):
    file=open(filename,'r')#second param
    re=[]
    classLables=[]
    for line in file.readlines():
        cols = line.strip().split('\t')
        re.append(cols[:-1])
        classLables.append(int(cols[-1]))
    m=len(re)
    zre=zeros((m,3))
    for i in range(0, m):
        zre[i]=re[i]
    return zre,classLables
        
def autoNorm(dataSet):
    minVal = dataSet.min(0)
    maxVal = dataSet.max(0)
    interval = maxVal-minVal
    m=dataSet.shape[0]
    returnSet=dataSet-tile(minVal,(m, 1))
    returnSet=returnSet/tile(interval,(m, 1))
    return returnSet,interval,minVal

def datingClassTest(filename):
    mRatio=0.1
    datas,labels=file2matrix(filename)
    size=datas.shape[0]
    normData,ranges,minVals=autoNorm(datas)
    m=int(mRatio*size)
    trainingSet=normData[m:]
    errorCount=0
    print(m)
    for i in range(0,m):
        data=normData[i]
        label=labels[i]
        classResult=classify0(data, trainingSet, labels[m:], 3)
        print("the classifier came back with: %d, the real answer is: %d, %s" % (classResult, label,data))
        if classResult!=label:
            errorCount+=1
    print('the error rate is {:.2%}'.format((errorCount/m)))
  
def img2vector(filename):
    file=open(filename,'r')#second param
    re=zeros((1024))
    for i in range(0,32):
        fr=file.readline().strip()
        for j in range(0, 32):
            re[32*i+j]=int(fr[j])
    return re

def handwritingClassTest():
    import os
    trainingFolder=os.listdir(inputPath+'2.KNN/trainingDigits')#mtfker! ,not '/' at end
    testingFolder=os.listdir(inputPath+'2.KNN/testDigits')
    trnsize=len(trainingFolder)
    labels=[]
    trndata=zeros((trnsize,1024))
    for i in range(trnsize):
        fileName=trainingFolder[i]#shit! it is a list of filenames ,damn it
        labels.append(int(fileName.split('_')[0]))
        trndata[i]=img2vector(inputPath+'2.KNN/trainingDigits/%s'%(fileName))
    tstsize=len(testingFolder)
    errorcount=0
    for i in range(tstsize):
        fileName=testingFolder[i]
        label=int(fileName.split('_')[0])
        data=img2vector(inputPath+'2.KNN/testDigits/%s'%(fileName))
        clsre=classify0(data, trndata, labels, 3)
        print('the classify result is %d the label is %d'%(clsre,label))
        errorcount+=clsre!=label
    print('the error rate is {:.2%}'.format((errorcount/tstsize)))

if __name__=='__main__':
    #datingClassTest(inputPath+'2.KNN/datingTestSet2.txt')
    handwritingClassTest()
    
    
    