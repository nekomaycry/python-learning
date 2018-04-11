from numpy import *
import operator

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
        classCount[label]=classCount.get(label,1)+1#motherfucker
    sortedClassCount=sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)#sorted  you motherfucker
    return sortedClassCount[0][0]
    
def file2matrix(filename):
    file=open(filename,'r')#second param
    m=len(file.readlines())
    re=zeros(m,3)
    index=0
    classLables=[]
    for line in file.readlines:
        cols = line.strip().split('\t')
        re[index]=cols[0:3]
        classLables.append(int(cols[-1]))
        index+=1
    return re
        
def autoNorm(dataSet):
    minVal = dataSet.min(0)
    maxVal = dataSet.max(0)
    interval = maxVal-minVal
    m=dataSet.shape[0]
    returnSet=dataSet-tile(minVal,(m, 1))
    returnSet=returnSet/tile(interval,(m, 1))
    return returnSet,interval,minVal

def datingClassTest():
    mRatio=0.1
    datas=file2matrix(filename);
    normData
    size=datas.shape[0]
    m=int(mRatio*size)
    errorCount=0
    for d in datas[0:m]:
        classify0(d, datas[m:], labels, k)
        
def img2vector(filename):
    pass

if __name__=='__main__':
    dataSet,labels=createDataSet()
    print(classify0([8,8], dataSet, labels, 3))

    
    
    