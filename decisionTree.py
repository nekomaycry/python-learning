from numpy import *
from math import log
import operator

def createDataSet():
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no surfacing', 'flippers']
    return dataSet,labels

def calcShannonEnt(dataSet):
    clses=[ele[-1] for ele in dataSet]
    m=len(clses)
    clscount={}
    for i in range(m):
        clscount[clses[i]]=clscount.get(clses[i],0)+1
    entropy=0;
    for k in clscount:
        p=clscount[k]/m
        entropy-=log(p,2)*p
    return entropy
        
def splitDataSet(dataSet, index, value):
    re=[]
    m=len(dataSet)
    for i in range(m):
        if dataSet[i][index]==value:
            featured=dataSet[i][:index]
            featured.extend(dataSet[i][index+1:])
            re.append(featured)
    return re

def chooseBestFeatureToSplit(dataSet):
    m,n=shape(dataSet)
    minEntr=0
    index=0
    for i in range(n-1):
        vals=[ele[i] for ele in dataSet]
        unqVals=set(vals)
        entsum=0;
        for val in unqVals:
            spld=splitDataSet(dataSet, i, val)
            entsum+=calcShannonEnt(spld)
        if i==0:
            minEntr=entsum
        elif minEntr>entsum:
            minEntr=entsum
            index=i
    return index,minEntr

def majorityCnt(classList):
    clsdic={}
    for ele in classList:
        clsdic[ele]=clsdic.get(ele,0)+1
    ags=sorted(clsdic,key=operator.itemgetter(1),reverse=True)
    return ags[0]

def createTree(dataSet, labels):
    pass

def classify(inputTree, featLabels, testVec):
    pass

def storeTree(inputTree, filename):
    pass

def grabTree(filename):
    pass

if __name__=='__main__':
    dataSet,labels=createDataSet()
    print(majorityCnt([ele[-1] for ele in dataSet]))