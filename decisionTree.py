from numpy import *
from math import log
import operator
import pickle
from varibles import *
#import decisionTreePlot as dtPlot

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
    baseEnt=calcShannonEnt(dataSet)
    baseinfogain,index=0.0,-1
    for i in range(n-1):
        vals=[ele[i] for ele in dataSet]
        unqVals=set(vals)
        entsum=0;
        for val in unqVals:
            spld=splitDataSet(dataSet, i, val)
            entsum+=calcShannonEnt(spld)*len(spld)/m#yeah the p huh?
        infogain=baseEnt-entsum
        if infogain>baseinfogain:
            baseinfogain=infogain
            index=i
    return index

def majorityCnt(classList):
    clsdic={}
    for ele in classList:
        clsdic[ele]=clsdic.get(ele,0)+1
    ags=sorted(clsdic,key=operator.itemgetter(1),reverse=True)
    return ags[0]

def createTree01(dataSet, labels_):
    classlist=[ele[-1] for ele in dataSet]
    if classlist.count(classlist[0])==len(classlist):
        return classlist[0]
    if shape(dataSet)[1]==0:
        return majorityCnt(labels_)
    labels=labels_[:]
    index=chooseBestFeatureToSplit(dataSet)
    bestFeatLabel=labels[index]    
    myTree = {bestFeatLabel: {}}# this is the mtfker    
    del(labels[index])
    unqFeat=set([ele[index] for ele in dataSet])
    for val in unqFeat:
        myTree[bestFeatLabel][val]=createTree(splitDataSet(dataSet, index, 
            val), labels)#oh shit
    return myTree

def createTree(dataSet, labels):
    """
    Desc:
        创建决策树
    Args:
        dataSet -- 要创建决策树的训练数据集
        labels -- 训练数据集中特征对应的含义的labels，不是目标变量
    Returns:
        myTree -- 创建完成的决策树
    """
    classList = [example[-1] for example in dataSet]
    # 如果数据集的最后一列的第一个值出现的次数=整个集合的数量，也就说只有一个类别，就只直接返回结果就行
    # 第一个停止条件：所有的类标签完全相同，则直接返回该类标签。
    # count() 函数是统计括号中的值在list中出现的次数
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    # 如果数据集只有1列，那么最初出现label次数最多的一类，作为结果
    # 第二个停止条件：使用完了所有特征，仍然不能将数据集划分成仅包含唯一类别的分组。
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)

    # 选择最优的列，得到最优列对应的label含义,特征的名称
    bestFeat = chooseBestFeatureToSplit(dataSet)
    # 获取label的名称
    bestFeatLabel = labels[bestFeat]
    # 初始化myTree
    myTree = {bestFeatLabel: {}}
    # 注：labels列表是可变对象，在PYTHON函数中作为参数时传址引用，能够被全局修改
    # 所以这行代码导致函数外的同名变量被删除了元素，造成例句无法执行，提示'no surfacing' is not in list
    del(labels[bestFeat])
    # 取出最优列，然后它的branch做分类
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        # 求出剩余的标签label
        subLabels = labels[:]
        # 遍历当前选择特征包含的所有属性值，在每个数据集划分上递归调用函数createTree()
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels)
        # print('myTree', value, myTree)
    return myTree


def classify(inputTree, featLabels, testVec):
    fstfe=list(inputTree.keys())[0]
    index=featLabels.index(fstfe)
    valueOfFeat=inputTree[fstfe][testVec[index]]
    if isinstance(valueOfFeat, dict):
        return classify(valueOfFeat, featLabels, testVec)
    else:
        return valueOfFeat
    
    
def test1(testVec):
    dataSet,labels=createDataSet()
    myTree=createTree(dataSet, labels)
    return classify(myTree, labels, testVec)
    
def test2():

    fr = open(inputPath+'3.DecisionTree/lenses.txt')

    lenses = [inst.strip().split('\t') for inst in fr.readlines()]

    lensesLabels = ['age', 'prescript', 'astigmatic', 'tearRate']

    lensesTree = createTree(lenses, lensesLabels)
    print(lensesTree)
    #dtPlot.createPlot(lensesTree)
    
def storeTree(inputTree, filename):
    file=open(filename,'wb')
    pickle.dump(inputTree, file)
    file.close()

def grabTree(filename):
    file=open(filename,'rb')
    return pickle.load(file)

if __name__=='__main__':
    #print(test1([1,1]))
    test2()