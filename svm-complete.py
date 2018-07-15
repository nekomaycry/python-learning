# -*- coding:utf-8 -*-
from varibles import *
from numpy import *
import matplotlib.pyplot as plt

class optStruct:
    def __init__(self,dataMatIn,classLables,C,toler,kTup):
        self.X=dataMatIn
        self.lableMat=classLables
        self.C=C
        self.tol=toler
        self.m=shape(dataMatIn)[0]
        self.alphas=mat(zeros((self.m,1)))
        self.b=0
        self.eCache=mat(zeros((self.m,2)))
        self.K=mat(zeros((self.m,self.m)))
        for i in range(self.m):
            self.K[:,i]=kernelTrans(self.X, self.X[i,:], kTup)

def loadDataSet(fileName):
    dataMat=[];lableMat=[]
    fr=open(fileName,'r')
    for line in fr.readlines():
        lineArr=line.strip().split('\t')
        dataMat.append([float(lineArr[0]),float(lineArr[1])])
        lableMat.append(float(lineArr[2]))
    return dataMat,lableMat
            
def clipAlpha(alp,H,L):
    if alp>H:
        alp=H
    if alp<L:
        alp=L
    return alp
        
def calcEk(os,k):
    #fXk=wTx+b=Σ(i=1,m)aiyi*Σ(j=1,n)X(i,n)X(k,n)+b
    fXk=float(multiply(os.alphas,os.lableMat).T*os.K[:,k])+os.b
    #fXk=float(multiply(os.alphas,os.lableMat).T*(os.X*os.X[k,:].T))+os.b
    #Ek= [(wTx+b)*lable[k]-1]*lable[k]
    Ek=fXk-float(os.lableMat[k])
    return Ek

def updateEk(os,k):
    Ek=calcEk(os, k)
    os.eCache[k]=[1,Ek]

def selectJrand(i,m):
    j=i
    while j==i:
        j=int(random.uniform(0,m))
    return j

#the goal is to choose the second alpha so that we’ll take the maximum step during each optimization
def selectJ(i,os,Ei):
    maxK=-1;maxDeltaE=0;Ej=0
    os.eCache[i]=[1,Ei]
    #nonzero return zhe index array of non-zero element
    validEcacheList=nonzero(os.eCache[:,0].A)[0]
    if len(validEcacheList)>1:
        for k in validEcacheList:
            if k==i:
                continue
            Ek=calcEk(os, k)
            deltaE=abs(Ei-Ek)
            if deltaE>maxDeltaE:
                maxDeltaE=deltaE
                maxK=k
                Ej=Ek
        return maxK,Ej
    else:
        #choose an alpha by random when this function run at 1st time
        j=selectJrand(i, os.m)
        Ek=calcEk(os, j)
        return j,Ek

def innerL(i,os,useK=True):
    Ei=calcEk(os, i)
    # when inaccuracy is greater than toler(which is rongcuolv) and (alphas[i]>0 or alphas[i]<C) ,then optimize alphas[i] 
    if ((os.lableMat[i]*Ei < -os.tol) and (os.alphas[i] < os.C)) or \
        ((os.lableMat[i]*Ei > os.tol) and (os.alphas[i] > 0)):
        #pick the maximum step
        j,Ej = selectJ(i,os,Ei)
        alphaIold=os.alphas[i].copy()
        alphaJold=os.alphas[j].copy()
        if os.lableMat[i]!=os.lableMat[j]:
            L=max(0,os.alphas[j]-os.alphas[i])
            H=min(os.C,os.C+os.alphas[j]-os.alphas[i])
        else:
            L=max(0,os.alphas[j]+os.alphas[i]-os.C)
            H=min(os.C,os.alphas[j]+os.alphas[i])
        if L==H:
            print('L==H')
            return 0
        if useK:
            eta=2.0*os.K[i,j]-os.K[i,i]-os.K[j,j]
        else:
            eta=2.0*os.X[i,:]*os.X[j,:].T-os.X[i,:]*os.X[i,:].T-os.X[j,:]*os.X[j,:].T
        if eta>=0.0:
            print('eta>=0')
            return 0
        os.alphas[j]-=os.lableMat[j]*(Ei-Ej)/eta
        os.alphas[j]=clipAlpha(os.alphas[j], H, L)
        #for alpha[j] is updated , Ej should be updated too
        updateEk(os, j)
        if os.alphas[j]-alphaJold<0.00001:
            print('alphas[j] not moving enough')
            return 0
        os.alphas[i]+=os.alphas[i]*os.alphas[j]*(alphaJold-os.alphas[j])
        updateEk(os, i)
        if useK:
            b1=os.b-Ei-os.lableMat[i]*(os.alphas[i]-alphaIold)*os.K[i,i]-os.lableMat[j]*(os.alphas[j]-alphaJold)*os.K[i,j]
            b2=os.b-Ej-os.lableMat[i]*(os.alphas[i]-alphaIold)*os.K[i,j]-os.lableMat[j]*(os.alphas[j]-alphaJold)*os.K[j,j]
        else:
            b1=os.b-Ei-os.lableMat[i]*(os.alphas[i]-alphaIold)*os.X[i,:]*os.X[i,:].T-os.lableMat[j]*(os.alphas[j]-alphaJold)*os.X[i,:]*os.X[j,:].T
            b2=os.b-Ej-os.lableMat[i]*(os.alphas[i]-alphaIold)*os.X[i,:]*os.X[j,:].T-os.lableMat[j]*(os.alphas[j]-alphaJold)*os.X[j,:]*os.X[j,:].T
        if os.alphas[i]>0 and os.alphas[i]<os.C:
            os.b=b1
        elif os.alphas[j]>0 and os.alphas[j]<os.C:
            os.b=b2
        else:
            b=(b1+b2)/2.0
        return 1
    else:
        return 0

def smoP(dataMatIn,classLables,C,toler,maxIter,kTup=('lin',0)):
    os=optStruct(mat(dataMatIn),mat(classLables).transpose(),C,toler,kTup)
    iter=0;entireSet=True;alphaPairChanged=0
    #when iter>=maxIter or in 'if entireSet:' block, no alpha changed
    while iter<maxIter and (alphaPairChanged>0 or entireSet):
        alphaPairChanged=0
        #when it come here first time or in the last else block, no alpha changed
        if entireSet:
            for i in range(os.m):
                alphaPairChanged+=innerL(i, os)
            print('fullset, iter: %d,i:%d,pairs changed: %d'%(iter,i,alphaPairChanged))
            iter+=1
        else:
            #indexes of alphas which between 0 and C
            nonBoundIs=nonzero((os.alphas.A>0)*(os.alphas.A<C))[0]
            for i in nonBoundIs:
                alphaPairChanged+=innerL(i, os)
            print('non-bound, iter: %d,i:%d,pairs changed: %d'%(iter,i,alphaPairChanged))
            iter+=1
        if entireSet:
            entireSet=False
        elif alphaPairChanged==0:
            entireSet=True
        print("iteration number: %d" % iter)
    return os.b,os.alphas

def kernelTrans(X,A,kTup):
    m,n=shape(X)
    K=mat(zeros((m,1)))
    if kTup[0]=='lin':
        K=X*A.T
    elif kTup[0]=='rbf':
        for j in range(m):
            deltaRow=X[j,:]-A
            K[j]=deltaRow*deltaRow.T
        K=exp(K/(-1*kTup[1]**2))
    else:
        raise NameError('the kernel is not recognized')
    return K

def testRbf(k1=1.3):
    dataArr,lableArr=loadDataSet(inputPath+'/6.SVM/testSetRBF.txt')
    b,alphas=smoP(dataArr, lableArr, 200, 0.0001, 10000,('rbf',k1))
    dataMat=mat(dataArr)
    lableMat=mat(lableArr).transpose()
    svInd=nonzero(alphas.A>0)[0]
    svs=dataMat[svInd]
    lableSV=lableMat[svInd]
    print('there are %d support vectors'% shape(svs)[0])
    m,n=shape(dataMat)
    errorCount=0
    for i in range(m):
        kernelEval=kernelTrans(svs, dataMat[i,:], ('rbf',k1))
        predict=kernelEval.T*multiply(lableSV,alphas[svInd])+b
        if sign(predict)!=sign(lableArr[i]):
            errorCount+=1
    print('the error rate is %f'% (float(errorCount)/m))

if __name__=='__main__':
    testRbf()
        
        
    