# -*- coding:utf-8 -*-
from varibles import *
from numpy import *
import matplotlib.pyplot as plt


def loadDataSet(fileName):
    dataMat=[];lableMat=[]
    fr=open(fileName,'r')
    for line in fr.readlines():
        lineArr=line.strip().split('\t')
        dataMat.append([float(lineArr[0]),float(lineArr[1])])
        lableMat.append(float(lineArr[2]))
    return dataMat,lableMat

def selectJrand(i,m):
    j=i
    while j==i:
        j=int(random.uniform(0,m))
    return j

def clipAlpha(alp,H,L):
    if alp>H:
        alp=H
    if alp<L:
        alp=L
    return alp

def smoSimple(dataMatIn,classLables,C,toler,MaxIter):
    """
    C is slack variable. C>=alpha>=0
    toler is rongcuolv
    """ 
    dataMatrix=mat(dataMatIn)
    lableMat=mat(classLables).transpose()
    b=0;m,n=shape(dataMatrix)
    alphas=mat(zeros((m,1)))
    iter=0
    while iter<MaxIter:
        alphaPairsChanged =0
        for i in range(m):
            #.T is equal to .transpose()
            #alphas=matrix([a1,a2,a3,..am])T,lableMat=matrix([y1,y2,y3,..ym])T
            #np.multiply(alphas,lableMat)=matrix([a1*y1,a2*y2...am*ym])T  ,multiply here does not return outer product
            #fXi=Σajyj(xj·xi)+b ,which is wT·xi+b
            fXi=(multiply(alphas,lableMat).T*(dataMatrix*dataMatrix[i,:].T))+b
            #lableMat[i] is -1 or 1,   Ei=wT·xi+b-lable[i]=[(wTx+b)*lable[k]-1]*lable[k]
            Ei = fXi - float(lableMat[i])
            #\ connect lines to one
            #lableMat[i]*Ei =lable[i]*(wT·xi+b)-lable[i]**2=lable[i]*(wT·xi+b)-1>=0
            # when inaccuracy is greater than toler(which is rongcuolv) and (alphas[i]>0 or alphas[i]<C) ,then optimize alphas[i] 
            # why not write as 'abs(lableMat[i]*Ei)>abs(toler) and (alphas[i]<C or alphas[i]>0)'
            if ((lableMat[i]*Ei < -toler) and (alphas[i] < C)) or \
                ((lableMat[i]*Ei > toler) and (alphas[i] > 0)):
                j = selectJrand(i,m)
                fXj = float(multiply(alphas,lableMat).T*(dataMatrix*dataMatrix[j,:].T)) + b
                #Ej>0 when lable[i] is 1 else then lable[i] is-1
                Ej = fXj - float(lableMat[j])
                alphaIold = alphas[i].copy()
                alphaJold = alphas[j].copy()
                # use L and H to modify alphas[j] to an area between 0,c
                # if L==H then continue
                # lableMat[i] != lableMat[j] means i,j are not on the same side of border, use -. if equal,use +
                if (lableMat[i] != lableMat[j]):
                    L = max(0, alphas[j] - alphas[i])
                    H = min(C, C + alphas[j] - alphas[i])
                else:
                    L = max(0, alphas[j] + alphas[i] - C)
                    H = min(C, alphas[j] + alphas[i])
                    # no optimization
                    if L == H:
                        print("L==H")
                        continue 
                #Eta is the optimal amount to change alpha[j]
                eta = 2.0 * dataMatrix[i, :]*dataMatrix[j, :].T - dataMatrix[i, :]*dataMatrix[i, :].T - dataMatrix[j, :]*dataMatrix[j, :].T
                #when eta = 0 , calculation of new alpha is too complex for now, so skip it
                if eta >= 0:
                    print("eta>=0")
                    continue
                #calc it
                alphas[j] -= lableMat[j]*(Ei - Ej)/eta
                # clip it
                alphas[j] = clipAlpha(alphas[j], H, L)
                # if alpha[j] has changed by a small amount
                if (abs(alphas[j] - alphaJold) < 0.00001):
                    print("j not moving enough")
                    continue
                # change alphas[i] by the same amount of lableMat[j] ,but in the opposite direction
                alphas[i] += lableMat[j]*lableMat[i]*(alphaJold - alphas[j])
                # set the constant term b for these two alphas
                # w= Σ[1~n] ai*yi*xi => b = yj- Σ[1~n] ai*yi(xi*xj)
                # b1 - b = (y1-y) - Σ[1~n] yi*(a1-a)*(xi*x1)
                b1 = b - Ei- lableMat[i]*(alphas[i]-alphaIold)*dataMatrix[i, :]*dataMatrix[i, :].T - lableMat[j]*(alphas[j]-alphaJold)*dataMatrix[i, :]*dataMatrix[j, :].T
                b2 = b - Ej- lableMat[i]*(alphas[i]-alphaIold)*dataMatrix[i, :]*dataMatrix[j, :].T - lableMat[j]*(alphas[j]-alphaJold)*dataMatrix[j, :]*dataMatrix[j, :].T
                if (0 < alphas[i]) and (C > alphas[i]):
                    b = b1
                elif (0 < alphas[j]) and (C > alphas[j]):
                    b = b2
                else:
                    b = (b1 + b2)/2.0
                alphaPairsChanged += 1
                print("iter: %d i:%d, pairs changed %d" % (iter, i, alphaPairsChanged))
        #outside the for loop , if no alpha changed the iter added by 1 ,else set iter to 0
        #if no alpha changed during maxiter ,then while loop end
        if (alphaPairsChanged == 0):
            iter += 1
        else:
            iter = 0
        print("iteration number: %d" % iter)
    return b, alphas
    
def calcWs(alphas, dataArr, classLabels):       
    X=mat(dataArr)
    lableMat=mat(labelArr).transpose()
    m,n=shape(X)
    w=mat(zeros((n,1)))
    for i in range(n):
        w+=multiply(alphas[i]*lableMat[i],X[i,:].T)
    return w


def plotfig_SVM(xMat, yMat, ws, b, alphas):
    """
    参考地址：
       http://blog.csdn.net/maoersong/article/details/24315633
       http://www.cnblogs.com/JustForCS/p/5283489.html
       http://blog.csdn.net/kkxgx/article/details/6951959
    """
    xMat = mat(xMat)
    yMat = mat(yMat)

    # b原来是矩阵，先转为数组类型后其数组大小为（1,1），所以后面加[0]，变为(1,)
    b = array(b)[0]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    # 注意flatten的用法
    ax.scatter(xMat[:, 0].flatten().A[0], xMat[:, 1].flatten().A[0])
    # x最大值，最小值根据原数据集dataArr[:, 0]的大小而定
    x = arange(-1.0, 10.0, 0.1)
    # 根据x.w + b = 0 得到，其式子展开为w0.x1 + w1.x2 + b = 0, x2就是y值
    y = (-b-ws[0, 0]*x)/ws[1, 0]
    ax.plot(x, y)
    for i in range(shape(yMat[0, :])[1]):
        if yMat[0, i] > 0:
            ax.plot(xMat[i, 0], xMat[i, 1], 'cx')
        else:
            ax.plot(xMat[i, 0], xMat[i, 1], 'kp')
    # 找到支持向量，并在图中标红
    for i in range(100):
        if alphas[i] > 0.0:
            ax.plot(xMat[i, 0], xMat[i, 1], 'ro')
    plt.show()


if __name__ == "__main__":
    dataArr, labelArr = loadDataSet(inputPath+'/6.SVM/testSet.txt')
    b, alphas = smoSimple(dataArr, labelArr, 0.6, 0.001, 40)
    print('/n/n/n')
    print('b=', b)
    print('alphas=', alphas)
    #alphas[alphas > 0],取数组中大于0的元素 只对numpy 数组可用
    print('alphas[alphas>0]=', alphas[alphas > 0])
    print('shape(alphas[alphas > 0])=', shape(alphas[alphas > 0]))
    for i in range(100):
        if alphas[i] > 0:
            print(dataArr[i], labelArr[i])
    ws = calcWs(alphas, dataArr, labelArr)
    plotfig_SVM(dataArr, labelArr, ws, b, alphas)