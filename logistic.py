from varibles import *
import numpy as np

def load_data_set():
    fr=open(inputPath+'5.Logistic/TestSet.txt','r')
    data_mat=[];lable_mat=[]
    for line in fr.readlines():
        lineArr=line.strip().split()
        data_mat.append([1.0,float(lineArr[0]),float(lineArr[1])])
        lable_mat.append(int(lineArr[2]))
    return data_mat,lable_mat

def sigmoid(inputX):
    return 1.0/(1+np.exp(-inputX))

#gradascent
def gradascent(data_mat,lable_mat):
    datamatrix=np.mat(data_mat)
    lablematrix=np.mat(lable_mat).transpose()# to a column vector
    m,n=np.shape(data_mat)
    weight=np.ones((n,1))
    alpha=0.01
    maxcycles=500
    #fitting function(which is nihe hanshu) hi(¦È)=¦²(j=1,n)¦Èj*xij
    #loss function j(¦È)=(1/2m)*¦²(i=1,m)(yi-hi(¦È))**2
    #grad j(¦È)=(-1/m)¦²(i=1,m)(yi-hi(¦È))xij
    #to minimize the loss ,update the ¦È with the nagative grad, writen as follow:
    #¦È=¦È+(1/m)¦²(i=1,m)(yi-hi(¦È))xij
    for i in range(maxcycles):
        h=sigmoid(datamatrix*weight)#h is a column vector
        error=(lablematrix-h)
        weight=weight+alpha*datamatrix.transpose()*error
    return weight

#random grad ascent 
def stocGradAscent0(data_mat,lable_mat):
    m,n=np.shape(data_mat)
    weight=np.ones(n)
    alpha=0.01
        #¦È=¦È+(yi-hi(¦È))xij
    for i in range(m):
        h=sigmoid(sum(data_mat[i]*weight))#h is a number
        error=lable_mat[i]-h
        weight=weight+alpha*error*np.array(data_mat[i])
    return weight    
        
def stocGradAscent1(data_mat, class_labels, num_iter=150):
    m,n=np.shape(data_mat)
    weight=np.ones(n)
    alpha=0.01
    for i in range(num_iter):
        data_index=[i for i in range(m)]
        for j in range(m):
            alpha=4/(i+j+1.0)+0.01
            rand_index=int(np.random.uniform(0,len(data_index)))
            h=sigmoid(sum(data_mat[rand_index]*weight))
            error=class_labels[rand_index]-h
            weight=weight+alpha*error*np.array(data_mat[rand_index])
            del(data_index[rand_index])
    return weight

def classifyVector(inX,weight):
    result=sigmoid(sum(inX*weight))
    if result>0.5:
        return 1.0
    else:
        return 0

def colicTest():
    frTrain = open(inputPath+'5.Logistic/horseColicTraining.txt','r')
    frTest = open(inputPath+'5.Logistic/horseColicTest.txt','r')
    training_set=[];training_lables=[]
    for line in frTrain.readlines():
        currLine=line.strip().split('\t')
        lineArr=[]
        for i in range(21):
            lineArr.append(float(currLine[i]))
        training_set.append(lineArr)
        training_lables.append(float(currLine[21]))
    weights=stocGradAscent1(np.array(training_set), training_lables,500)
    errorCount=0;totalCount=0
    for line in frTest.readlines():
        currLine=line.strip().split('\t')
        lineArr=[]
        for i in range(21):
            lineArr.append(float(currLine[i]))
        result=classifyVector(np.array(lineArr), weights)
        if int(result)!=int(currLine[21]):
            errorCount+=1;
        totalCount+=1;
    print('the error rate is {:.2%}'.format(float(errorCount)/totalCount))

def plotBestFit(wei):
    import matplotlib.pyplot as plt
    data_mat,lable_mat=load_data_set()
    if isinstance(wei, np.matrix):
        weights=wei.getA()
    else:
        weights=wei
    dataArr=np.array(data_mat)
    n=np.shape(dataArr)[0]
    xcord1=[];ycord1=[]
    xcord2=[];ycord2=[]
    for i in range(n):
        if int(lable_mat[i])==1:
            xcord1.append(dataArr[i][1])
            ycord1.append(dataArr[i][2])
        else:
            xcord2.append(dataArr[i][1])
            ycord2.append(dataArr[i][2])            
    fig=plt.figure();
    ax=fig.add_subplot(111)
    ax.scatter(xcord1,ycord1,s=30,c='red')
    ax.scatter(xcord2,ycord2,s=30,c='green')
    x=np.arange(-3.0,3.0,0.1)
    '''
    dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
    w0*x0+w1*x1+w2*x2=f(x) 
    x0 is 1,x1 is x,x2 is y,then:
    w0+w1*x+w2*y=f(x)
    f(x)=0->y=(-w0-w1x)/w2
    '''
    y=(-weights[0]-weights[1]*x)/weights[2]
    ax.plot(x,y)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()
    
def plot_test():
    data_mat,lable_mat=load_data_set()
     #weight=gradascent(data_mat, lable_mat)
    weight=stocGradAscent1(data_mat, lable_mat)
    plotBestFit(weight)    

def testLR():
    data_mat,lable_mat=load_data_set()
    #weight=gradascent(data_mat, lable_mat)
    weight=stocGradAscent0(data_mat, lable_mat)
    print(weight)
    

if __name__=='__main__':
    #plot_test()
    colicTest()