from numpy import *
from varibles import *

def load_data_set():
    posting_list = [
        ['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
        ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
        ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
        ['stop', 'posting', 'stupid', 'worthless', 'gar e'],
        ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
        ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    class_vec = [0, 1, 0, 1, 0, 1] 
    return posting_list, class_vec

def create_vocab_list(data_set):
    #use set and |
    vocabs=set([])
    datasize=len(data_set)
    for i in range(datasize):
        vocabs=vocabs|set(data_set[i])
    return list(vocabs)

def setOfwords2vec(vocab_list, input_set):
    re=zeros(len(vocab_list))
    for w in input_set:
        if w in vocab_list:
            re[vocab_list.index(w)]=1
        else:
            print('the word %s is not in the vocablist'%w)
    return re

def bagOfwords2vec(vocab_list, input_set):
    re=zeros(len(vocab_list))
    for w in input_set:
        if w in vocab_list:
            re[vocab_list.index(w)]+=1
        else:
            print('the word %s is not in the vocablist'%w)
    return re    

def trainNB(trainMatrix,trainCatelog):
    m,n=shape(trainMatrix)
    pAbuse=sum(trainCatelog)/float(m)
    p0num,p1num=ones(n),ones(n)
    p0Denom,p1Denom=2.0,2.0
    for i in range(m):
        if trainCatelog[i]==1:
            p1num+=trainMatrix[i]
            p1Denom+=sum(trainMatrix[i])#sum([1,2,3])-->6
        else:
            p0num+=trainMatrix[i]
            p0Denom+=sum(trainMatrix[i])
    
    p1vect=log(p1num/p1Denom)#the log of numpy
    p0vect=log(p0num/p0Denom)#the possibility
    return p1vect,p0vect,pAbuse

def classify(testvec,p1vect,p0vect,pAbuse):
    p0=sum(p0vect*testvec)+log(1-pAbuse)#ln(p(w|c1)p(c1))=(ln(p(w1|c1))+ln(p(w2|c1))+...)+ln(p(c1))
    p1=sum(p1vect*testvec)+log(pAbuse)
    if p0>p1:
        return 0
    else:
        return 1
    
def testing_naive_bayes():
    list_post, list_classes = load_data_set()
    vocab_list = create_vocab_list(list_post)
    train_mat = []
    for post_in in list_post:
        train_mat.append(
            setOfwords2vec(vocab_list, post_in)
        )
    p0v, p1v, p_abusive = trainNB(train_mat, list_classes)
    test_one = ['love', 'my', 'dalmation']
    test_one_doc = array(setOfwords2vec(vocab_list, test_one))
    print('the result is: {}'.format(classify(test_one_doc, p0v, p1v, p_abusive)))
    test_two = ['stupid', 'garbage']
    test_two_doc = array(setOfwords2vec(vocab_list, test_two))
    print('the result is: {}'.format(classify(test_two_doc, p0v, p1v, p_abusive)))

def textParse(tx):
    import re
    reg=re.compile('\\W+')#compile 
    splited=reg.split(tx)
    return [ele.lower() for ele in splited if len(ele)>2]

def spamTest():
    doclist,classlist=[],[]
    for i in range(1,26):
        txt=textParse(open(inputPath+'4.NaiveBayes/email/spam/%d.txt'%i).read())#textParse? faq
        doclist.append(txt)
        classlist.append(1)
        try:
            txt=textParse(open(inputPath+'4.NaiveBayes/email/ham/%d.txt'%i).read())#textParse? faq
        except:
            txt = textParse(open(inputPath+'4.NaiveBayes/email/ham/%d.txt'%i, encoding='Windows 1252').read())        
       
        doclist.append(txt)
        classlist.append(0)
    vocablist=create_vocab_list(doclist)
    import random
    testingSet=[int(j) for j in random.sample(range(50), 10)]
    trainingSet=list(set(range(50))-set(testingSet))
    trainMat,trainClass=[],[]
    for docIndex in trainingSet:
        trainMat.append(bagOfwords2vec(vocablist, doclist[docIndex]))
        trainClass.append(classlist[docIndex])
    p1vect,p0vect,pAbuse=trainNB(trainMat, trainClass)
    errorcount=0
    for testIndex in testingSet:
        re=classify(bagOfwords2vec(vocablist, doclist[testIndex]), p1vect, p0vect, pAbuse)
        errorcount+=re!=classlist[testIndex]
    print('the error rate is {:.2%}'.format((errorcount/len(doclist))))
    
#--------rss---------
def calc_most_freq(vocab_list, full_text):
    from operator import itemgetter
    freqDict={}
    for tok in vocab_list:
        freqDict[tok]=full_text.count(tok)
    sortedFreq=sorted(freqDict.items(),key=itemgetter(1),reverse=True)
    return sortedFreq[0:30]





if __name__=='__main__':
    import feedparser
    testing_naive_bayes()
    #spamTest()