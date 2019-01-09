from numpy import *
def loadDataSet():
    postingList=[['my','dog','has','flea','problem','help','please'],
                 ['maybe','not','take','him','to','dog','park','stupid'],
                 ['my','dalmation','is','so','cute','I','love','him'],
                 ['stop','posting','stupid','worthless','garbage'],
                 ['mr','licks','ate','my','steak','how','to','stop','him'],
                 ['quit','buying','worthless','dog','food','stupid']
                 ]
    classVec=[0,1,0,1,0,1]
    return postingList,classVec
def createVocabList(dataSet):
    #创建一个包含在所有文档中出现的不重复词的列表
    vocabSet=set([]) #创建一个空集
    for document in dataSet:
        vocabSet=vocabSet | set(document) #创建两个集合的并集
    return list(vocabSet)
def setOfWords2Vec(vocabList,inputSet): #词集模型
    #输入词汇表和输入的文档 返回文档向量 表示词汇表中的词是否在输入文件中出现
    returnVec=[0]*len(vocabList) #创建一个所含元素均为0的向量
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)]=1
        else: print("the word:%s is not in my Vocabulary!" %word)
    return returnVec
def bagOfWords2Vec(vocabList,inputSet): #词袋模型 一个词在句中出现的次数不止一次
    returnVec=[0]*len(vocabList)
    for word in inputSet:
        if word in inputSet:
            returnVec[word]+=1
    return returnVec

def trainNB0(trainMatrix,trainCategory):
    #朴素贝叶斯分类器训练  计算每个词汇在侮辱性和非侮辱性语句中出现的频率
    #文档向量矩阵及每篇文档类别标签所构成的向量
    numtrainDocs=len(trainMatrix)   #训练文档向量矩阵中有几篇文档
    numWords=len(trainMatrix[0])    #一篇文档有多少个字 已经转化为词汇表 长度固定
    pAbusive=sum(trainCategory)/float(numtrainDocs)
    p0Num=ones(numWords);p1Num=ones(numWords) #防止相乘时一个是0全部是0
    p0Denom=2.0;p1Denom=2.0     #所有词出现数初始化为1 分母初始化为2
    for i in range(numtrainDocs):
        if trainCategory[i]==1:     #当发现类别相同时
            p1Num+=trainMatrix[i]   #该词对应的个数加1
            p1Denom+=sum(trainMatrix[i]) #总数加1
        else:
            p0Num+=trainMatrix[i]
            p0Denom+=sum(trainMatrix[i])
    p1Vect=log(p1Num/p1Denom)
    p0Vect=log(p0Num/p0Denom)       #采用对数避免小数相乘产生下溢或浮点数舍入而因此产生的误差
    return p0Vect,p1Vect,pAbusive
def classifyNB(vec2Classfiy,p0Vec,p1Vec,pClass1):
    #要分类的向量ve2Classfiy,通过trainNB0训练得到的3个概率
    p1=sum(vec2Classfiy*p1Vec)+log(pClass1) #根据公式可得，用自然对数代替 log(x)+log(y)=log(xy)
    p0=sum(vec2Classfiy*p0Vec)+log(1-pClass1)
    if p1>p0:
        return 1
    else:
        return 0
def testingNB():
    listOPosts,listClasses=loadDataSet()
    myVocabList=createVocabList(listOPosts)
    trainMat=[]
    for postinDoc in listOPosts:
        trainMat.append(setOfWords2Vec(myVocabList,postinDoc))
    p0V,p1V,pAb=trainNB0(array(trainMat),array(listClasses))
    testEntry=['love','my','dalmation']
    thisDoc=array(setOfWords2Vec(myVocabList,testEntry))
    print(testEntry,'classified as: ',classifyNB(thisDoc,p0V,p1V,pAb))
    testEntry=['stupid','garbage']
    thisDoc=array(setOfWords2Vec(myVocabList,testEntry))
    print(testEntry,'classified as:',classifyNB(thisDoc,p0V,p1V,pAb))



