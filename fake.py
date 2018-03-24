#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  4 15:48:56 2018

@author: Heqi
"""

import pandas as pd
import numpy as np
from heapq import nlargest
import io
from itertools import chain
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

## get data from files
fake_dir = 'clean_fake.txt'
real_dir = 'clean_real.txt'
fake_label = 'fake'
real_label = 'real'
fake_file = io.open(fake_dir,'r+') 
real_file = io.open(real_dir,'r+')
fake_read = fake_file.read()
real_read = real_file.read()

fake_text = fake_read.split('\n')
real_text = real_read.split('\n')

fake_vocal = str.split(fake_read)
real_vocal = str.split(real_read)
vocal_list_full = list()
vocal_list_full.extend(fake_vocal)
vocal_list_full.extend(real_vocal)

def preVocallist(vocallist):
    vocallist = pd.DataFrame(data=vocallist, columns=['keywords'])
    vocallist = vocallist.drop_duplicates(subset=['keywords'],keep='first')
    vocallist = vocallist.dropna(subset=['keywords'])
    vocallist = vocallist.reset_index(drop=True)
    vocallist = list(chain(vocallist['keywords']))
  
    return vocallist

vocal_list_full = preVocallist(vocal_list_full)

data = fake_text + real_text
dataset = pd.DataFrame(data=data, columns=['text'])
dataset['class'] = fake_label
dataset['class'][len(fake_text):(len(fake_text)+len(real_text))] = real_label
## seperate the trainset,validation set and testset(0.75,0.15,0.15)
train_set = dataset.sample(frac = 0.70, random_state = 200)
validation_test = dataset.drop(train_set.index)
validation_set = validation_test.sample(frac = 0.5, random_state = 100)
test_set = validation_test.drop(validation_set.index)

train_set = train_set.reset_index(drop = True)
validation_set = validation_set.reset_index(drop = True)
test_set = test_set.reset_index(drop = True)


## convert word to AscII
def str2ascii(dataset):
    words = list(chain(dataset['text']))
    text = []
    for i in range(len(words)):
        w = words[i]
        w = ''.join(str(ord(c)) for c in w)
        text.append(w)
    dataset['text'] = text
    
#str2ascii(dataset)

def countVocal(vocallist, datalist):
    vocalcount = np.zeros(len(vocallist))
    for vocal in datalist:
        if vocal in vocallist:
           vocalcount[vocallist.index(vocal)] += 1
    return vocalcount
    
##============================part1========================
def showResult_part1():
    fakecount = countVocal(vocal_list_full, fake_vocal)
    realcount = countVocal(vocal_list_full, real_vocal)

    #three_vocal_fake = list()
    #three_vocal_real = list()
    a = nlargest(50,zip(fakecount,vocal_list_full))
    b = nlargest(50,zip(realcount,vocal_list_full))

    print("=====================part1===============================")
    print(" 3 examples of specific keywords in fake :", a[17],",", a[34], ",", a[39])
    #for i in range(len(a)):
        #print(a[i])
    print(" 3 examples of specific keywords in real :", b[20],b[28],b[31])
    #for i in range(len(b)):
        #print(b[i])

   
##==========================part2================================
        
def getVocallist(dataset):
    texts = list(chain(dataset['text']))
    vocallist = list()
    text = ''
    for i in range(len(texts)):
        t = texts[i]+'\n'
        text += t
        #print("text:", text)
        #print("i:", i)
    vocallist = str.split(text)
    return vocallist

vocal_list = getVocallist(train_set)
vocal_list = preVocallist(vocal_list)
## compute how many times the vocal exist in the text.count(xi=1/c) or count(xi=1/c-)
def probCount(vocallist, trainset):
    problist = np.zeros(len(vocallist))
    for i in range(len(vocallist)):
        for text in trainset:
            if vocallist[i] in text:
                problist[i] += 1
    return problist

def getWordprob(m,p):
    fakecount = probCount(vocal_list,train_set['text'][train_set['class'] == fake_label])
    realcount = probCount(vocal_list,train_set['text'][train_set['class'] == real_label])  
    ## p(xi=1/c)

    fakepr = (fakecount + m*p)/ (train_set[train_set['class']==fake_label].shape[0]+m)
    realpr = (realcount + m*p)/ (train_set[train_set['class']==real_label].shape[0]+m)
    ## c = spam prior
    return fakepr, realpr

m = 1
p = 0.001
fakeprob, realprob = getWordprob(m,p)
pc = (train_set[train_set['class']==fake_label].shape[0])/(train_set.shape[0])
   


def probFunction(textprob,prior_pf):
    prob = np.exp(sum(np.log(textprob)))*prior_pf

    return prob
    
## dataset is the data, problist is the fakeprob/realprob, vacallist is the list of vocal    
def getProbdata(dataset,problist,vocallist):
    textnum = dataset.shape[0]

    probdata = np.zeros((textnum, len(problist)))
    for i in range(textnum):
        probtext = problist.copy()
        probtext[:] = [1-x for x in probtext]       
        wordslist = str.split(dataset['text'][i])
     
        indexlist = list()
        for words in wordslist: 
            if words in vocallist:  
                m = vocallist.index(words)       
                indexlist.append(m)

        for k in indexlist:           
            probtext[k]=1-probtext[k]
 
         
        probdata[i,:]  =   probtext
        
        
    return probdata

##probfake is the fakeprob,probreal is the realprob            
def trainNaivebayes(dataset, vocallist, probfake,probreal, prior):
    fakelist = list()
    reallist = list()
    resultlist = list()
    fake_problist = getProbdata(dataset,probfake,vocallist)
    real_problist = getProbdata(dataset,probreal,vocallist)

    
    for k in range(dataset.shape[0]):
        a = probFunction(fake_problist[k], prior)
        b = probFunction(real_problist[k],(1-prior))
        fakelist.append(a)
        reallist.append(b)
        if a >=b:
            resultlist.append('fake')
        else:
            resultlist.append('real')

    return fakelist, reallist, resultlist


def performance(resultlist,dataset):
    count_real = 0
    count_fake = 0
    for i in range(len(resultlist)):
        if resultlist[i] == dataset['class'][i]:
            if  dataset['class'][i] == fake_label :
                count_fake += 1
            elif dataset['class'][i] == real_label:
                count_real += 1
    perf_real = count_real/dataset[dataset['class']==real_label].shape[0]
    perf_fake = count_fake/dataset[dataset['class']==fake_label].shape[0]
    perf = (count_real+count_fake)/dataset.shape[0]
    print("count_real:",count_real)
    print("count_fake:",count_fake)
    
    return perf_real, perf_fake, perf


       
def tuneParameter():
    m = np.arange(1, 50, 5)
    p = [0.001, 0.051,0.101,0.151,0.201,0.251,0.301,0.351,0.401,0.451]
    perflist = list()
    index = list()
    for i in m:
        for j in p:
            index.append([i,j])
            fake_p,real_p = getWordprob(i,j)
            result = trainNaivebayes(validation_set, vocal_list, fake_p,real_p, pc)[2]
            perf = performance(result,validation_set)[2]
            perflist.append(perf)           
    k = np.argmax(perflist)
    
    print("Using grid search to tune the hyperparameter m and p")
    for f in range(len(index)):       
        print("m,p:", index[f], "performance is :", perflist[f])
    return index[k]
                
def showResult_part2():
    global m
    global p
    prediction_validation = trainNaivebayes(validation_set, vocal_list, fakeprob,realprob, pc)[2]
    perf_real_valid, perf_fake_valid,perf_valid = performance(prediction_validation, validation_set)
    
    prediction_train = trainNaivebayes(train_set, vocal_list, fakeprob,realprob, pc)[2]
    perf_real_train, perf_fake_train,perf_train = performance(prediction_train, train_set)
    
    prediction_test = trainNaivebayes(test_set, vocal_list, fakeprob,realprob, pc)[2]
    perf_real_test, perf_fake_test,perf_test = performance(prediction_test, test_set)
    
    print("======================part2======================")
    print("m:",m ,",","p:", ",", p)
    print("Performance of real news on validation set:", perf_real_valid)
    print("Performance of fake news on validation set:", perf_fake_valid)
    print("Overall Performance on validation set:", perf_valid)
    
    print("Performance of real news on train set:", perf_real_train)
    print("Performance of fake news on train set:", perf_fake_train)
    print("Overall Performance on train set:", perf_train)
    
    print("Performance of real news on test set:", perf_real_test)
    print("Performance of fake news on test set:", perf_fake_test)
    print("Overall Performance on test set:", perf_valid)
    
###===============================part3=========================================

trainword_count = probCount(vocal_list, train_set['text'])


train_vocal_df = pd.DataFrame(data= vocal_list, columns=['words'])

train_vocal_df['wordcount'] = pd.DataFrame(trainword_count, dtype=float)
train_vocal_df['fakeprob'] = pd.DataFrame(fakeprob, dtype=float)
train_vocal_df['realprob'] = pd.DataFrame(realprob, dtype=float)
train_vocal_df['wordprob'] = train_vocal_df['wordcount']/(train_set.shape[0])
#pf_w = (train_vocal_df['fakecount'] + m*p)/ (train_vocal_df['traincount']+m)
train_vocal_df['pf_w'] = train_vocal_df['fakeprob']*pc/ train_vocal_df['wordprob'] 
train_vocal_df['pr_w'] = train_vocal_df['realprob']*(1-pc)/ train_vocal_df['wordprob'] 
train_vocal_df['1'] = pd.DataFrame(np.ones(train_vocal_df.shape[0]), dtype=float)
train_vocal_df['pf_nw'] = (train_vocal_df['1']- train_vocal_df['fakeprob'])*pc/(train_vocal_df['1']- train_vocal_df['wordprob'])
train_vocal_df['pr_nw'] = (train_vocal_df['1']- train_vocal_df['realprob'])*(1-pc)/(train_vocal_df['1']- train_vocal_df['wordprob'])
train_vocal_df.drop(['1'], axis=1, inplace=True)


def showResult_part3a():
    present_fake = train_vocal_df.nlargest(10, 'pf_w')
    present_real = train_vocal_df.nlargest(10, 'pr_w')
    absence_fake = train_vocal_df.nsmallest(10, 'pf_w')
    absence_real = train_vocal_df.nsmallest(10, 'pr_w')

    present_fake = list(chain(present_fake['words']))
    present_real = list(chain(present_real['words']))
    absence_fake = list(chain(absence_fake['words']))
    absence_real = list(chain(absence_real['words']))
    
    print("10 words presence most strongly predicts as real:")
    print("===================")
    for i in range(10):
        print(present_real[i])
    print("===================")
    print("10 words ansence most strongly predicts as real:")
    print("===================")
    for i in range(10):
        print(absence_real[i])
    print("===================")
    print("10 words presence most strongly predicts as fake:")
    print("===================")
    for i in range(10):
        print(present_fake[i])
    print("===================")
    print("10 words ansence most strongly predicts as fake:")
    print("===================")
    for i in range(10):
        print(absence_fake[i])
    print("===================")
##=================part3b===========================##
    
def showResult_part3b():
    vocal_notstop = train_vocal_df[~train_vocal_df['words'].isin(list(ENGLISH_STOP_WORDS))]
    present_fake = vocal_notstop.nlargest(10, 'pf_w')
    present_real = vocal_notstop.nlargest(10, 'pr_w')
    absence_fake = vocal_notstop.nsmallest(10, 'pf_w')
    absence_real = vocal_notstop.nsmallest(10, 'pr_w')
    
    present_fake = list(chain(present_fake['words']))
    present_real = list(chain(present_real['words']))
    absence_fake = list(chain(absence_fake['words']))
    absence_real = list(chain(absence_real['words']))
    
    print("10 non-stopwords presence most strongly predicts as real:")
    print("===================")
    for i in range(10):
        print(present_real[i])
    print("===================")
    print("10 non-stopwords presence most strongly predicts as fake:")
    print("===================")
    for i in range(10):
        print(present_fake[i])
    print("===================")   

##=======================part8a==================###

def getEntropyatom(prob):
    return -prob*np.log(prob)

def mutualInfo(pf,pr,p_x,p_nx,pf_x,pr_x,pf_nx, pr_nx):
    mi = getEntropyatom(pf)+getEntropyatom(pr)-[p_x*(getEntropyatom(pf_x)+getEntropyatom(pr_x))+ p_nx*(getEntropyatom(pf_nx)+getEntropyatom(pr_nx))]
    return mi


def getMIonnode(a, b, word):
    pf = a/(a+b)
    pr = b/(a+b)
    word_df = train_vocal_df[train_vocal_df['words'] == word]
    p_x = word_df['wordprob'].iloc[0]
    p_nx = 1-p_x
    pf_x = word_df['pf_w'].iloc[0]
    pr_x = word_df['pr_w'].iloc[0]
    pf_nx = word_df['pf_nw'].iloc[0]
    pr_nx = word_df['pr_nw'].iloc[0]
    
    result = mutualInfo(pf,pr,p_x,p_nx,pf_x,pr_x,pf_nx, pr_nx)
    print("The mutual information of the first split on the training data:", result)

fakenum = train_set[train_set['class'] == fake_label].shape[0]
realnum = train_set[train_set['class'] == real_label].shape[0]
##The first split is 'the' [905,1381]   
def showResult_part8a():
    getMIonnode(fakenum, realnum, 'the')
    
   
def showResult_part8b():
    getMIonnode(fakenum, realnum, 'hillary')
       

def main():
    showResult_part1()
    showResult_part2()
    showResult_part3a()
    showResult_part3b()
    showResult_part8a()
    showResult_part8b()

if __name__=="__main__":
    main()
    
print ("-_- Thanks!!!")
 