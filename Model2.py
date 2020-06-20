# -*- coding: utf-8 -*-
"""
Updated on Sat Jun 20 14:49:27 2020

@author: larissa
"""

import pandas as pd
import numpy as np
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support
import csv
from nltk.classify import maxent
import json

class HMMModel2:
    
    def trainingModel(self,trainingFile):
        lexical_prob=dict()
        trans_prob=dict()
        start_1=0
        start_0=0
        count=[]
    
        with open(trainingFile) as csvfile:
            readCSV = csv.reader(csvfile, delimiter=',')
            next(readCSV, None)
            for row in readCSV:
                s=row[0].replace('[',' ').replace(']',' ').split()
                m=[int(i) for i in row[2].replace('[',' ').replace(',',' ').replace(']',' ').split()]
                dict_iteration=dict(zip(s, m))
                for i in dict_iteration.items():
                    if i in lexical_prob:
                        lexical_prob[i]+=1
                    else:
                        lexical_prob[i]=1
    
                if(m[0]==0):
                    start_0+=1
                else:
                    start_1+=1
                for i in range(len(m)-1):
    
                    tup=(m[i],m[i+1])
                    if tup in trans_prob:
                        trans_prob[tup]+=1
                    else:
                        trans_prob[tup]=1
            count.append(trans_prob[(0,0)]+trans_prob[(0,1)])
            count.append(trans_prob[(1,0)]+trans_prob[(1,1)])
            return lexical_prob,trans_prob,start_1,start_0,count
    
    def accuracy(self,output,actual):
        print(accuracy_score(actual,output))
        
    def fscore(self,output,actual):
        print(f1_score(actual, output))
        print(precision_recall_fscore_support(actual, output,average='binary'))
        
    #Features used- word, part of speech of that word, length of the word,position of the word , part of speech of previous word, 
    #part of speech of next word
    def get_train_tuple(self,word,pos,tag,length,word_position,prev_pos,next_pos):
        feature_dict = {"POS":pos, "word":word, "length":length, "position":word_position, "prev_pos":prev_pos, "next_ps":next_pos}
        return (feature_dict,tag)
    
    def get_test_dict(self,word,pos,length,word_position,prev_pos,next_pos):
        feature_dict = {"POS":pos, "word":word, "length":length, "position":word_position, "prev_pos":prev_pos, "next_ps":next_pos}
        return feature_dict

    def getTrain(self,trainPath):
        
        train = []
        with open(trainPath) as csvfile:
                readCSV = csv.reader(csvfile, delimiter=',')
                next(readCSV, None)
                for row in readCSV:
                    s=row[0].replace('[',' ').replace(']',' ').split()
                    m=[int(i) for i in row[2].replace('[',' ').replace(',',' ').replace(']',' ').split()]
                    p = row[1].replace('[',' ').replace(',',' ').replace(']',' ').replace('\'','').split()
                    for i in range(len(s)):
                        #associate start pos with first word's previus pos and end pos for last word's next pos
                        if(i==0):
                            prev_pos='start'
                        else:
                            prev_pos=p[i-1]
                            
                        if(i==(len(s)-1)):
                            next_pos='end'
                        else:
                            next_pos=p[i+1]
                            
                        train.append(self.get_train_tuple(s[i],p[i],m[i],len(s[i]),i,prev_pos,next_pos))
    
        return train

    #encoded the maxEnt classifer using the tuples created in getTrain function
    #Use the encoded part to train the MaxentClassifier
        
    def train_maxEnt(self,train):
        #global train
        encoding = maxent.TypedMaxentFeatureEncoding.train(
        train, count_cutoff=3, alwayson_features=True)
        classifier = maxent.MaxentClassifier.train(
        train, bernoulli=False, encoding=encoding, trace=0,max_iter=15)
        return classifier
    
    
    #Create tuples to be passed to the Viterbi Function
    def getTest(self,filePath):
        test = []
        with open(filePath) as csvfile:
                readCSV = csv.reader(csvfile, delimiter=',')
                next(readCSV, None)
                for row in readCSV:
                    s=row[0].replace('[',' ').replace(']',' ').split()
                    #m=[int(i) for i in row[2].replace('[',' ').replace(',',' ').replace(']',' ').split()]
                    p = row[1].replace('[',' ').replace(',',' ').replace(']',' ').replace('\'','').split()
                    for i in range(len(s)):
                        
                        if(i==0):
                            prev_pos='start'
                        else:
                            prev_pos=p[i-1]
                            
                        if(i==(len(s)-1)):
                            next_pos='end'
                        else:
                            next_pos=p[i+1]
                            
                        test.append(self.get_test_dict(s[i],p[i],len(s[i]),i,prev_pos,next_pos))
        return test

    
    #Viterbi using Maximum Entropy Markov Model
    def viterbiFuncMEMM(self,sentenceWhole,lexical_prob,start_1,start_0,count,file, disb):
        sentence=sentenceWhole.split()
        n=len(sentence)
        score=np.zeros((2,n))
        bptr=np.zeros((2,n), dtype=int)
        
        writer=open(file,"a+")
        
        features = [{},{}] #eg -> TODO
        len_p=len(lexical_prob)
        
        k = 0.000001
        
        #Initialization
        for i in range(2):
            if (sentence[0],i) not in lexical_prob:
                lexical_prob[sentence[0],i]=0
            score[i][0]= disb[0].prob(i) * (((lexical_prob[sentence[0],i])+k)/(count[i]+k*len_p))
            bptr[i][0] = -1
            
        #Iteration
        for word in range(1,n):
            for label in range(2):
                scores_for_next_label=[]
                for j in range(2):
                    Pti=disb[word].prob(label)
                    sc=score[j][word-1]*Pti
                    scores_for_next_label.append(sc)
                if (sentence[word],label) not in lexical_prob:
                    lexical_prob[(sentence[word],label)]=0
                Pwi=(lexical_prob[(sentence[word],label)]+k)/(count[label]+k*len_p)
                score[label][word]=max(scores_for_next_label)*Pwi
                bptr[label][word]=round(scores_for_next_label.index(max(scores_for_next_label)))
                
        #Backtracking
        t=[-1]*n
          
        t[n-1] = np.where(score == (max(score[i][n-1] for i in range(2))))[0][0]
        for i in range(n-2,-1,-1):
            t[i]=bptr[t[i+1]][i+1]
        for i in t:
            writer.write(str(i)+"\n")
        writer.close()
        return t
    
#Run the code for validation and see the accuracy and fscore and precision and recall

def main():
    
    with open("configuration.json") as json_data_file:
        data = json.load(json_data_file)
    
    #read the file names for the training set and test set
    trainPath = data['files']['train_file']
    valPath = data['files']['val_file']
    valOutPath = 'valOut.txt'
    me=HMMModel2()
    #valPath = 'data['files']['test_file']
    test = me.getTest(valPath)
    train = me.getTrain(trainPath)
    classifier = me.train_maxEnt(train)
    classifier.classify_many(test)
    disb = classifier.prob_classify_many(test)
    #global disb
    lexical_prob,trans_prob,start_1,start_0,count=me.trainingModel(trainPath)
    testingFile=valPath
    file=valOutPath
    global tags
    global m
    tags = []
    m = []
    with open(testingFile) as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        next(readCSV, None)
        for row in readCSV:
            tags.extend(me.viterbiFuncMEMM(row[0],lexical_prob,start_1,start_0,count,file,disb))
            arr = [int(i) for i in row[2].replace('[',' ').replace(',',' ').replace(']',' ').split()]
            m.extend(arr)
    me.accuracy(tags,m)
    me.fscore(tags,m)
    
if __name__=='__main__':
    main()
