# -*- coding: utf-8 -*-
"""
Updated on Sat Jun 20 10:01:14 2020

@author: Larissa
"""
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from string import punctuation
from sklearn.metrics import precision_recall_fscore_support
import csv
import json

class HMMModel1:
    
    #Training the model
    def trainingModel(self,trainingFile):
        lexical_prob=dict()
        trans_prob=dict()
        start_1=0
        start_0=0
        count=[]
    
        with open(trainingFile) as csvfile:
            readCSV = csv.reader(csvfile, delimiter=',')
            #Ignore the heading
            next(readCSV, None)
            for row in readCSV:
                #read each sentence which is row[0]
                s=row[0].lower().replace('[',' ').replace(']',' ').split()
                #read the metaphor labels with respect to each word which is row[2]
                m=[int(i) for i in row[2].replace('[',' ').replace(',',' ').replace(']',' ').split()]
                dict_iteration=dict(zip(s, m))
                
                #assign counts to each (word, metaphor label) pair
                for i in dict_iteration.items():
                    if i in lexical_prob:
                        lexical_prob[i]+=1
                    else:
                        lexical_prob[i]=1
                
                #assign count of start words that are metaphor and those that are not 
                #Used for Initiaization step in Viterbi
                if(m[0]==0):
                    start_0+=1
                else:
                    start_1+=1
                    
                #Check the transition and count for each type of transition    
                for i in range(len(m)-1):
    
                    tup=(m[i],m[i+1])
                    if tup in trans_prob:
                        trans_prob[tup]+=1
                    else:
                        trans_prob[tup]=1
            #count for transitions starting from 0 and 1 respectively            
            count.append(trans_prob[(0,0)]+trans_prob[(0,1)])
            count.append(trans_prob[(1,0)]+trans_prob[(1,1)])
            return lexical_prob,trans_prob,start_1,start_0,count
        
    #Viterbi
    def viterbiFunc(self,sentenceWhole,lexical_prob,trans_prob,start_1,start_0,count,file):
        sentence=sentenceWhole.split()
    
        n=len(sentence)
        score=np.zeros((2,n))
        bptr=np.zeros((2,n), dtype=int)
        
        writer=open(file,"a+")
        len_p=len(lexical_prob)
        
        k = 0.00001
        
        #Initialization
        for i in range(2):
            #if a word, label pair not found, assign a 0 count to it. it will be handled by smoothing 
            if (sentence[0],i) not in lexical_prob:
                lexical_prob[sentence[0],i]=0
            score[i][0]=(start_0/(start_0+start_1)) * (((lexical_prob[sentence[0],i])+k)/(count[i]+k*len_p))
            bptr[i][0] = -1
            
        #Iteration
        for word in range(1,n):
            for label in range(2):
                scores_for_next_label=[]
                for j in range(2):
                    Pti=trans_prob[(j,label)]/count[j]
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

    def accuracy(self,output,actual):
        print(accuracy_score(actual,output))
        
    def fscore(self,output,actual):
        print(precision_recall_fscore_support(actual, output,average='binary'))
        

def main():
    with open("configuration.json") as json_data_file:
        data = json.load(json_data_file)
    
    #read the file names for the training set and test set
    trainPath = data['files']['train_file']
    valPath = data['files']['val_file']
    valOutPath = 'valOut.txt'
    
    hmm=HMMModel1()
    lexical_prob,trans_prob,start_1,start_0,count=hmm.trainingModel(trainPath)
    testingFile=valPath
    file=valOutPath
    tags = []
    m =[]
    #sum12=0
    with open(testingFile) as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        next(readCSV, None)
        for row in readCSV:
            #tags=Predicted values by Viterbi 
            tags.extend(hmm.viterbiFunc(row[0].lower(),lexical_prob,trans_prob,start_1,start_0,count,file))
            arr = [int(i) for i in row[2].replace('[',' ').replace(',',' ').replace(']',' ').split()]
            #m= Actual values in the train data
            m.extend(arr)
    hmm.accuracy(tags,m)   
    hmm.fscore(tags,m)
    
if __name__=='__main__':
    main()