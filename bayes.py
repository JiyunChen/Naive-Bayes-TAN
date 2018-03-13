# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 20:38:59 2018

@author: chenj
"""

#import pandas as pd
from pandas.core.frame import DataFrame
import math
from scipy.io import arff
import sys
import copy
'''
def readArff(fileName):  
    arffFile = open(fileName,'r')  
    data = []; d={}  
    for line in arffFile.readlines():  
        if not (line.startswith('@')):  
            if not (line.startswith('%')):  
                if line !='\n':  
                    L=line.strip('\n')  
                    k=L.split(',')  
                    data.append(k)                      
        else:
            if (line.startswith("@attribute")):
                J=line.strip('\n')
                key=re.findall(r"\'(.+?)\'",J)[0]
                value=re.findall(r"{(.+?)}",J)[0]
                value=re.sub('\s','',value)
                value=value.split(",")
                d[key]=value
    arffFile.close()
    data=DataFrame(data);feature=list(d.keys())
    data.columns=feature
    return [data,feature,d]
#use=readArff("lymph_train.arff")
#train=use[0]
#feature=use[-1]
'''
def naive_bayes_p(train,feature,x,x_value,y_value):
    temp=train.loc[train['class']==y_value.encode()]
    num_y=temp.shape[0]
    num_x=temp.loc[temp[x]==x_value].shape[0]
    l=len(feature[x])
    p=(num_x+1)/(num_y+l)
    return p

#use_test=readArff("lymph_test.arff")
#test=use_test[0]

def naive_bayes(train,test,feature,metanames):
    f=copy.deepcopy(metanames);f.remove("class")
    result=[]
    for i in range(test.shape[0]):
        d={};n=0;
        for index in feature["class"]:
            p=(train.loc[train['class']==index.encode()].shape[0]+1)/(train.shape[0]+len(feature["class"]))
            for j in f:
                temp=naive_bayes_p(train,feature,j,test.at[i,j],index)
                p=temp*p
            d[index]=p
            if p>n:
                best=index
                n=p
        result.append([best,d[best]/sum(list(d.values()))])
    return result
#naive_bayes(train,test,feature,meta.names())
    
def print_naive(train,test,feature,metanames):
    pre=naive_bayes(train,test,feature,metanames)
    f=copy.deepcopy(metanames);f.remove("class");
    for index in f:
        print(index," class")
    actual=test["class"];num=0
    print()
    for i in range(len(actual)):
        print(pre[i][0],actual[i].decode(),format(pre[i][1],'0.12f'))
        if pre[i][0]==actual[i].decode():
            num=num+1
    print()
    print(num)     
#print_naive(train,test,feature,meta.names())

######################### tan ###############################
def cond_mutu_infor(train,feature,x_i,x_j):
    """
    to calculate the conditional mutual information between the x_i and x_j
    """
    if x_i==x_j:
        return -1
    else:
        p_all=[];l=len(feature["class"]);all_num=train.shape[0];
        for y_value in feature["class"]:
            num_y_value=train.loc[train["class"]==y_value.encode()].shape[0];
            for index_i in feature[x_i]:
                index_i=index_i.encode()
                temp_i=train.loc[train[x_i]==index_i]
                p_x_i_y=naive_bayes_p(train,feature,x_i,index_i,y_value)
                for index_j in feature[x_j]:
                    
                    index_j=index_j.encode()
                    temp_i_j=temp_i.loc[temp_i[x_j]==index_j]
                    p_x_j_y=naive_bayes_p(train,feature,x_j,index_j,y_value)
                    #y_value=y_value.encode()
                    num_y_i_j=temp_i_j.loc[temp_i_j["class"]==y_value.encode()].shape[0]
                    c=len(feature[x_i])*len(feature[x_j])
                    p_x_i_j_y=(num_y_i_j+1)/(num_y_value+c)
                    all_c=c*l
                    pxijy=(num_y_i_j+1)/(all_num+all_c)
                    p_use=pxijy*math.log(p_x_i_j_y/(p_x_i_y*p_x_j_y),2)
                    p_all.append(p_use)
        #print(p_all)
    I=sum(p_all)
    return I

def build_the_tree(train,feature,metanames):
    attr=copy.deepcopy(metanames);attr.remove("class");
    node=attr[0];record={};record[node]="";
    attr.remove(node)
    tree=[];tree.append(node)
    while len(attr)!=0:
        n=0;
        for t in tree:
            for index in attr:
                n_best=cond_mutu_infor(train,feature,t,index)
                if n_best>n:
                    n=n_best
                    node=index
                    parent=t
        record[node]=parent
        attr.remove(node)
        tree.append(node)
    return record
#res=build_the_tree(train,feature,meta.names())      

def print_the_tree(feature,tree,metanames):
    f=copy.deepcopy(metanames)
    f.remove("class")
    for index in f:
        print(index+" "+tree[index]+" "+"class")
#print_the_tree(feature,res,meta.names())

def pro_child(train,feature,child,child_value,parent,parent_value,y_value):
    temp_y=train.loc[train["class"]==y_value.encode()]
    temp_y_p=temp_y.loc[temp_y[parent]==parent_value]
    num_y_p=temp_y_p.shape[0]
    temp_y_p_c=temp_y_p.loc[temp_y_p[child]==child_value]
    num_y_p_c=temp_y_p_c.shape[0]
    l=len(feature[child])
    p=(num_y_p_c+1)/(num_y_p+l)
    return p

def predict_tan(train,test,feature,model,metanames):
    f=copy.deepcopy(metanames);f.remove("class");result=[]
    for i in range(test.shape[0]):
        d={};n=0
        for y_value in feature["class"]:
            p=(train.loc[train['class']==y_value.encode()].shape[0]+1)/(train.shape[0]+len(feature["class"]))
            for index in f:
                if index==f[0]:
                    p=p*naive_bayes_p(train,feature,index,test.at[i,index],y_value)
                else:
                    p=p*pro_child(train,feature,index,test.at[i,index],model[index],test.at[i,model[index]],y_value)
            d[y_value]=p 
            if p>n:
                best=y_value
                n=p
        result.append([best,d[best]/sum(list(d.values()))])
    return result
#predict_tan(train,test,feature,res,meta.names())

def print_tan(train,test,feature,metanames):
    f=copy.deepcopy(metanames);f.remove("class");
    res=build_the_tree(train,feature,metanames)
    pre=predict_tan(train,test,feature,res,metanames)
    print_the_tree(feature,res,metanames)
    print()
    actual=test["class"];num=0
    for i in range(len(actual)):
        print(pre[i][0],actual[i].decode(),format(pre[i][1],'0.12f'))
        if pre[i][0]==actual[i].decode():
            num=num+1
    print()
    print(num)    
'''
import datetime as dt  
times1 = dt.datetime.now()
print_tan("lymph_train.arff","lymph_test.arff")
times2 = dt.datetime.now()
print('Time spent: '+ str(times2-times1))
'''

def main():
    trainset,meta = arff.loadarff(open(sys.argv[1]))
    feature={}
    for i in meta.names():
        feature[i]=list(meta[i][1])
    train = DataFrame(trainset,columns=meta.names())
    testset,meta = arff.loadarff(open(sys.argv[2]))
    test = DataFrame(testset,columns=meta.names())
    optional=sys.argv[3]
    if optional == "n":
        print_naive(train,test,feature,meta.names())
    elif optional == 't':
        print_tan(train,test,feature,meta.names())
        
if __name__ == '__main__':
    main()       
