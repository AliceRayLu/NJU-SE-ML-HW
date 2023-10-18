from sklearn import datasets
from sklearn.metrics import f1_score
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from collections import defaultdict
from matplotlib import pyplot as plt
from math import log2
from treeplotter import create_plot

iris=datasets.load_iris()
print(iris.keys())
iris

print(iris.feature_names)
X=pd.DataFrame(iris.data,columns=iris.feature_names)
X.head()

X.info()
X.shape

X.describe()

Y=pd.DataFrame(iris.target,columns=['target'])
Y.head()


Y.info()
Y.shape

Y.describe()

Y.value_counts()


X_train, X_testAndVal, Y_train, Y_testAndVal = train_test_split(X,Y,test_size=0.2,random_state=23)
X_test, X_val, Y_test, Y_val = train_test_split(X_testAndVal,Y_testAndVal,test_size=0.5,random_state=23)


for i in [X_train, X_test, Y_train, Y_test]:
    i.reset_index(drop=True,inplace=True)

X_train.head()


X_train['target'] = Y_train # 合并标签
X_test['target'] = Y_test
X_val['target'] = Y_val

X_train.head()

class DecisionTree():

    def AttrValueSubset(self, data,attribute,val):
        '''generate the data set of a particular attribute value'''
        return data[data[attribute] == val].drop(attribute,axis=1)

    def entropy(self,data):
        '''calculate the entropy of a given dataset'''
        k = data.shape[0] # the total num of data
        label_count = defaultdict(int)
        for d in data.iloc[:,-1].values:
            label_count[d] += 1 # store labels number in dict
        entropy = 0
        for label in label_count:
            p = label_count[label] / k
            entropy -= p * log2(p)
        return entropy
    
    def feature_entropy(self,data,feature):
        '''calculate the entropy of one feature'''
        seri = data[feature].value_counts().values
        feature_entropy = 0
        for val in seri:
            p = val/np.sum(seri)
            feature_entropy -= p*log2(p)
        return feature_entropy

    def Gain(self, data, attribute, root_entropy):
        '''
        calculate the entropy gain 
        when using a particular attribute to divide
        '''
        k = data.shape[0]
        values = data[attribute].value_counts().index.to_list()
        son_entropy = 0
        for val in values:
            subset = self.AttrValueSubset(data,attribute,val)
            p = subset.shape[0]/k
            # calculate the expectation of attribute division
            son_entropy -= p*self.entropy(subset)
        return root_entropy-son_entropy
    
    def maxGainSplit(self, data, attribute, root_entropy):
        '''
        calculate the max entropy gain 
        of a given attribute dividing into two parts
        '''
        k = data.shape[0]
        values = data[attribute].value_counts().sort_values().index.to_list()
        min_entropy = float('inf')
        val = -1
        for i in range(len(values)-1):
            # divide through 2
            div_val = (values[i]+values[i+1])/2
            subset1 = data[data[attribute] <= div_val]
            subset2 = data[data[attribute] > div_val]
            new_entropy = -subset1.shape[0]/k*self.entropy(subset1)
            new_entropy -= subset2.shape[0]/k*self.entropy(subset2)
            if new_entropy < min_entropy:
                min_entropy = new_entropy
                val = div_val
        return root_entropy-min_entropy, val

    def fit(self, dataset, type):
        '''generate a decision tree from the dataset'''
        # get all the attributes names
        attributes = dataset.columns.to_list()[:-1]
        # return cases
        tmp = dataset.iloc[:,-1].value_counts().sort_values(ascending=False)
        # if there's only one target value left
        # or there's only one attribute left, return max appearance
        if tmp.shape[0] == 1 or len(attributes) == 1:
            return tmp.index.to_list()[0]

        max_attr = ""
        root_entropy = self.entropy(dataset)
        div_val_for_c45 = -1
        if type == "ID3":
            # find the max gain attribute
            max_gain = 0
            for attr in attributes:
                gain = self.Gain(dataset,attr,root_entropy)
                if gain > max_gain:
                    max_gain = gain
                    max_attr = attr
        elif type == "C4.5":
            # find the max gain ratio
            max_ratio = 0
            for attr in attributes:
                # add 0.001 in case there is 0
                max_gain, attrVal = self.maxGainSplit(dataset,attr,root_entropy)
                ratio = max_gain/(self.feature_entropy(dataset,attr)+0.00001)
                if ratio > max_ratio:
                    max_ratio = ratio
                    max_attr = attr
                    div_val_for_c45 = attrVal
        else:
            print("Algorithm not supported.")
            return            
        # do it recursively
        children = {}
        if type == "ID3":
            chosen_values = dataset[max_attr].value_counts().index.to_list()
            for val in chosen_values:
                children[val] = self.fit(self.AttrValueSubset(dataset,max_attr,val),type)
        else:
            subset1 = dataset[dataset[max_attr] <= div_val_for_c45]
            subset2 = dataset[dataset[max_attr] > div_val_for_c45]
            children[-div_val_for_c45] = self.fit(subset1,type)
            children[div_val_for_c45] = self.fit(subset2,type)
        return {max_attr:children}
    
    def searchC45(self, tree, data):
        '''find the predict result: C4.5'''
        attr = list(tree.keys())[0]
        value = abs(list(tree[attr].keys())[1])
        subtree = {}
        # if val < value, search minus subtree, else otherwise.
        if(data[attr] <= value):
            subtree = tree[attr][-value]
        else:
            subtree = tree[attr][value]
        if isinstance(subtree,dict):
            return self.searchC45(subtree,data)
        return subtree

    
    def search_tree(self, tree, data):
        '''find the predict result on tree for one specific piece of data'''
        attr = list(tree.keys())[0]
        values = tree[attr]
        min_df = float('inf')
        subtree = {}
        # find the closest value of discrete values
        for val in values:
            if abs(data[attr]-val) < min_df:
                min_df = abs(data[attr]-val)
                subtree = values[val]
        if isinstance(subtree,dict):
            return self.search_tree(subtree,data)
        return subtree
    
    def predict(self, tree, dataset, type):
        '''predict the classification result of given data'''
        results = []
        for i in range(dataset.shape[0]):
            if type == "ID3":
                results.append(self.search_tree(tree,dataset.iloc[i,:]))
            elif type == "C4.5":
                results.append(self.searchC45(tree,dataset.iloc[i,:]))
            else:
                print("Error: algorithm not supported.")
        return results

# test ID3
test_tree_ID3 = DecisionTree().fit(X_train,type="ID3")
result_ID3 = DecisionTree().predict(test_tree_ID3,X_test,type="ID3")
# test C4.5
test_tree_C45 = DecisionTree().fit(X_train,type="C4.5")
result_C45 = DecisionTree().predict(test_tree_C45,X_test,type="C4.5")

print(test_tree_ID3)

create_plot(test_tree_ID3)

create_plot(test_tree_C45)

print("ID3-Micro-F1: %f"% f1_score(list(X_test.iloc[:,-1].values),result_ID3,average="micro"))
print("C4.5-Micro-F1: %f"% f1_score(list(X_test.iloc[:,-1].values),result_C45,average="micro"))

print("ID3-Macro-F1: %f"% f1_score(list(X_test.iloc[:,-1].values),result_ID3,average="macro"))
print("C4.5-Macro-F1: %f"% f1_score(list(X_test.iloc[:,-1].values),result_C45,average="macro"))
