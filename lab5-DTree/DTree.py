from sklearn import datasets
from sklearn.metrics import f1_score
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from collections import defaultdict
from matplotlib import pyplot as plt
from math import log2
from treeplotter import create_plot

# sklearn中存储有iris数据集，直接使用sklearn中的数据集
iris=datasets.load_iris()
print(iris.keys())
iris
# 进一步查看data/target/feature_names等属性。使用pandas格式化数据集，将属性数据(data)和标签数据(target)分别读取存储。
print(iris.feature_names)
X=pd.DataFrame(iris.data,columns=iris.feature_names)
X.head()

# 数据集具有4列属性值，即4种不同属性。接下来使用pandas的api描述数据的一些属性。
X.info()
X.shape

X.describe()

# 数据集一共有150条数据，都以float数据类型存储。同时，数据集中不存在`null`数据，不需要对空值数据进行处理。
# 
# 接下来处理target部分。

Y=pd.DataFrame(iris.target,columns=['target'])
Y.head()

Y.info()
Y.shape

Y.describe()
Y.value_counts()

# 由上述代码可知类别标签有三种类型，同样不需要处理空值情况。
# 将数据集划分为20%测试集和80%训练集。从上述`value_counts`统计来看，每一类在数据集中的数量是相同的，因此可以抽样划分训练集和测试集。
# 
# 为了方便后续处理，将测试集再一次分为验证集和测试集，总体训练集、验证集、测试集的比例为8:1:1.

X_train, X_testAndVal, Y_train, Y_testAndVal = train_test_split(X,Y,test_size=0.2,random_state=23)
X_test, X_val, Y_test, Y_val = train_test_split(X_testAndVal,Y_testAndVal,test_size=0.5,random_state=23)

# 由于重新抽取训练集和测试集后原始的下标/索引会被打乱，因此需要重设索引。
for i in [X_train, X_test, Y_train, Y_test]:
    i.reset_index(drop=True,inplace=True)

X_train.head()

# 将处理过的pandas格式数据合并。
X_train['target'] = Y_train # 合并标签
X_test['target'] = Y_test
X_val['target'] = Y_val

X_train.head()


# 使用字典结构来代表决策树模型，输入数据以`pandas.DataFrame`格式存储。

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
                max_gain, attrVal = self.maxGainSplit(dataset,attr,root_entropy)
                # add 0.00001 in case there is 0
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
            # split into 2 subsets
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

# iris数据集中，4种属性都是连续值，因此用离散的方式（即ID3算法）会导致准确率偏低，存在过拟合现象。
# 因此引入C4.5算法，使用二分法给连续值分割。

# 使用matplotlib库绘制注解的方式来绘制决策树，此处借鉴@author: yangmqglobe的实现。绘制部分代码保存在`treeplotter.py`中。

print(test_tree_C45)


create_plot(test_tree_ID3)
create_plot(test_tree_C45)

# 由上述可视化结果可以发现，决策树的分支过多，可视化效果不佳。

print("ID3-Micro-F1: %f"% f1_score(list(X_test.iloc[:,-1].values),result_ID3,average="micro"))
print("C4.5-Micro-F1: %f"% f1_score(list(X_test.iloc[:,-1].values),result_C45,average="micro"))

print("ID3-Macro-F1: %f"% f1_score(list(X_test.iloc[:,-1].values),result_ID3,average="macro"))
print("C4.5-Macro-F1: %f"% f1_score(list(X_test.iloc[:,-1].values),result_C45,average="macro"))



