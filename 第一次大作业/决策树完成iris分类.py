#!/usr/bin/env python
# coding: utf-8

# 决策树分类

# In[1]:


#-*- coding: utf-8 -*-
from sklearn import datasets#导入方法类

iris = datasets.load_iris()#加载iris数据集
iris_feature = iris.data#特征数据
iris_target = iris.target#分类数据
iris_target#查看 iris_target


# In[2]:


from sklearn.model_selection import train_test_split
feature_train,feature_test,target_train,target_test = train_test_split(iris_feature,iris_target,test_size=0.33,random_state=42)
target_train


# 其中feature_train,feature_test,target_train,target_test分别代表训练集特征，测试集特征，训练集目标值，验证集特征，test_size参数代表划分到测试集数据占全部数据的百分比，一般情况下，将整个训练集划分为70%训练集和30%测试集，最后的random_state参数表示乱序程度。

# In[4]:


#-*- coding:utf-8 -*-
from sklearn.tree import DecisionTreeClassifier
dt_model = DecisionTreeClassifier()#所有参数均置为默认状态
dt_model.fit(feature_train,target_train)#使用训练集训练模型
predict_results = dt_model.predict(feature_test)#使用模型对测试集进行预测


# 从scikit_learn中导入决策树分类器，然后实验fit方法和predict方法对模型进行训练和预测

# In[5]:


print('predict_results:',predict_results)
print('target_test:',target_test)


# In[6]:


from sklearn.metrics import accuracy_score
print(accuracy_score(predict_results,target_test))


# 最终预测准确度为98%。
