#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import required modules
import itertools
import numpy as np
import matplotlib.pylab as plt
from matplotlib.ticker import NullFormatter
import pandas as pd
from sklearn import preprocessing

# read csv file which is our dataset
df = pd.read_csv('diabetes.csv')
df.head()


# In[2]:


# print out shape of dataset
df.shape


# In[3]:


# remove Outcome column from the dataset
x = df.drop(columns = ['Outcome'])

# print out first 5 raw from dataset
x.head()


# In[4]:


# Print target values
y = df['Outcome'].values
y[0:5]


# In[5]:


from sklearn.model_selection import train_test_split
# create train and test dataset
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1, stratify=y)


# In[6]:


from sklearn.neighbors import KNeighborsClassifier as KNN
knn = KNN(n_neighbors=3)
knn.fit(x_train, y_train)


# In[7]:


knn.predict(x_test)[0:5]


# In[8]:


knn.score(x_test, y_test)


# In[9]:


from sklearn.model_selection import cross_val_score

# cv is cross-validation
knn_cv = KNN(n_neighbors=3)
# train model with cv of 5
cv_scores = cross_val_score(knn_cv, x, y, cv=5)
print(cv_scores)
print("cv scores mean: {}".format(np.mean(cv_scores)))


# In[10]:


from sklearn.model_selection import GridSearchCV as gcv

# create new knn model
knn2 = KNN()
# create a dictionary of all values we want to test for n_neighbors
param_grid = {'n_neighbors': np.arange(1, 25)}
# use gridsearch to test all values for n_neighbors 
knn_gscv = gcv(knn2, param_grid, cv=5)
# fit model to data 
knn_gscv.fit(x, y)
# check top performing n_neighbors value
print("Best Parameter of this nearest neighbor is: {}".format(knn_gscv.best_params_))
# check mean score for the top performing value of n_neighbors
print("Avarage Score of This model is: {}".format(knn_gscv.best_score_))

