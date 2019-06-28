#!/usr/bin/env python
# coding: utf-8

# In[8]:


import sys
import numpy
import matplotlib
import pandas
import sklearn

print('python: '+ sys.version)
print('numpy: ' + numpy.__version__)
print('matplotlib: '+ matplotlib.__version__)
print('pandas: '+ pandas.__version__)
print('sklearn: '+ sklearn.__version__)


# In[10]:


import numpy as np
from sklearn import preprocessing, model_selection
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn import model_selection
from sklearn.metrics import classification_report, accuracy_score
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
import pandas as pd


# In[11]:


# load the data
url="https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data"
names=['id','clump_thickness','uniform_cell_size','uniform_cell_shape','marginal_adhesion','single_epithelial_size',
       'bare_nuclei','bland_chromatin','normal_nucleoli','mitoses','class']
df = pd.read_csv(url, names=names)


# In[12]:


# preprocess the data
df.replace('?',-99999, inplace=True)
print(df.axes)
df.drop(['id'], 1, inplace=True)
# print shape of the data
print(df.shape)


# In[13]:


# do data visualization
print(df.describe())


# In[14]:


df.hist(figsize = (10,10))
plt.show()


# In[15]:


scatter_matrix(df, figsize=(18,18))
plt.show()


# In[17]:


# create x and y dataset for training
x = np.array(df.drop(['class'], 1))
y = np.array(df['class'])

x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size=0.2)

# specify testing opinions
seed = 8
scoring = 'accuracy'

# define the model to train
models = []
models.append(('KNN', KNeighborsClassifier(n_neighbors=5)))
models.append(('SVM', SVC(gamma='auto')))

# evaluate each model in turn
results = []
names = []
for name, model in models:
    kfold = model_selection.KFold(n_splits=10, random_state=seed)
    cv_results = model_selection.cross_val_score(model, x_train, y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)


# In[18]:


# make prediction based on validation dataset
for name, model in models:
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    print(name)
    print(accuracy_score(y_test, predictions))
    print(classification_report(y_test, predictions))


# In[20]:


clf = SVC(gamma='auto')

clf.fit(x_train, y_train)
accuracy = clf.score(x_test, y_test)
print(accuracy)

example = np.array([[4,2,1,1,1,2,3,2,4]])
example = example.reshape(len(example), -1)
prediction = clf.predict(example)
print(prediction)


# In[ ]:




