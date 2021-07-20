#!/usr/bin/env python
# coding: utf-8

# In[1]:


import dask_ml
import pandas as pd
import os, uuid
import json
from sklearn.ensemble import RandomForestClassifier
import dask
import dask.dataframe as dd
from dask_ml.preprocessing import DummyEncoder
from dask import dataframe as dd 
import pickle
# In[3]:


main_df = dd.read_csv('test_data.csv')
columns_req=pd.read_csv("column_list_new.csv")
#main_dff = dd.from_pandas(main_df, npartitions=7)


# In[4]:


df=main_df[columns_req["Variable_list"]]


# In[5]:


no_columns=len(columns_req.index)


# In[6]:





# In[7]:


x=df.iloc[:,0:no_columns-1]


# In[8]:


x=x.categorize()


# In[9]:


de = DummyEncoder()
X_test = de.fit_transform(x)


# In[10]:



Pkl_Filename = "HumanEvent_Model.pkl"  

with open(Pkl_Filename, 'rb') as file:  
    clf = pickle.load(file)


# In[11]:


pred_test=clf.predict(X_test)


# In[12]:


pred_test_df=dd.from_array(pred_test)


# In[13]:


pred_test_df=pred_test_df.to_frame()


# In[14]:


prediction=x.merge(pred_test_df)


# In[16]:





# In[17]:



prediction.to_csv("prediction_new.csv", single_file = True)


print("End!!")
