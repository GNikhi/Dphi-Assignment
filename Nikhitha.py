#!/usr/bin/env python
# coding: utf-8

# In[63]:


import pandas as pd
import numpy as np

loan_data  = pd.read_csv("https://raw.githubusercontent.com/dphi-official/Datasets/master/Loan_Data/loan_train.csv" )


# In[64]:


loan_data.describe()


# In[65]:


loan_data.head()


# In[67]:


loan_data.drop(['Loan_ID'], axis = 1, inplace = True )


# In[68]:


loan_data.rename({"Unnamed: 0":"a"}, axis="columns", inplace=True)
loan_data.drop(["a"], axis=1, inplace=True)


# In[69]:


loan_data.head()


# In[70]:


loan_data.info()


# In[71]:


num_cols = loan_data.select_dtypes(include=np.number).columns
cat_cols = loan_data.select_dtypes(include = 'object').columns
loan_data[num_cols] = loan_data[num_cols].fillna(loan_data[num_cols].mean()) 
loan_data[cat_cols] = loan_data[cat_cols].fillna(loan_data[cat_cols].mode().iloc[0])


# In[72]:


loan_data.isnull().sum() / len(loan_data) * 100


# In[73]:


loan_data.shape
loan_data = pd.get_dummies(loan_data, columns=cat_cols)
loan_data.shape
loan_data.head()


# In[74]:


loan_data.head()


# In[75]:


loan_data['Loan_Status'].value_counts()  


# In[76]:


X = loan_data.drop(columns = ['Loan_Status'])       
Y = loan_data.Loan_Status


# In[77]:


loan_data.info()


# In[78]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3, random_state = 42)


# In[79]:


from sklearn.ensemble import RandomForestClassifier   
rfc = RandomForestClassifier()

rfc.fit(X_train, Y_train)   


# In[80]:


def print_results(results):
    print('BEST PARAMS: {}\n'.format(results.best_params_))

    means = results.cv_results_['mean_test_score']
    stds = results.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, results.cv_results_['params']):
        print('{} (+/-{}) for {}'.format(round(mean, 3), round(std * 2, 3), params))


# In[81]:


from sklearn.model_selection import GridSearchCV
rfc = RandomForestClassifier()
parameters = {
    'n_estimators': [80, 90, 100, 125, 150],
    'max_depth': [2,3,4,5,8,16,None]
}
cv = GridSearchCV(rfc, parameters, cv=5)
cv.fit(X_train, Y_train)

print_results(cv)


# In[82]:


cv.best_score_


# In[83]:


print(cv.best_estimator_)


# In[84]:


test_data = pd.read_csv('https://raw.githubusercontent.com/dphi-official/Datasets/master/Loan_Data/loan_test.csv')


# In[88]:


test_data.info()


# In[89]:


test_data.head()
test_data.drop(['Loan_ID'], axis = 1, inplace = True)


# In[90]:


num_cols1 = test_data.select_dtypes(include=np.number).columns
cat_cols1 = test_data.select_dtypes(include = 'object').columns
test_data[num_cols1] = test_data[num_cols1].fillna(test_data[num_cols1].mean()) 
test_data[cat_cols1] = test_data[cat_cols1].fillna(test_data[cat_cols1].mode().iloc[0])


# In[91]:


test_data.info()


# In[92]:


test_data = pd.get_dummies(test_data, columns=cat_cols)


# In[93]:


missing_levels_cols= list(set(loan_data.columns) - set(test_data.columns))
print(len(missing_levels_cols))
for c in missing_levels_cols:
    test_data[c]=0

# Select only those columns which are there in training data
test_data=test_data[loan_data.columns]


# In[94]:


from sklearn import preprocessing

final_ts = pd.DataFrame(data=test_data)
final_ts.columns= test_data.columns
print(final_ts.head())
print(final_ts.shape)

final_ts


# In[95]:


test_data.info()


# In[96]:


final_ts.drop(columns = ['Loan_Status'], inplace = True)


# In[97]:


print(test_data.columns)


# In[98]:


print(loan_data.columns)


# In[99]:


predictions = cv.predict(final_ts)


# In[100]:


print(predictions)


# In[101]:


res = pd.DataFrame(predictions) #preditcions are nothing but the final predictions of your model on input features of your new unseen test data
res.index = final_ts.index # its important for comparison. Here "test_new" is your new test dataset
res.columns = ["prediction"]
res.to_csv("prediction_results.csv", index = False)      # the csv file will be saved locally on the same location where this notebook is located.


# In[ ]:




