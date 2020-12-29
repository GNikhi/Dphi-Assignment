#!/usr/bin/env python
# coding: utf-8

# In[171]:


import pandas as pd
import numpy as np

loan_data  = pd.read_csv("https://raw.githubusercontent.com/dphi-official/Datasets/master/Loan_Data/loan_train.csv" )


# In[172]:


loan_data.describe()


# In[173]:


loan_data.head()


# In[174]:


loan_data.rename({"Unnamed: 0":"a"}, axis="columns", inplace=True)
loan_data.drop(["a"], axis=1, inplace=True)


# In[175]:


loan_data.head()


# In[176]:


loan_data.info()


# In[177]:


num_cols = loan_data.select_dtypes(include=np.number).columns
cat_cols = loan_data.select_dtypes(include = 'object').columns
loan_data[num_cols] = loan_data[num_cols].fillna(loan_data[num_cols].mean()) 
loan_data[cat_cols] = loan_data[cat_cols].fillna(loan_data[cat_cols].mode().iloc[0])


# In[178]:


loan_data.isnull().sum() / len(loan_data) * 100


# In[179]:


loan_data.shape
loan_data = pd.get_dummies(loan_data, columns=cat_cols)
loan_data.shape
loan_data.head()


# In[180]:


loan_data.head()


# In[181]:


loan_data['Loan_Status'].value_counts()  


# In[182]:


X = loan_data.drop(columns = ['Loan_Status'])       
Y = loan_data.Loan_Status


# In[183]:


loan_data.info()


# In[207]:


from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3, random_state = 42)


# In[208]:


from sklearn.ensemble import GradientBoostingClassifier

model = GradientBoostingClassifier()
model.fit(X_train, Y_train)


# In[209]:


def print_results(results):
    print('BEST PARAMS: {}\n'.format(results.best_params_))

    means = results.cv_results_['mean_test_score']
    stds = results.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, results.cv_results_['params']):
        print('{} (+/-{}) for {}'.format(round(mean, 3), round(std * 2, 3), params))


# In[210]:


from sklearn.model_selection import GridSearchCV
gbc = GradientBoostingClassifier()
parameters = {
    'n_estimators': [80, 90, 100, 125, 150],
    'max_depth': [2,3,4,5,8,16,None],
    'learning_rate': [0.03, 0.1, 0.3, 0.5]
}
cv = GridSearchCV(gbc, parameters, cv=5)
cv.fit(X_train, Y_train)

print_results(cv)


# In[212]:


cv.best_score_


# In[213]:


print(cv.best_estimator_)


# In[214]:


test_data = pd.read_csv('https://raw.githubusercontent.com/dphi-official/Datasets/master/Loan_Data/loan_test.csv')


# In[215]:


test_data.describe()


# In[216]:


test_data.info()


# In[217]:


test_data.head()


# In[218]:


num_cols1 = test_data.select_dtypes(include=np.number).columns
cat_cols1 = test_data.select_dtypes(include = 'object').columns
test_data[num_cols1] = test_data[num_cols1].fillna(test_data[num_cols1].mean()) 
test_data[cat_cols1] = test_data[cat_cols1].fillna(test_data[cat_cols1].mode().iloc[0])


# In[219]:


test_data.info()


# In[220]:


test_data = pd.get_dummies(test_data, columns=cat_cols)


# In[221]:


missing_levels_cols= list(set(loan_data.columns) - set(test_data.columns))
print(len(missing_levels_cols))
for c in missing_levels_cols:
    test_data[c]=0

# Select only those columns which are there in training data
test_data=test_data[loan_data.columns]


# In[222]:


from sklearn import preprocessing

final_ts = pd.DataFrame(data=test_data)
final_ts.columns= test_data.columns
print(final_ts.head())
print(final_ts.shape)

final_ts


# In[223]:


test_data.info()


# In[224]:


final_ts.drop(columns = ['Loan_Status'], inplace = True)


# In[227]:


predictions = cv.predict(final_ts)


# In[228]:


print(predictions)


# In[229]:


res = pd.DataFrame(predictions) #preditcions are nothing but the final predictions of your model on input features of your new unseen test data
res.index = final_ts.index # its important for comparison. Here "test_new" is your new test dataset
res.columns = ["prediction"]
res.to_csv("prediction_results.csv", index = False)      # the csv file will be saved locally on the same location where this notebook is located.


# In[ ]:




