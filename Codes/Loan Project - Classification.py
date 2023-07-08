#!/usr/bin/env python
# coding: utf-8

# # Loan Project - Classification

# ***
# **Importing the required libraries & packages**

# In[1]:


import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import pyodbc
from math import sqrt
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score,mean_absolute_error,mean_squared_error,confusion_matrix,classification_report
import pickle
import warnings
warnings.filterwarnings('ignore')


# **Changing The Default Working Directory Path**

# In[2]:


os.chdir('C:\\Users\\Shridhar\\Desktop\\Loan Project')


# ## Data Reading:

# **Processed the input data using Structured Query Language _(SQL)_ and done some cleaning, with the help of <span style = 'background : green'><span style = 'color : white'> pyodbc </span> </span> package connecting Jupyter Notebook with SQL Server in the following 3 cells.**

# In[3]:


server = 'SHRIDHAR\SQLEXPRESS'
db = 'LoanProject'


# In[4]:


conn = pyodbc.connect('DRIVER={SQL Server};SERVER='+server+';DATABASE='+db+';UID=Shri; PWD=12345678;Trusted connection=YES')


# In[5]:


sql = 'select * from acc_ord_card_disp_client_dist aocdcd join loan_trans lt on lt.account_id= aocdcd.disposition_account_id'


# **Reading the SQL File using Pandas Command**

# In[6]:


df = pd.read_sql(sql,conn)


# **Exporting the Data after processing in SQL, the final data is converted to Comma Seperated Values _(CSV)_ File**

# In[7]:


df.to_csv('Loan Final Data.csv',index = False)


# ## Exploratory Data Analysis:

# **Checking the Null values of all the columns in the dataset.**

# In[8]:


df.isna().sum()


# **Checking the dataset whether it's having duplicate values or not**

# In[9]:


df.duplicated().sum()


# **Getting to describe the numerical columns of the dataset**

# In[10]:


df.describe()


# **Finding the shape of the dataset**

# In[11]:


df.shape


# **Since it has many columns, to extract the columns we need getting all the column names. So, that we can identify the necessary column**

# In[12]:


df.columns


# ## Data Cleaning:

# **Checking the value counts for the `loan_status` column from the dataset.**

# In[13]:


df['loan_status'].value_counts()


# **Label Encoding the `loan_status` column using mapping function**

# In[14]:


df['loan_status']=df['loan_status'].map({'A' : 0, 'B' : 1, 'C' : 2, 'D' : 3})


# **After Label Encoding , checking the values to verify there's no missing arguments in the `loan_status` column**

# In[15]:


df.groupby('loan_status').size()


# ## Data Visualization:

# **Plotting the bar graph with `loan_status` and `loan_duration` and saving the png file**

# In[16]:


plt.rcParams['figure.figsize']=20,10
sns.barplot(x= 'loan_status', y = 'loan_duration' ,data = df, ci = None)
plt.title('Loan Status vs Loan Duration')
plt.savefig('Loan Status vs Loan Duration.png')
plt.show()


# **Plotting the bar graph with `loan_status` and `loan_amount` and saving the png file**

# In[17]:


sns.barplot(x= 'loan_status', y = 'loan_amount' ,data = df, ci = None)
plt.title('Loan Status vs Loan Amount')
plt.savefig('Loan Status vs Loan Amount.png')
plt.show()


# **Plotting the Bar Graph with count of customers based on the `loan_duration` and confirm that there are no null values and identify all unique values from the `loan_duration` and saving the PNG File**

# In[18]:


Duration = df['loan_duration'].value_counts()
plot = sns.barplot(x = Duration.index, y = Duration.values, data = df)
for p in plot.patches:
    plot.annotate(p.get_height(),(p.get_x() + p.get_width() / 2.0,p.get_height()),
                 ha = 'center',va = 'center',xytext = (0,5),textcoords = 'offset points')
plt.title('Loan Duration')
plt.xlabel('Duration')
plt.ylabel('Counts')
plt.savefig('Loan Duration.png')
plt.show()


# **Getting the Correlation Values from the needed columns from the dataset using Seaborn Heatmap & saving the PNG File**

# In[19]:


cor = df.iloc[:,46:50].corr()
sns.heatmap(cor, cmap = 'viridis', cbar = True, annot = True, square = True)
plt.title('Correlation Heat Map')
plt.savefig('Correlation Heat Map.png')
plt.show()


# **Assigning the dependent and independent variable**

# In[20]:


x = df.iloc[:,46:49]
y = df.iloc[:,49]


# ## Data Preprocessing:

# **Standardizing the independent variable of the dataset**

# In[21]:


sc = StandardScaler()
x = sc.fit_transform(x)


# ## Model Fitting:

# **Defining the Function for the ML algorithms using GridSearchCV Algorithm and Predicting the Dependent Variable by fitting the given model and create the pickle file of the model with the given Algo_name. Further getting the Algorithm Name, Best Parameters of the algorithm, Accuracy Score in percentage format, Mean Absolute error, Mean Squared Error, Root Mean Squared error, Confusion Matrix and Classification Report between the predicted values and dependent test dataset**

# In[22]:


def FitModel(x,y,algo_name,algorithm,GridSearchParams,cv):
    np.random.seed(10)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 51)
    grid = GridSearchCV(estimator = algorithm, param_grid = GridSearchParams, cv = cv,
                       scoring = 'accuracy', verbose = 0,n_jobs = -1)
    grid_result = grid.fit(x_train,y_train)
    pred = grid_result.predict(x_test)
    best_params = grid_result.best_params_
    pickle.dump(grid_result,open(algo_name,'wb'))
    print('Algorithm Name : ',algo_name,'\n')
    print('Best Params : ',best_params,'\n')
    print('Percentage of Accuracy Score : {} %'.format(100 * accuracy_score(y_test,pred)),'\n')
    print('Mean Absolute Error : ',mean_absolute_error(y_test,pred),'\n')
    print('Mean Squared Error : ',mean_squared_error(y_test,pred),'\n')
    print('Root Mean Squared Error : ',sqrt(mean_squared_error(y_test,pred)),'\n')
    print('Confusion Matrix : \n',confusion_matrix(y_test,pred),'\n')
    print('Classification Report : \n',classification_report(y_test,pred))


# **Running the function with empty parameters since the Logistic Regression model doesn't need any special parameters and fitting the Logistic Regression Algorithm and getting the Algorithm Name, Best Parameters of the algorithm, Accuracy Score in percentage format, Mean Absolute error, Mean Squared Error, Root Mean Squared error, Confusion Matrix and Classification Report between the predicted values and dependent test dataset and also the pickle file with the name Logistic Regression.**

# In[23]:


param = {}
FitModel(x, y, 'Logistic Regression', LogisticRegression(), param, cv = 10)


# **Running the function with some appropriate parameters and fitting the Decision Tree Classifier Algorithm and getting the Algorithm Name, Best Parameters of the algorithm, Accuracy Score in percentage format, Mean Absolute error, Mean Squared Error, Root Mean Squared error, Confusion Matrix and Classification Report between the predicted values and dependent test dataset and also the pickle file with the name Decision Tree.**

# In[24]:


params = {'criterion' :['entropy','gini'],
          'max_depth' :[1,2,3,4],
          'max_features' :['auto','sqrt'],
          'min_samples_split' :[5,6,7,8],
          'min_samples_leaf' :[9,10,11,12]}
FitModel(x, y, 'Decision Tree', DecisionTreeClassifier(), params, cv =10)


# **Running the function with some appropriate parameters and fitting the Random Forest Classifier Algorithm and getting the Algorithm Name, Best Parameters of the algorithm, Accuracy Score in percentage format, Mean Absolute error, Mean Squared Error, Root Mean Squared error, Confusion Matrix and Classification Report between the predicted values and dependent test dataset and also the pickle file with the name Random Forest.**

# In[25]:


params = {'n_estimators' :[111,222,333,444],
          'criterion' :['entropy','gini'],
          'max_depth': [4], 
          'max_features': ['auto'], 
          'min_samples_leaf': [11], 
          'min_samples_split': [6]}
FitModel(x, y, 'Random Forest', RandomForestClassifier(), params, cv =10)


# **Running the function with some appropriate parameters and fitting the KNeighbors Classifier Algorithm and getting the Algorithm Name, Best Parameters of the algorithm, Accuracy Score in percentage format, Mean Absolute error, Mean Squared Error, Root Mean Squared error, Confusion Matrix and Classification Report between the predicted values and dependent test dataset and also the pickle file with the name KNeighbors.**

# In[26]:


params = {'n_neighbors' :[3,5,7,10,13]}
FitModel(x, y, 'KNeighbors', KNeighborsClassifier(), params, cv =10)


# **Running the function with some appropriate parameters and fitting the Support Vector Machine Classifier Algorithm and getting the Algorithm Name, Best Parameters of the algorithm, Accuracy Score in percentage format, Mean Absolute error, Mean Squared Error, Root Mean Squared error, Confusion Matrix and Classification Report between the predicted values and dependent test dataset and also the pickle file with the name SVC.**

# In[27]:


params = {'C' : [0.1,1,100,1000],
          'gamma' :[0.001,0.01,0.1,1]}
FitModel (x, y,'SVC', SVC(), params, cv =10)


# **Running the function with some appropriate parameters and fitting the XGBoost Classifier Algorithm and getting the Algorithm Name, Best Parameters of the algorithm, Accuracy Score in percentage format, Mean Absolute error, Mean Squared Error, Root Mean Squared error, Confusion Matrix and Classification Report between the predicted values and dependent test dataset and also the pickle file with the name XGBoost.**

# In[28]:


params = {'n_estimators' :[111,222,333,444]}
FitModel (x, y,'XGBoost', XGBClassifier(), params, cv = 10)


# **Running the function with some appropriate parameters and fitting the CatBoost Classifier Algorithm and getting the Algorithm Name, Best Parameters of the algorithm, Accuracy Score in percentage format, Mean Absolute error, Mean Squared Error, Root Mean Squared error, Confusion Matrix and Classification Report between the predicted values and dependent test dataset and also the pickle file with the name CatBoost.**

# In[29]:


params = {'verbose' :[0]}
FitModel(x, y, 'CatBoost', CatBoostClassifier(), params , cv = 10)


# **Running the function with empty parameters since the Light GBM Classifier model doesn't need any special parameters and fitting the Light GBM Classifier Algorithm and getting the Algorithm Name, Best Parameters of the algorithm, Accuracy Score in percentage format, Mean Absolute error, Mean Squared Error, Root Mean Squared error, Confusion Matrix and Classification Report between the predicted values and dependent test dataset and also the pickle file with the name Light GBM.**

# In[30]:


param = {}
FitModel(x, y, 'Light GBM', LGBMClassifier(), param, cv=10)


# ## Model Testing: 

# **Loading the pickle file with the algorithm which gives highest accuracy score**

# In[31]:


model = pickle.load(open('XGBoost','rb'))


# **Predicting the dependent variable of the whole data using the loaded pickle file and getting the Accuracy Score in percentage format, Mean Absolute error, Mean Squared Error, Root Mean Squared error, Confusion Matrix and Classification Report between the predicted values and dependent dataset**

# In[32]:


fpred = model.predict(x)
print('Percentage of Accuracy Score : {} %'.format(100 * accuracy_score(y,fpred)),'\n')
print('Mean Absolute Error : ',mean_absolute_error(y,fpred),'\n')
print('Mean Squared Error : ',mean_squared_error(y,fpred),'\n')
print('Root Mean Squared Error : ',sqrt(mean_squared_error(y,fpred)),'\n')
print('Confusion Matrix : \n',confusion_matrix(y,fpred),'\n')
print('Classification Report : \n',classification_report(y,fpred))


# **Making the Predicted value as a new dataframe with new column name as `Predicted Loan Status (Approx.)` and concating it with the original data, so that we can able to compare the differences between Predicted Loan Status and Original Loan Status.**

# In[33]:


prediction = pd.DataFrame(fpred,columns = ['Predicted Loan Status(Approx.)'])
pred_df = pd.concat([df,prediction],axis = 1)


# **Exporting the data with Predicted Loan Status to a Comma Seperated Value _(CSV)_ file**

# In[34]:


pred_df.to_csv('Predicted Loan Status Data.csv',index = False)

