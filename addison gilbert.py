#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


path = "C:/Users/BLESS/Downloads/energydata_complete.csv"
df = pd.read_csv(path)


# In[3]:


df.head()


# ### From the dataset, fit a linear model on the relationship between the temperature in the living room in Celsius (x = T2) and the temperature outside the building (y = T6)
# 

# In[123]:


#df.drop(columns = ['Appliances'],axis = 1,inplace = True)


# In[112]:


#question12
from sklearn.linear_model import LinearRegression
LinearObj  = LinearRegression()


# In[113]:


X = df[["T2"]]
Y = df["T6"]
LinearObj.fit(X,Y)


# In[114]:


Yhat = LinearObj.predict(X)
Yhat[0:5]


# In[115]:


#12. this answer is correct
r2_value = LinearObj.score(X,Y)
round(r2_value,2)


# In[116]:


from sklearn.metrics import r2_score
r2_score = r2_score(Y, Yhat)
round(r2_score, 2)


# In[117]:


#Q13
from sklearn.metrics import mean_absolute_error
mae = mean_absolute_error(Y,Yhat)
round(mae, 2)


# In[118]:


#question 14
Residual_Sum_Squares = np.sum(np.square(Y - Yhat))
round(Residual_Sum_Squares, 2)


# In[119]:


#question 15
#finding Root Mean Squared Error
from sklearn.metrics import  mean_squared_error
Root_Mean_Squared_Error = np.sqrt(mean_squared_error(Y, Yhat))
round(Root_Mean_Squared_Error, 3)


# In[120]:


#Q16
Coefficient_of_Determination=linear_model2.score(x_train,y_train)
round(Coefficient_of_Determination,2)


# In[121]:


def get_weights_df(model, fit, col_name):
    
    #this function returns the weight of every feature
    weights = pd.Series(model.coef_, fit.columns).sort_values()
    weights_df = pd.DataFrame(weights).reset_index()
    weights_df.columns = ['Features', col_name]
    weights_df[col_name].round(3)
    return weights_df


# In[122]:


linear_model_weights = get_weights_df(linear_model2, x_train, 'Linear_Model_Weight')
linear_model_weights


# In[110]:


#Q17
print("highest weight {}".format(linear_model_weights.max()))
print("lowest weight {}".format(linear_model_weights.min()))


# In[127]:


#question 18
from sklearn.linear_model import Ridge
ridge_reg = Ridge(alpha=0.4)
ridge_reg.fit(X,Y)


# In[131]:


pred_test_rr=ridge_reg.predict(X)
print(np.sqrt(mean_squared_error(Y,pred_test_rr))) 


# In[129]:


Lasso_model_weights = get_weights_df(lasso_reg , x_train, 'Lasso_Model_Weight')
Lasso_model_weights


# In[126]:


#20
pred_test_lasso= lasso_reg.predict(x_test)
print(np.sqrt(mean_squared_error(y_test,pred_test_lasso))) 


# In[ ]:





# In[4]:


# droping light and date columns
df.drop(columns = ['date','lights'],axis = 1,inplace = True)
df.head()


# In[ ]:





# In[83]:


#normalizing the dataset to a common scale using the minMax scaler
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
#normalizing the features
normalised_df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
predictor_columns = normalised_df.drop(columns=['T6'])
target_column = normalised_df['T6']


# In[84]:


from sklearn.model_selection import train_test_split
x_train,x_test, y_train, y_test = train_test_split(predictor_columns,target_column, test_size=0.3, random_state=42)


# In[87]:


linear_model2 = LinearRegression()
#fit the model to the training dataset
linear_model2.fit(x_train, y_train)
#obtain predictions
predicted_values = linear_model2.predict(x_test)


# In[88]:


#question 13
from sklearn.metrics import mean_absolute_error
mae = mean_absolute_error(y_test, predicted_values)
round(mae, 2)


# In[89]:


#question 14
Residual_Sum_Squares = np.sum(np.square(y_test - predicted_values))
round(Residual_Sum_Squares, 2)


# In[90]:


#question 15
#finding Root Mean Squared Error
from sklearn.metrics import  mean_squared_error
Root_Mean_Squared_Error = np.sqrt(mean_squared_error(y_test, predicted_values))
round(Root_Mean_Squared_Error, 3)


# In[91]:


#question16
Coefficient_of_Determination=linear_model2.score(x_train,y_train)


# In[92]:


round(Coefficient_of_Determination,2)


# In[93]:


def get_weights_df(model, fit, col_name):
    
    #this function returns the weight of every feature
    weights = pd.Series(model.coef_, fit.columns).sort_values()
    weights_df = pd.DataFrame(weights).reset_index()
    weights_df.columns = ['Features', col_name]
    weights_df[col_name].round(3)
    return weights_df


# In[94]:


linear_model_weights = get_weights_df(linear_model2, x_train, 'Linear_Model_Weight')


# In[95]:


linear_model_weights


# In[96]:


print("highest weight {}".format(linear_model_weights.max()))
print("lowest weight {}".format(linear_model_weights.min())) 


# In[97]:


#question 18
from sklearn.linear_model import Ridge
ridge_reg = Ridge(alpha=0.4)
ridge_reg.fit(x_train, y_train)


# In[98]:


pred_test_rr=ridge_reg.predict(x_test)
print(np.sqrt(mean_squared_error(y_test,pred_test_rr))) 


# In[99]:


#question19
from sklearn.linear_model import Lasso
lasso_reg = Lasso(alpha=0.001)
lasso_reg.fit(x_train, y_train)


# In[100]:


Lasso_model_weights = get_weights_df(lasso_reg , x_train, 'Lasso_Model_Weight')


# In[101]:


Lasso_model_weights


# In[102]:


Lasso_model_weights.min()


# In[103]:


pred_test_lasso= lasso_reg.predict(x_test)
print(np.sqrt(mean_squared_error(y_test,pred_test_lasso))) 


# In[ ]:





# In[ ]:





# In[ ]:




