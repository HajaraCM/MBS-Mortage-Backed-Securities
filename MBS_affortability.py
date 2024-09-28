#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,classification_report,confusion_matrix
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


# <div style="text-align: center; background-color: orange; padding: 10px;">
#     <h2 style="font-weight: bold;">MONTHLY INSTALLMENT AFFORDABILITY</h2>
# </div>

# In[2]:


data_new=pd.read_csv('MSB-Mortgage-Backed-Securities-Pipeline-main-LoanExport.csv')


# In[3]:


data_new.head()


# In[4]:


data_new.drop(['EverDelinquent','Prepayment','SellerName'],inplace=True,axis=1)


# In[5]:



# Define a threshold for monthly installment affordability
affordability_threshold = 0.3 #installment should be less than 30% of monthly income

# Create the AffordabilityRisk binary indicator
data_new['AffordabilityRisk'] = ((data_new['MonthlyInstallment'] / data_new['MonthlyIncome']) > affordability_threshold).astype(int)  # 1 if installment is more than 30% of income, 0 otherwise


# In[6]:


from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
cat_col=['IsFirstTimeHomebuyer','PPM','PropertyState','ServicerName','PropertyType','Channel','LTV_Range','CreditRange','RePayRange']
data_new[cat_col]=data_new[cat_col].apply(le.fit_transform)
data_new.head()


# In[7]:


one_col=['LoanPurpose','Occupancy']
data_one=pd.get_dummies(data_new[one_col], drop_first=True)
data_one.head()


# In[8]:


data_new=pd.concat([data_new,data_one], axis = 1)
data_new.drop(['LoanPurpose','Occupancy'],inplace=True,axis=1)
data_new.head()


# In[9]:


import matplotlib.pyplot as plt

plt.hist(data_new['MonthlyInstallment'] / data_new['MonthlyIncome'], bins=50, edgecolor='k')
plt.axvline(x=affordability_threshold, color='r', linestyle='--', label=f'Threshold: {affordability_threshold}')
plt.xlabel('Installment to Income Ratio')
plt.ylabel('Frequency')
plt.title('Distribution of Installment to Income Ratio')
plt.legend()
plt.show()


# In[10]:


X= data_new.drop('AffordabilityRisk',axis=1)
y = data_new.AffordabilityRisk

import pandas as pd

# Assuming 'data' is your DataFrame and 'AffordabilityRisk' is your target variable

# Calculate correlation matrix
correlation_matrix = data_new.corr()

# Get the correlation of each feature with the target variable 'AffordabilityRisk'
affordability_corr = correlation_matrix['AffordabilityRisk'].abs()

# Get top 10 features excluding the target itself
top_features = affordability_corr.sort_values(ascending=False).index[1:11]

print("Top 10 Features based on correlation with Affordability:\n", top_features)


# In[11]:


X=X[top_features]


# In[12]:


X


# In[13]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[14]:


# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# In[15]:


# Train a model with the selected features
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = model.predict(X_test_scaled)

# Evaluate the model
print("Prepayment Risk Prediction Accuracy Score:", accuracy_score(y_test, y_pred))
print("Prepayment Risk Classification Report:\n", classification_report(y_test, y_pred))


# In[16]:


model_nb = GaussianNB()
model_nb.fit(X_train_scaled, y_train)

# Make predictions
y_pred = model_nb.predict(X_test_scaled)

# Evaluate the model
print("Prepayment Risk Prediction Accuracy Score:", accuracy_score(y_test, y_pred))
print("Prepayment Risk Classification Report:\n", classification_report(y_test, y_pred))


# In[17]:


rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled, y_train)

# Make predictions
y_pred= rf_model.predict(X_test_scaled)

# Evaluate the model
print("Prepayment Risk Prediction Accuracy Score:", accuracy_score(y_test, y_pred))
print("Prepayment Risk Classification Report:\n", classification_report(y_test, y_pred))


# In[18]:


from xgboost import XGBClassifier
xgb_model = XGBClassifier()
xgb_model.fit(X_train_scaled, y_train)

# Make predictions
y_pred= xgb_model.predict(X_test_scaled)

# Evaluate the model
print("Prepayment Risk Prediction Accuracy Score:", accuracy_score(y_test, y_pred))
print("Prepayment Risk Classification Report:\n", classification_report(y_test, y_pred))


# In[19]:


import joblib

# Save the pipeline
joblib.dump(model_nb, 'MBS_affordability_pipeline.pkl')
pipeline = joblib.load('MBS_affordability_pipeline.pkl')


# In[ ]:




