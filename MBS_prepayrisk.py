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
from sklearn.preprocessing import LabelEncoder


# In[2]:


data=pd.read_csv('MSB-Mortgage-Backed-Securities-Pipeline-main-LoanExport.csv')


# In[3]:


data.head()


# In[4]:


data.columns


# <div style="text-align: center; background-color: orange; padding: 10px;">
#     <h2 style="font-weight: bold;">ENCODING</h2>
# </div>

# In[5]:



data_pipe=pd.read_csv('MSB-Mortgage-Backed-Securities-Pipeline-main-LoanExport.csv')


# In[26]:


data_pipe.head()


# In[27]:


# Set the chosen threshold based on your decision
chosen_threshold =33984.0  # Example: 90th percentile

# Create the PrepaymentRisk binary indicator based on the chosen threshold
data_pipe['PrepaymentRisk'] = (data_pipe['Prepayment'] > chosen_threshold).astype(int)  # 1 if prepayment > threshold, 0 otherwise

# Display the distribution of the new PrepaymentRisk column
print(data_pipe['PrepaymentRisk'].value_counts())


# In[28]:


data_pipe.drop(['EverDelinquent','Prepayment','SellerName'],inplace=True,axis=1)


# In[29]:


X= data_pipe.drop('PrepaymentRisk', axis=1)
y = data_pipe.PrepaymentRisk


# In[30]:


# Feature selection (example top features, replace with actual top features)
top_features = ['MonthlyIncome', 'DTI', 'NumBorrowers', 'Channel', 'CurrentUPB', 'Totalpayment',
       'InterestAmount', 'MonthlyInstallment','LoanPurpose']


# In[31]:


X=X[top_features]


# In[32]:


from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

# Define a custom transformer for label encoding
class LabelEncoderTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns
        self.label_encoders = {}
    
    def fit(self, X, y=None):
        for col in self.columns:
            le = LabelEncoder()
            self.label_encoders[col] = le.fit(X[col])
        return self
    
    def transform(self, X):
        X = X.copy()
        for col, le in self.label_encoders.items():
            X[col] = le.transform(X[col])
        return X


# In[33]:




# Define the column transformer
preprocessor = ColumnTransformer(
    transformers=[
        ('onehot', OneHotEncoder(drop='first', sparse=False, handle_unknown='ignore'), ['LoanPurpose']),
        ('label', LabelEncoderTransformer(columns=['Channel']), ['Channel']),
        ('scaler', StandardScaler(), ['MonthlyIncome', 'DTI', 'NumBorrowers', 'CurrentUPB', 'Totalpayment', 'InterestAmount', 'MonthlyInstallment'])
    ],
    remainder='passthrough'
)

# Define the pipeline
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier',GaussianNB())  # You can replace this with any other classifier
])


# In[34]:


# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
pipeline.fit(X_train, y_train)

# Make predictions and evaluate
y_pred = pipeline.predict(X_test)
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))


# In[35]:



import joblib
# Save the pipeline
joblib.dump(pipeline, 'MBS_prepayrisk_pipeline.pkl')
pipeline = joblib.load('MBS_prepayrisk_pipeline.pkl')


# In[ ]:




