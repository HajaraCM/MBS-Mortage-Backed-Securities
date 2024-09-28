

# In[1]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import  RandomForestRegressor
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.naive_bayes import GaussianNB



# In[2]:


data = pd.read_csv(r'C:\Users\HP\Desktop\MBS\MSB-Mortgage-Backed-Securities-Pipeline-main-LoanExport-Revised.csv')


# In[3]:


data.head()


# In[4]:


# monthly interest rate
data['OrigInterestRate_Monthly'] =  np.round((data['OrigInterestRate'] / 12) / 100, 4)

# monthly installment
def calculateEmi(principal, monthly_interest_rate, loan_term_months):
    numerator = (1 + monthly_interest_rate) ** loan_term_months
    denominator = numerator - 1
    interest = numerator / denominator
    emi = principal * monthly_interest_rate * interest
    return np.int64(emi)

data['MonthlyInstallment'] = data.apply(
        lambda features: calculateEmi(
            principal=features['OrigUPB'], 
            monthly_interest_rate=features['OrigInterestRate_Monthly'],
            loan_term_months=features['OrigLoanTerm']), axis=1)

# current unpaid principal

def get_currentUPB(principal, monthly_interest_rate, monthly_installment,
                   payments_made):
    monthly_interest = monthly_interest_rate * principal
    monthly_paid_principal = monthly_installment - monthly_interest
    unpaid_principal = principal - (monthly_paid_principal * payments_made)
    return np.int32(unpaid_principal)

data['CurrentUPB'] = data.apply(
        lambda features: get_currentUPB(
            monthly_interest_rate=features['OrigInterestRate_Monthly'],
            principal=features['OrigUPB'], 
            monthly_installment=features['MonthlyInstallment'],
            payments_made=features['MonthsInRepayment']), axis=1)

# monthly income
def calculate_monthly_income(dti, emi):
    dti = dti if dti <1 else dti / 100
    # Calculate montly income
    if dti == 0:
        monthly_income = emi
    else:
        monthly_income = emi / dti
    return np.int64 (monthly_income)

data['MonthlyIncome'] = data.apply(
        lambda features: calculate_monthly_income(
            dti = features['DTI'],
            emi= features['MonthlyInstallment']), axis=1)

# prepayment
def calculatePrepayment(dti, monthly_income):
    if (dti < 40):
        prepayment = monthly_income / 2;
    else:
        prepayment = monthly_income * 3 / 4;
    return np.int64(prepayment)

data['Prepayment'] = data.apply(
        lambda features: calculatePrepayment(
            dti=features['DTI'],
            monthly_income=features['MonthlyIncome']), axis=1)
data['Prepayment']=(data['Prepayment']*24)-(data['MonthlyInstallment']*24)

# total payment and interest amount
data['Totalpayment'] = data['MonthlyInstallment'] * data['OrigLoanTerm']
data['InterestAmount'] = data['Totalpayment'] - data['OrigUPB']


# In[5]:


from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
cat_col=['IsFirstTimeHomebuyer','PPM','PropertyState','ServicerName','PropertyType','Channel','SellerName','LTV_Range','CreditRange','RePayRange']
data[cat_col]=data[cat_col].apply(le.fit_transform)
data.head()


# In[6]:


one_col=['LoanPurpose','Occupancy']
data_one=pd.get_dummies(data[one_col], drop_first=True)
data_one.head()


# In[7]:


data=pd.concat([data,data_one], axis = 1)
data.drop(['LoanPurpose','Occupancy'],inplace=True,axis=1)
data.head()


# In[8]:


# Split data into features and target
X = data.drop(['EverDelinquent', 'Prepayment'], axis=1)
y_class = data['EverDelinquent']
y_reg = data['Prepayment']

# Split into training and testing sets
X_train, X_test, y_class_train, y_class_test, y_reg_train, y_reg_test = train_test_split(
    X, y_class, y_reg, test_size=0.2, random_state=42
)


# In[9]:


# importing library for mi score for classification 
from sklearn.feature_selection import mutual_info_classif
# determine the mutual information
mutual_info = mutual_info_classif(X_train, y_class_train)
mutual_info


# In[10]:


mutual_info = pd.Series(mutual_info)
mutual_info.index = X.columns
mutual_info.sort_values(ascending=False)


# In[11]:


# For selecting best feature based on mi score
from sklearn.feature_selection import SelectKBest
#Now we Will select the  top 10 important features
selector= SelectKBest(mutual_info_classif, k=10)
x_train_selected=selector.fit_transform(X_train, y_class_train)
x_test_selected=selector.transform(X_test)


# In[12]:


selected_features=X_train.columns[selector.get_support()]
selected_features


# In[13]:


from sklearn.feature_selection import f_regression
selector= SelectKBest(score_func=f_regression,k=10)
x_train_sel=selector.fit_transform(X_train,y_reg_train)
x_test_sel=selector.transform(X_test)


# In[14]:


# Create a DataFrame for F-values and p-values
p_values=selector.pvalues_
f_values=selector.scores_
anova_df = pd.DataFrame({
    'Feature': X.columns,
    'F-value': f_values,
    'p-value': p_values
})

# Sort features by F-value in descending order
anova_df = anova_df.sort_values(by='F-value', ascending=False)

# Select the top 10 features
top_features = anova_df.head(10)

# Display only feature names
top_feature_names = top_features['Feature'].tolist()
print(top_feature_names)


# In[15]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler

class CustomPipelineWithFeatureSelection(BaseEstimator, TransformerMixin):
    def __init__(self, clf, reg, clf_features, reg_features):
        self.clf = clf
        self.reg = reg
        self.clf_features = clf_features
        self.reg_features = reg_features
        self.scaler_clf = StandardScaler()
        self.scaler_reg = StandardScaler()

    def fit(self, X, y_class, y_reg):
        # Ensure X is a DataFrame and contains the specified features
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X should be a pandas DataFrame")

        # Extract features for classification and scale them
        X_clf = X[self.clf_features]
        X_clf_scaled = self.scaler_clf.fit_transform(X_clf)
        self.clf.fit(X_clf_scaled, y_class)
        
        # Filter data where classification is 1
        X_filtered = X[y_class == 1]
        y_reg_filtered = y_reg[y_class == 1]
        
        # Extract features for regression and scale them
        X_filtered_reg = X_filtered[self.reg_features]
        X_filtered_reg_scaled = self.scaler_reg.fit_transform(X_filtered_reg)
        self.reg.fit(X_filtered_reg_scaled, y_reg_filtered)
        return self

    def predict(self, X):
        # Ensure X is a DataFrame and contains the specified features
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X should be a pandas DataFrame")

        # Extract features for classification and scale them
        X_clf = X[self.clf_features]
        X_clf_scaled = self.scaler_clf.transform(X_clf)
        y_class_pred = self.clf.predict(X_clf_scaled)
        
        # Initialize predictions for regression with NaNs
        y_reg_pred = np.full(X.shape[0], np.nan)
        
        # Filter data where classification is 1
        X_filtered = X[y_class_pred == 1]
        if len(X_filtered) > 0:
            # Extract features for regression and scale them
            X_filtered_reg = X_filtered[self.reg_features]
            X_filtered_reg_scaled = self.scaler_reg.transform(X_filtered_reg)
            y_reg_pred_filtered = self.reg.predict(X_filtered_reg_scaled)
            # Assign regression predictions to corresponding positions
            y_reg_pred[y_class_pred == 1] = y_reg_pred_filtered
        return y_class_pred, y_reg_pred

# Define feature sets
clf_features = ['Units', 'PropertyType', 'OrigLoanTerm', 'MonthsDelinquent','MonthsInRepayment', 
                'FirstPaymentYear', 'MaturityYear', 'CreditRange','RePayRange', 'Occupancy_O']

reg_features = ['MonthlyIncome', 'InterestAmount', 'Totalpayment', 'MonthlyInstallment', 'OrigUPB', 
                'CurrentUPB', 'DTI', 'NumBorrowers', 'LoanPurpose_N', 'OCLTV']
# Create and fit the custom pipeline with Random Forest Regressor
pipeline = CustomPipelineWithFeatureSelection(
    clf=GaussianNB(),
    reg=RandomForestRegressor(n_estimators=100, max_depth=10, min_samples_split=5, min_samples_leaf=5, random_state=42),
    clf_features=clf_features,
    reg_features=reg_features
)

# Fit the pipeline
pipeline.fit(X_train, y_class_train, y_reg_train)

# Make predictions
y_class_pred, y_reg_pred = pipeline.predict(X_test)



# In[16]:


# Evaluate classification model
print("Naive Bayes Classification Report:")
print("Accuracy:", accuracy_score(y_class_test, y_class_pred))
print("Precision:", precision_score(y_class_test, y_class_pred, average='weighted'))
print("Recall:", recall_score(y_class_test, y_class_pred, average='weighted'))
print("F1 Score:", f1_score(y_class_test, y_class_pred, average='weighted'))


# In[17]:


mask=~np.isnan(y_reg_pred)
print("Random Regressor Performance Metrics:")
print("Mean Absolute Error:", mean_absolute_error(y_reg_test[mask], y_reg_pred[mask]))
print("Mean Squared Error:", mean_squared_error(y_reg_test[mask],y_reg_pred[mask]))
print("R^2 Score:", r2_score(y_reg_test[mask], y_reg_pred[mask]))


# In[18]:


# Create DataFrame of predicted values
results_df = pd.DataFrame({
    'True Classification': y_class_test,
    'Predicted Classification': y_class_pred,
    'True Regression': y_reg_test,
    'Predicted Regression': y_reg_pred
})
results_df


# In[19]:


import joblib

# Save the pipeline
joblib.dump(pipeline, 'MBS_combined_pipeline.pkl')


# In[ ]:




