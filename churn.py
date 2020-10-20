# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 13:53:21 2020

@author: Chamsedine
"""

#import library
import pandas as pd
import timeit
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import warnings
warnings.filterwarnings("ignore")
import streamlit as st
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder 

#main script
st.title("Chunrn-App-Bacth-Prediction")

df = pd.read_csv("Churn.csv", sep=",") #st.cache

#print the description, the correlation and the describe of dataset
if st.sidebar.checkbox("Show what the dataframe look like"):
    st.write("Display the first five rows\n", df.head())
    st.write("Some information on dataset", df.describe())
    st.write("Shape of dataframe", df.shape)
    
#Print valid and fraud transactions
fraud = df[df.Exited==1]
valid = df[df.Exited==0]

outlier_percentage = (df.Exited.value_counts()[1] / df.Exited.value_counts()[0]) * 100

if st.sidebar.checkbox("show fraud and valid transaction details"):
    st.write("Fraudulent Transaction are : %.3f%%" %outlier_percentage)
    st.write("Fraud cases:", len(fraud))
    st.write("Valid cases:", len(valid))
    

#Impute nans with mean for numeris and most frequent for categoricals
cat_imp = SimpleImputer(strategy="most_frequent")
if len(df.loc[:,df.dtypes == 'object'].columns) != 0:
    df.loc[:,df.dtypes == 'object'] = cat_imp.fit_transform(df.loc[:,df.dtypes == 'object'])
    imp = SimpleImputer(missing_values = np.nan, strategy="mean")
    df.loc[:,df.dtypes != 'object'] = imp.fit_transform(df.loc[:,df.dtypes != 'object'])


# One hot encoding for categorical variables
cats = df.dtypes == 'object'
le = LabelEncoder() 
for x in df.columns[cats]:
    sum(pd.isna(df[x]))
    df.loc[:,x] = le.fit_transform(df[x])
    onehotencoder = OneHotEncoder() 
    df.loc[:, ~cats].join(pd.DataFrame(data=onehotencoder.fit_transform(df.loc[:,cats]).toarray(), columns= onehotencoder.get_feature_names()))

    
    
#Obtains X(feature) ans y(target)
X = df.drop(["Exited"], axis = 1)
y = df.Exited

#Split the data into training et test set
from sklearn.model_selection import train_test_split
size = st.sidebar.slider("Test Set Size", min_value = 0.2, max_value = 0.4)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = size, random_state = 42)



if st.sidebar.checkbox("Show the shape of Training and Test set Features and Labels"):
    st.write("X_train: ", X_train.shape)
    st.write("X_test: ", X_test.shape)
    st.write("y_train: ", y_train.shape)
    st.write("y_test: ", y_test.shape)


#import classification and models metrics
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.model_selection import cross_val_score


logreg = LogisticRegression()
knn = KNeighborsClassifier()
svc = SVC()
rforest = RandomForestClassifier(random_state=42)
etree = ExtraTreesClassifier(random_state=42)


features = X_train.columns.tolist()


#Feature selection through feature importance
#@st.cache
def feature_sort(model, X_train, y_train):
    # feature selection
    mod = model
    mod.fit(X_train, y_train)
    # get importance
    imp = mod.feature_importances_
    return imp

#Classifiers for feature importance
clf = ["Extra Trees", "Random Forest"]
mod_feature = st.sidebar.selectbox("Which model for feature importance?", clf)

start_time = timeit.default_timer()
if mod_feature == "Extra Trees":
    model = etree
    importance = feature_sort(model, X_train, y_train)
elif mod_feature == "Random Forest":
    model = rforest
    importance = feature_sort(model, X_train, y_train)

elapsed = timeit.default_timer() - start_time
st.write("Execution Time for feature selection: %.2f minutes" %(elapsed/60))

#plot of feature importance
if st.sidebar.checkbox("Show plot of feature importance"):
    plt.bar([x for x in range(len(importance))], importance)
    plt.title("Feature Importance")
    plt.xlabel("Feature( variale number)")
    plt.ylabel("Importance")
    st.pyplot()


feature_imp = list(zip(features,importance))
feature_sort = sorted(feature_imp, key = lambda x: x[1])

n_top_features = st.sidebar.slider('Number of top features', min_value = 5, max_value = 20)

top_features = list(list(zip(*feature_sort[-n_top_features:]))[0])

if st.sidebar.checkbox('Show selected top features'):
    st.write('Top %d features in order of importance are: %s'%(n_top_features,top_features[::-1]))

X_train_sfs=X_train[top_features]
X_test_sfs=X_test[top_features]

X_train_sfs_scaled=X_train_sfs
X_test_sfs_scaled=X_test_sfs


#Import performance metrics, inbalanced rectifiers
from sklearn.metrics import confusion_matrix, classification_report, matthews_corrcoef
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import NearMiss
np.random.seed(42)

smt = SMOTE()
nr = NearMiss()

def compute_performance(model, X_train, y_train, X_test, y_test):
    start_time = timeit.default_timer()
    scores = cross_val_score(model,X_train,y_train,cv=3,scoring="accuracy").mean()
    "Accuracy: ", scores
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    "Confusion Matrix: ", cm
    cr = classification_report(y_test, y_pred)
    "Classification Report: ", cr
    mcc = matthews_corrcoef(y_test, y_pred)
    "Matthews Correlation Coefficient: ",mcc
    elapsed = timeit.default_timer() - start_time
    "Execution Time for performance computation: %.2f minutes"%(elapsed/60)
   
#Run the differents classification models and  rectifiers 

if st.sidebar.checkbox("Run credit card fraud detection"):
    alg = ["Extra Trees", "k Nearest Neighbor", "Random Forest", "Logistic Regression","Suppor Vector Machine"]
    classifier = st.sidebar.selectbox("which algorithm", alg)
    rectifier = ["SMOTE", "Near Miss", "No Reactifier"]
    imb_rect = st.sidebar.selectbox("Which inbalanced class rectifier?", rectifier)
    
    if classifier == "Logistic Regression":
        model = logreg
        if imb_rect == "No Reactifier":
            compute_performance(model, X_train_sfs_scaled, y_train,X_test_sfs_scaled,y_test)
        elif imb_rect=='SMOTE':
                rect=smt
                st.write('Shape of imbalanced y_train: ',np.bincount(y_train))
                X_train_bal, y_train_bal = rect.fit_sample(X_train_sfs_scaled, y_train)
                st.write('Shape of balanced y_train: ',np.bincount(y_train_bal))
                compute_performance(model, X_train_bal, y_train_bal,X_test_sfs_scaled,y_test)
        elif imb_rect=='Near Miss':
            rect=nr
            st.write('Shape of imbalanced y_train: ',np.bincount(y_train))
            X_train_bal, y_train_bal = rect.fit_sample(X_train_sfs_scaled, y_train)
            st.write('Shape of balanced y_train: ',np.bincount(y_train_bal))
            compute_performance(model, X_train_bal, y_train_bal,X_test_sfs_scaled,y_test)
    elif classifier == 'k Nearest Neighbor':
        model=knn
        if imb_rect=='No Rectifier':
            compute_performance(model, X_train_sfs_scaled, y_train,X_test_sfs_scaled,y_test)
        elif imb_rect=='SMOTE':
                rect=smt
                st.write('Shape of imbalanced y_train: ',np.bincount(y_train))
                X_train_bal, y_train_bal = rect.fit_sample(X_train_sfs_scaled, y_train)
                st.write('Shape of balanced y_train: ',np.bincount(y_train_bal))
                compute_performance(model, X_train_bal, y_train_bal,X_test_sfs_scaled,y_test)
        elif imb_rect=='Near Miss':
            rect=nr
            st.write('Shape of imbalanced y_train: ',np.bincount(y_train))
            X_train_bal, y_train_bal = rect.fit_sample(X_train_sfs_scaled, y_train)
            st.write('Shape of balanced y_train: ',np.bincount(y_train_bal))
            compute_performance(model, X_train_bal, y_train_bal,X_test_sfs_scaled,y_test)    
    
    elif classifier == 'Support Vector Machine':
        model=svm
        if imb_rect=='No Rectifier':
            compute_performance(model, X_train_sfs_scaled, y_train,X_test_sfs_scaled,y_test)
        elif imb_rect=='SMOTE':
                rect=smt
                st.write('Shape of imbalanced y_train: ',np.bincount(y_train))
                X_train_bal, y_train_bal = rect.fit_sample(X_train_sfs_scaled, y_train)
                st.write('Shape of balanced y_train: ',np.bincount(y_train_bal))
                compute_performance(model, X_train_bal, y_train_bal,X_test_sfs_scaled,y_test)
        elif imb_rect=='Near Miss':
            rect=nr
            st.write('Shape of imbalanced y_train: ',np.bincount(y_train))
            X_train_bal, y_train_bal = rect.fit_sample(X_train_sfs_scaled, y_train)
            st.write('Shape of balanced y_train: ',np.bincount(y_train_bal))
            compute_performance(model, X_train_bal, y_train_bal,X_test_sfs_scaled,y_test)    
        
    elif classifier == 'Random Forest':
        model=rforest
        if imb_rect=='No Rectifier':
            compute_performance(model, X_train_sfs_scaled, y_train,X_test_sfs_scaled,y_test)
        elif imb_rect=='SMOTE':
                rect=smt
                st.write('Shape of imbalanced y_train: ',np.bincount(y_train))
                X_train_bal, y_train_bal = rect.fit_sample(X_train_sfs_scaled, y_train)
                st.write('Shape of balanced y_train: ',np.bincount(y_train_bal))
                compute_performance(model, X_train_bal, y_train_bal,X_test_sfs_scaled,y_test)
        elif imb_rect=='Near Miss':
            rect=nr
            st.write('Shape of imbalanced y_train: ',np.bincount(y_train))
            X_train_bal, y_train_bal = rect.fit_sample(X_train_sfs_scaled, y_train)
            st.write('Shape of balanced y_train: ',np.bincount(y_train_bal))
            compute_performance(model, X_train_bal, y_train_bal,X_test_sfs_scaled,y_test)  
            
    elif classifier == 'Extra Trees':
        model=etree
        if imb_rect=='No Rectifier':
            compute_performance(model, X_train_sfs_scaled, y_train,X_test_sfs_scaled,y_test)
        elif imb_rect=='SMOTE':
                rect=smt
                st.write('Shape of imbalanced y_train: ',np.bincount(y_train))
                X_train_bal, y_train_bal = rect.fit_sample(X_train_sfs_scaled, y_train)
                st.write('Shape of balanced y_train: ',np.bincount(y_train_bal))
                compute_performance(model, X_train_bal, y_train_bal,X_test_sfs_scaled,y_test)
        elif imb_rect=='Near Miss':
            rect=nr
            st.write('Shape of imbalanced y_train: ',np.bincount(y_train))
            X_train_bal, y_train_bal = rect.fit_sample(X_train_sfs_scaled, y_train)
            st.write('Shape of balanced y_train: ',np.bincount(y_train_bal))
            compute_performance(model, X_train_bal, y_train_bal,X_test_sfs_scaled,y_test)
    
    
 
    
    
    
                             




    
        
















    

    
    



