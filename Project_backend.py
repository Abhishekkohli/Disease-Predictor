# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 00:06:25 2020

@author: MAHE
"""
import matplotlib.pyplot as plt
import sklearn as sk
from sklearn.metrics import classification_report as cr
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import RFE
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


# For categorical columns having unique values greater than 10, Label Encoding is done and for the rest, OneHotEncoding is done
#label encoding
def label_encode(P,label_list):
    label_encoder=sk.preprocessing.LabelEncoder()
    for col in label_list:
        P[col]=label_encoder.fit_transform(P[col])
    return P

#one hot encoding
def one_hot_encode(X,l):
    one_hot_encoder=sk.preprocessing.OneHotEncoder(sparse='False')
    X_encode = one_hot_encoder.fit_transform(X[l])
    X_encode.index = X.index
    X.drop(l,axis=1)
    X=pd.concat([X,X_encode],axis=1)
    return X
    
#Imputing missing values
def miss(X,l):
    impute=SimpleImputer(strategy='most_frequent')    
    y=pd.DataFrame(impute.fit_transform(X[l]))
    y.columns=X.columns
    return y
    
#Implementing Logistic Regression
def logistic_regression(X,X1,Y,Y1):
    global accuracy_models
    lr=LogisticRegression(multi_class='auto',solver='liblinear')
    lr.fit(X,Y)
    pred=lr.predict(X1)
    print(accuracy_score(Y1,pred))
    accuracy_models.append(accuracy_score(Y1,pred))
    cd=confusion_matrix(Y1,pred,range(len(Y1.unique())))
    print(cd)
    print(cr(Y_test,pred))
    return lr

def k_nearest_neighbors(X,X1,Y,Y1):
    global accuracy_models
    nn=range(3,11)
# Empty list that will hold cv scores
    cv_scores = []
# Perform 5-fold cross validation
# ---------------------------------
    for k in nn:
        knn = sk.neighbors.KNeighborsClassifier(n_neighbors=k)
        scores = cross_val_score(knn, X,Y, cv=5,scoring='accuracy')
        cv_scores.append(scores.mean())
    optimal_k = nn[cv_scores.index(max(cv_scores))]
    knn = sk.neighbors.KNeighborsClassifier(n_neighbors=optimal_k)
    knn.fit(X,Y)
    pred=knn.predict(X1)
    print(accuracy_score(Y1,pred))
    accuracy_models.append(accuracy_score(Y1,pred))
    cd=confusion_matrix(Y1,pred,range(len(Y1.unique())))
    print(cd)
    print(cr(Y_test,pred))
    return knn

def random_forest_classifier(X,X1,Y,Y1):
    global accuracy_models
    rfc=RandomForestClassifier(criterion='gini',n_estimators=200,random_state=0,max_leaf_nodes=1000,)
    rfc.fit(X,Y)
    pred=rfc.predict(X1)
    print(accuracy_score(Y1,pred))
    accuracy_models.append(accuracy_score(Y1,pred))
    cd=confusion_matrix(Y1,pred,range(len(Y1.unique())))
    print(cd)
    print(cr(Y_test,pred))
    return rfc 

# Reading the datasets into dataframe
disease_train = pd.read_csv('Training.csv')

disease_train.shape

disease_test=pd.read_csv("Testing.csv")

disease_test.shape
print("Total no. of categories %d "%(len(disease_test.iloc[:,-1].unique())))
cols_train=list(disease_train.columns)
print("columns : ")
print(cols_train)
# Since disease_test dataframe has a very less number of rows as compared to disease_train, so it is not appropriate to use it for validation.
#Hence,we don't use it for validation, and split our training dataset to get validation dataset. 

disease_train.head()

# Check for categorical values in the dataset
cols_categorical=[col for col in cols_train if disease_train[col].dtype == 'object']

print("total categorical columns %d"%(len(cols_categorical)-1))

cols_categorical.pop()

if len(cols_categorical)!=0 :
    label_list=[]
    for col in cols_categorical:
        if disease_train[col].nunique() > 10:
            label_list.append(col)
            cols_categorical.remove(col)
    disease_train=label_encode(disease_train,label_list)
    disease_train=one_hot_encode(disease_train,cols_categorical)

#Although We do not need to convert categorical values of target column into numeric ones as sklearn model handles it internally but for correlation finding
# we have to convert it into numeric ones using Label Encoding as it will be better than One Hot Encoding   
target_cat=disease_train.iloc[:,-1]
disease_train=label_encode(disease_train,cols_train[-1:])    
target_label=disease_train.iloc[:,-1]
label_to_cat=set(zip(target_label,target_cat))
print("top 5 rows of dataframe")
print(disease_train.head())

#Checking for missing or wrong values except the target variable
cols_null=[]
for col in cols_train[0:-1]:
    t=0
    for j in disease_train.index:
        if not (disease_train[col][j]==0 or disease_train[col][j]==1):
            if t==0:
                cols_null.append(col)
                t=1
            disease_train.loc[j,col]=np.nan
print("list of columns having missing values",end="")            
print(cols_null) 
       
    
if len(cols_null)!=0 :
    disease_train=miss(disease_train,cols_null) 

#Dropping the rows having their target variables missing as ther is no point in doing imputation for that
disease_train.dropna(subset=['prognosis'],axis=0,inplace=True)

#Randomly shuffling the dataset
disease_train=disease_train.sample(frac=1)      

#USING FILTER METHOD

#Selecting only those columns not having a weak correlation with target variable
corr_matrix=disease_train.corr()
fig, ax = plt.subplots(figsize=(50,50))
sns.heatmap(corr_matrix,xticklabels=corr_matrix.columns,yticklabels=corr_matrix.columns)
corr_matrix.iloc[:,-1]=abs(corr_matrix.iloc[:,-1])

corr_feat = (corr_matrix.iloc[:,-1][corr_matrix.iloc[:,-1] >= 0.5])
print(len(corr_feat))

corr_feat = (corr_matrix.iloc[:,-1][corr_matrix.iloc[:,-1] >= 0.3])
print(len(corr_feat))

corr_feat = (corr_matrix.iloc[:,-1][corr_matrix.iloc[:,-1] >= 0.2])
print(len(corr_feat))

#Getting target variable
X=disease_train.iloc[:,:-1]
Y=disease_train.iloc[:,-1]

#Splitting dataset into training and validation
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.3,random_state=1)
print(X_train.shape)
print(X_test.shape)

#From the selected columns, from the columns which are highly correlated to each other, we keep the feature 
#which is the most correlated to target variable and remove the rest 
features=list(corr_feat.index)
features.remove('prognosis')
corr_feat_matrix=disease_train[list(corr_feat.index)].corr()
threshold=0.5
cols_set=set()
for i in range(len(corr_feat_matrix.columns)):
    for j in range(i):
        if corr_feat_matrix.iloc[j,i]>=threshold and ((corr_feat_matrix.columns[i] not in cols_set) and (corr_feat_matrix.columns[j] not in cols_set)):
                if corr_feat_matrix.iloc[i,-1] > corr_feat_matrix.iloc[j,-1]:
                    cols_set.add(corr_feat_matrix.columns[j])
                else:
                    cols_set.add(corr_feat_matrix.columns[i])
            

features1=list(set(features) - cols_set)
print("Optimal features set obtained by filter method")
print(features1)

#Checking the accuracy of filter method:
svc=sk.svm.SVC(kernel='linear',random_state=2)
svc.fit(X_train[features1],Y_train)
pred_features1=svc.predict(X_test[features1])
print("Accuracy :%f"%(accuracy_score(Y_test,pred_features1))) # ACCURACY : 0.38

#USING WRAPPER METHOD
#1. Backward elimination on training dataset

X_train=sm.add_constant(X_train)
X_train_dup=X_train.copy()
cols=list(X_train_dup.columns)
while(len(cols) > 0):
    ols_model=sm.OLS(Y_train,X_train_dup).fit()
    max_pvalue_label=ols_model.pvalues.idxmax()
    max_pvalue=max(ols_model.pvalues)
    #Generally, alpha=0.05 but because dataset is too large so alpha decreases
    if max_pvalue >= 0.01:
        cols.remove(max_pvalue_label)
        X_train_dup.drop([max_pvalue_label],axis=1,inplace=True) 
    else:
        break

features2=cols
features2.remove('const')
print("length of features set obtained by backward elimination %d"%(len(features2)))
X_train.drop(['const'],axis=1,inplace=True)

#Checking the accuracy of backward elimination method on testing dataset:
svc=sk.svm.SVC(kernel='linear',random_state=2)
svc.fit(X_train[features2],Y_train)
pred_features2=svc.predict(X_test[features2])
print("Accuracy :%f"%(accuracy_score(Y_test,pred_features2))) #ACCURACY: 0.99

model=LogisticRegression(multi_class='auto',solver='liblinear')

print("Implemeting RFE on certain no. of columns...")
#2.Recursive Feature Elimination(Although very time consuming but gives high accuracy)
nof_score=[]
nof=[]
cols_length=len(X_train.columns)
for i in range(0,cols_length,20) :
    rfe=RFE(model,n_features_to_select=i+1)
    X_train_rfe=rfe.fit_transform(X_train,Y_train)
    X_test_rfe=rfe.transform(X_test)
    scores=cross_val_score(model,X_train_rfe,Y_train,cv=3,scoring='accuracy')
    nof_score.append(scores.mean())
    nof.append(i+1)

print("To obtain minimal no. of features giving highest accuracy...")
#plotting scores
plt.subplots(figsize=(9,9))
plt.grid()
plt.plot(nof,nof_score)
plt.show()

high_score=0
no_of_feat=0
for i in range(46,57):
    rfe=sk.feature_selection.RFE(model,n_features_to_select=i)
    X_train_rfe=rfe.fit_transform(X_train,Y_train)
    X_test_rfe=rfe.transform(X_test)
    scores=cross_val_score(model,X_train_rfe,Y_train,cv=3,scoring='accuracy')
    acc=scores.mean()
    if high_score < acc:
        high_score=acc
        no_of_feat=i

print("no. of features obtained by RFE : %d"%(no_of_feat))        
rfe=sk.feature_selection.RFE(model,no_of_feat)
X_train_rfe=rfe.fit_transform(X_train,Y_train)
features3=[cols_train[i] for i in range(len(rfe.support_)) if rfe.support_[i]]        
print("Feature set via rfe")
print(features3)
#'Lethargy and Fatigue are same'
features3.remove('lethargy')
#Checking the accuracy of RFE method by implementing Support Vector Machines:
#Kernel='linear' as dataset is huge
accuracy_models=[]
svc=sk.svm.SVC(kernel='linear',random_state=2)
svc.fit(X_train[features3],Y_train)
pred_features3=svc.predict(X_test[features3])
accuracy_models.append(accuracy_score(Y_test,pred_features3)) 
print(accuracy_models[0]) #ACCURACY : 1.0 

#Hence,we  select RFE Method featuring.
X_train=X_train[features3]
X_test=X_test[features3]

model_list=[]

#Applying different classification algorithms
#Already, Support Vector Machine has been implemented
model_list.append(svc)
model_list.append(logistic_regression(X_train,X_test,Y_train,Y_test))
model_list.append(k_nearest_neighbors(X_train,X_test,Y_train,Y_test))
model_list.append(random_forest_classifier(X_train,X_test,Y_train,Y_test))

j=accuracy_models.index(max(accuracy_models))

