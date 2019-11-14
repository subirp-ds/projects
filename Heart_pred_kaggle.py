# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report


df=pd.read_csv("E:\data science\heart-disease-uci\heart.csv")
df.shape
df.info()

import seaborn as sns
#get correlations of each features in dataset
corrmat = df.corr()
top_corr_features = corrmat.index
plt.figure(figsize=(20,20))
#plot heat map
g=sns.heatmap(df[top_corr_features].corr(),annot=True,cmap="RdYlGn")
#check histogram to find variables which may be actually categorical
df.hist()
#df=df.drop(['fbs'], axis=1)
#identified cat data getting encoded
dataset = pd.get_dummies(df, columns = ['sex', 'cp','restecg', 'exang', 'slope', 'ca', 'thal'],drop_first=True)

#scaling numberical variables
from sklearn.preprocessing import StandardScaler
standardScaler = StandardScaler()
columns_to_scale = ['age', 'trestbps', 'chol', 'fbs', 'thalach', 'oldpeak']
dataset[columns_to_scale] = standardScaler.fit_transform(dataset[columns_to_scale])

#setting dependent variable

y = dataset['target']
X = dataset.drop(['target'], axis = 1)

#split train and test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

#apply KNN
#selecting best value of KNN with Cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
knn_scores = []
for k in range(1,21):
    knn_classifier = KNeighborsClassifier(n_neighbors = k)
    score=cross_val_score(knn_classifier,X_train,y_train,cv=10,scoring ='accuracy')
    knn_scores.append(score.mean())
#printing accuracy scores for KNN with N
for i in range(1,21):
    print(i, knn_scores[i-1], (i, knn_scores[i-1]))
    
#best value is 8
knn_classifier = KNeighborsClassifier(n_neighbors = 8)
#score=cross_val_score(knn_classifier,X,y,cv=10).mean()
knn_classifier.fit(X_train, y_train)
knn_pred = knn_classifier.predict(X_test)
    
knn_acc_train = knn_classifier.score(X_train, y_train)*100
knn_acc_test = knn_classifier.score(X_test, y_test)*100

print("Train Accuracy {:.2f}%".format(knn_acc_train))
print("Test Accuracy {:.2f}%".format(knn_acc_test))  

#apply Random Forest
#Tune parameter on Random Forest
#setting param grid
bootstrap=[True, False]
max_depth=[10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None]
max_features=['auto', 'sqrt']
min_samples_leaf=[1, 2, 4]
min_samples_split=[2, 5, 10]
#n_estimators=[100, 200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800]
n_estimators=[10, 50, 100, 200, 400, 600, 800, 1000]

random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
from pprint import pprint
pprint(random_grid)

#Randomzied CV to calculate score with best params for RF
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()
# Random search of parameters, using 3 fold cross validation, 
# search across 100 different combinations, and use all available cores
from sklearn.model_selection import RandomizedSearchCV
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, 
                               n_iter = 100, cv = 10, verbose=2, random_state=42, n_jobs = -1)

rf_random.fit(X_train, y_train)

rf_random.best_params_
#making the best rf classifier with params from above
rtree = RandomForestClassifier(n_estimators=50,min_samples_split=10,min_samples_leaf=2,
                               max_features='sqrt',max_depth=20,bootstrap=True)
rtree.fit(X_train, y_train)
rtree_pred = rtree.predict(X_test)
#rtree_conf = confusion_matrix(y_test, rtree_pred)
#rtree_class = classification_report(y_test, rtree_pred)
rtree_acc_train = rtree.score(X_train, y_train)*100
rtree_acc_test = rtree.score(X_test, y_test)*100

print("Train Accuracy {:.2f}%".format(rtree_acc_train))
print("Test Accuracy {:.2f}%".format(rtree_acc_test))


#apply xgboost
#tune parameters

import xgboost
xg_class=xgboost.XGBClassifier()

#set different param values
base_score=[0.25,0.5,0.75,0.9]
n_estimators = [50, 100, 500, 900, 1100]
max_depth = [2, 3, 5, 10, 15]
booster=['gbtree','gblinear']
learning_rate=[0.05,0.1,0.15,0.20]
min_child_weight=[1,2,3,4]

# Define the grid of hyperparameters to search
#params={
# "learning_rate"    : [0.05, 0.10, 0.15, 0.20, 0.25, 0.30 ] ,
# "max_depth"        : [ 3, 4, 5, 6, 8, 10, 12, 15],
# "min_child_weight" : [ 1, 3, 5, 7 ],
# "gamma"            : [ 0.0, 0.1, 0.2 , 0.3, 0.4 ],
# "colsample_bytree" : [ 0.3, 0.4, 0.5 , 0.7 ]
#    
#}

hyperparameter_grid = {
    'n_estimators': n_estimators,
    'max_depth':max_depth,
    'learning_rate':learning_rate,
    'min_child_weight':min_child_weight,
    'booster':booster,
    'base_score':base_score
    }

# Set up the random search CV with 4-fold cross validation
#random_cv = RandomizedSearchCV(estimator=xg_class,
#            param_distributions=params,
#            cv=5,scoring='roc_auc',n_jobs = 4,
#            verbose = 3,random_state=42)

random_cv = RandomizedSearchCV(estimator=xg_class,
            param_distributions=hyperparameter_grid,
            cv=5,scoring='roc_auc',n_jobs = 4,
            verbose = 3,random_state=42)

random_cv.fit(X_train,y_train)

random_cv.best_estimator_

#define regressor with above output of best parameters
#xg_class=xgboost.XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
#              colsample_bynode=1, colsample_bytree=0.7, gamma=0.1,
#              learning_rate=0.05, max_delta_step=0, max_depth=10,
#              min_child_weight=1, missing=None, n_estimators=100, n_jobs=1,
#              nthread=None, objective='binary:logistic', random_state=0,
#              reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
#              silent=None, subsample=1, verbosity=1)

xg_class=xgboost.XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
              colsample_bynode=1, colsample_bytree=0.7, gamma=0.1,
              learning_rate=0.2, max_delta_step=0, max_depth=3,
              min_child_weight=3, missing=None, n_estimators=50, n_jobs=1,
              nthread=None, objective='binary:logistic', random_state=0,
              reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
              silent=None, subsample=1, verbosity=1)


xg_class.fit(X_train,y_train)

xg_acc_train = xg_class.score(X_train, y_train)*100
xg_acc_test = xg_class.score(X_test, y_test)*100

print("Train Accuracy {:.2f}%".format(xg_acc_train))
print("Test Accuracy {:.2f}%".format(xg_acc_test))