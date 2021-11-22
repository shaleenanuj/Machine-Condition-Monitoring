import os
import pandas as pd
#import tensorflow
import numpy as np
from sklearn import preprocessing
import seaborn as sns
sns.set(color_codes=True)
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import make_pipeline
#Importing the necessary packages and libaries
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import svm, datasets
from sklearn.model_selection import GridSearchCV
from numpy.random import seed


merged_data = pd.read_csv('dataset_6classses_rms.csv')
merged_data = merged_data.drop(merged_data.columns[0], axis = 1)

num_cols = ['Bearing 1']
merged_data.columns = ['Bearing 1-1','Bearing 1-2','Bearing 2-1','Bearing 2-2', 'Bearing 3-1','Bearing 3-2','Bearing 4-1','Bearing 4-2', 'FaultType']
merged_data.reset_index(drop=True, inplace=True)

num_cols = ['Bearing 1-1','Bearing 1-2','Bearing 2-1','Bearing 2-2', 'Bearing 3-1','Bearing 3-2','Bearing 4-1','Bearing 4-2']

dataset_train = merged_data 
plt.plot(merged_data["FaultType"], merged_data[num_cols])
plt.xticks(rotation=70)



dataset = merged_data.values
# scaler = StandardScaler()
X = dataset[:,0:8].astype(float)
# X = scaler.fit_transform(X)
Y = dataset[:,8].astype(int)

svm_model = svm.SVC(kernel='rbf', C=84, gamma=2)
standardizer = StandardScaler()

pipeline = make_pipeline(standardizer, svm_model)

kf = KFold(n_splits=10, shuffle=True, random_state=1)

cv_results = cross_val_score(pipeline, X, Y, cv=kf, scoring="accuracy",n_jobs=-1) 

print(cv_results.mean())

