import os
import pandas as pd
#import tensorflow
import numpy as np
from sklearn import preprocessing
import seaborn as sns
sns.set(color_codes=True)
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score

merged_data = pd.read_csv('dataset_6classses_rms.csv')
merged_data = merged_data.drop(merged_data.columns[0], axis = 1)
um_cols = ['Bearing 1']
merged_data.columns = ['Bearing 1-1','Bearing 1-2','Bearing 2-1','Bearing 2-2', 'Bearing 3-1','Bearing 3-2','Bearing 4-1','Bearing 4-2', 'FaultType']
merged_data.reset_index(drop=True, inplace=True)

num_cols = ['Bearing 1-1','Bearing 1-2','Bearing 2-1','Bearing 2-2', 'Bearing 3-1','Bearing 3-2','Bearing 4-1','Bearing 4-2']

dataset_train = merged_data 

dataset = merged_data.values
scaler = StandardScaler()
X = dataset[:,0:8].astype(float)
X = scaler.fit_transform(X)
Y = dataset[:,8].astype(int)


# X_train, X_test, y_train, y_test = train_test_split(X, Y, stratify = Y , train_size=0.7, random_state=7)
X_train, X_test, y_train, y_test = train_test_split(X, Y, stratify = Y , train_size=0.7, shuffle=True)

Class1_acc_train = []
Class2_acc_train = []
Class3_acc_train = []
Class4_acc_train = []
Class5_acc_train = []
Class6_acc_train = []

Class1_acc_test = []
Class2_acc_test = []
Class3_acc_test = []
Class4_acc_test = []
Class5_acc_test = []
Class6_acc_test = []

trees_depth = []
forest_acc = []
forest_acc_training = []

for n_est in range(1,50,1):

    # rand_forest = RandomForestClassifier(criterion="entropy", random_state=0, max_features='sqrt',n_estimators=n_est, bootstrap=True)
    rand_forest = RandomForestClassifier(criterion="entropy", max_features='sqrt',n_estimators=12,max_depth=n_est, bootstrap=True)

    model = rand_forest.fit(X_train,y_train)

    pred = model.predict(X_test)

    pred_acc = confusion_matrix(y_test, pred)
    # pred_recall = recall_score(y_test, pred)
    print(type(pred_acc))

    cm_trace =  np.trace(pred_acc)
    elements_sum = np.sum(pred_acc)

    accuracy = cm_trace/elements_sum

    forest_acc.append(accuracy)
    trees_depth.append(n_est)

    print('\n\n',pred_acc)
    print('\n\n The accuracy in prediction for validation by Random Forest Classifier is: ',accuracy)


    pred_train_loss = model.predict(X_train)

    pred_acc_training = confusion_matrix(y_train, pred_train_loss)
    # pred_recall = recall_score(y_test, pred)
    print(type(pred_acc))

    cm_trace =  np.trace(pred_acc_training)
    elements_sum = np.sum(pred_acc_training)

    accuracy = cm_trace/elements_sum

    forest_acc_training.append(accuracy)
    

    print('\n\n',pred_acc_training)
    print('\n\n The accuracy in prediction for training data by Random Forest Classifier is: ',accuracy)


    test_error_1 = 1 - (pred_acc[0,0]/(np.sum(pred_acc,axis=1)[0]))
    
    train_error_1 = 1 - (pred_acc_training[0,0]/(np.sum(pred_acc_training,axis=1)[0]))

    test_error_2 = 1 - (pred_acc[1,1]/(np.sum(pred_acc,axis=1)[1]))
    
    train_error_2 = 1 - (pred_acc_training[1,1]/(np.sum(pred_acc_training,axis=1)[1]))
  
    test_error_3 = 1 - (pred_acc[2,2]/(np.sum(pred_acc,axis=1)[2]))
    
    train_error_3 = 1 - (pred_acc_training[2,2]/(np.sum(pred_acc_training,axis=1)[2]))
  
    test_error_4 = 1 - (pred_acc[3,3]/(np.sum(pred_acc,axis=1)[3]))
    
    train_error_4 = 1 - (pred_acc_training[3,3]/(np.sum(pred_acc_training,axis=1)[3]))

    test_error_5 = 1 - (pred_acc[4,4]/(np.sum(pred_acc,axis=1)[4]))
    
    train_error_5 = 1 - (pred_acc_training[4,4]/(np.sum(pred_acc_training,axis=1)[4]))

    test_error_6 = 1 - (pred_acc[5,5]/(np.sum(pred_acc,axis=1)[5]))
    
    train_error_6 = 1 - (pred_acc_training[5,5]/(np.sum(pred_acc_training,axis=1)[5]))

    Class1_acc_test.append(test_error_1)
    Class1_acc_train.append(train_error_1)
    Class2_acc_test.append(test_error_2)
    Class2_acc_train.append(train_error_2)
    Class3_acc_test.append(test_error_3)
    Class3_acc_train.append(train_error_3)
    Class4_acc_test.append(test_error_4)
    Class4_acc_train.append(train_error_4)
    Class5_acc_test.append(test_error_5)
    Class5_acc_train.append(train_error_5)
    Class6_acc_test.append(test_error_6)
    Class6_acc_train.append(train_error_6)
  


fig = plt.figure("Figure",figsize=(16,8))
subplot1 = fig.add_subplot(2,2,1)
plt.plot(trees_depth,forest_acc, '-r',label = 'Random Forest Validation Accuracy', lw = 3)
plt.plot(trees_depth,forest_acc_training, '-g',label = 'Random Forest Training Accuracy', lw = 3)

plt.xlabel('Decision trees max depth', color = 'k', fontsize = 15, fontweight = 'bold')
plt.ylabel('Validation set Accuracy', color = 'k', fontsize = 15, fontweight = 'bold')
plt.legend(loc = 'best', fontsize = 'large')
plt.xticks(np.arange(0,52,4), fontsize = 10)
plt.yticks(np.arange(0,1.1,0.1),fontsize = 10)
plt.grid(True, color = 'k')

subplot2 = fig.add_subplot(2,2,2)

plt.plot(trees_depth,Class1_acc_test, '-r',label = 'Class 1 Validation Error', lw = 3)
plt.plot(trees_depth,Class1_acc_train, '--g',label = 'Class 1 Training Error', lw = 3)
plt.plot(trees_depth,Class2_acc_test, '-b',label = 'Class 2 Validation Error', lw = 3)
plt.plot(trees_depth,Class2_acc_train, '--k',label = 'Class 2 Training Error', lw = 3)


plt.xlabel('Tree Depth', color = 'k', fontsize = 15, fontweight = 'bold')
plt.ylabel('Class Prediction Error', color = 'k', fontsize = 15, fontweight = 'bold')
plt.legend(loc = 'best', fontsize = 'large')
plt.grid(True, color = 'k')

subplot3 = fig.add_subplot(2,2,3)

plt.plot(trees_depth,Class3_acc_test, linestyle = 'solid',color= 'magenta',label = 'Class 3 Validation Error', lw = 3)
plt.plot(trees_depth,Class3_acc_train, linestyle = 'dotted',color= 'midnightblue',label = 'Class 3 Training Error', lw = 3)
plt.plot(trees_depth,Class4_acc_test, linestyle = 'solid',color= 'brown',label = 'Class 4 Validation Error', lw = 3)
plt.plot(trees_depth,Class4_acc_train, linestyle = 'dotted',color= 'darkcyan',label = 'Class 4 Training Error', lw = 3)

plt.xlabel('Tree Depth', color = 'k', fontsize = 15, fontweight = 'bold')
plt.ylabel('Class Prediction Error', color = 'k', fontsize = 15, fontweight = 'bold')
plt.legend(loc = 'best', fontsize = 'large')
plt.grid(True, color = 'k')

subplot3 = fig.add_subplot(2,2,4)

plt.plot(trees_depth,Class5_acc_test, linestyle = 'solid',color= 'darkgreen',label = 'Class 5 Validation Error', lw = 3)
plt.plot(trees_depth,Class5_acc_train, linestyle = 'dotted',color= 'crimson',label = 'Class 5 Training Error', lw = 3)
plt.plot(trees_depth,Class6_acc_test, linestyle = 'solid',color= 'black',label = 'Class 6 Validation Error', lw = 3)
plt.plot(trees_depth,Class6_acc_train, linestyle = 'dotted',color= 'indigo',label = 'Class 6 Training Error', lw = 3)

plt.xlabel('Tree Depth', color = 'k', fontsize = 15, fontweight = 'bold')
plt.ylabel('Class Prediction Error', color = 'k', fontsize = 15, fontweight = 'bold')
plt.legend(loc = 'best', fontsize = 'large')
plt.grid(True, color = 'k')


plt.show()

print([estimator.tree_.max_depth for estimator in rand_forest.estimators_])



