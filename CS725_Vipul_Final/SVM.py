import os
import pandas as pd
# import tensorflow
import numpy as np
from sklearn import preprocessing
import seaborn as sns
sns.set(color_codes=True)
import matplotlib.pyplot as plt
# %matplotlib inline


from numpy.random import seed



merged_data = pd.read_csv('dataset_6classses_rms.csv')
merged_data = merged_data.drop(merged_data.columns[0], axis = 1)



num_cols = ['Bearing 1']
merged_data.columns = ['Bearing 1-1','Bearing 1-2','Bearing 2-1','Bearing 2-2', 'Bearing 3-1','Bearing 3-2','Bearing 4-1','Bearing 4-2', 'FaultType']
merged_data.reset_index(drop=True, inplace=True)

num_cols = ['Bearing 1-1','Bearing 1-2','Bearing 2-1','Bearing 2-2', 'Bearing 3-1','Bearing 3-2','Bearing 4-1','Bearing 4-2']

dataset_train = merged_data

# plt.plot(merged_data["FaultType"], merged_data[num_cols])
# plt.xticks(rotation=70)
#Importing the necessary packages and libaries
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import svm, datasets
import matplotlib.pyplot as plt
import numpy as np

dataset = merged_data.values
scaler = StandardScaler()
X = dataset[:,0:8].astype(float)
X = scaler.fit_transform(X)
Y = dataset[:,8].astype(int)


X_train, X_test, y_train, y_test = train_test_split(X, Y , shuffle=True, train_size=0.7)

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

test_acc = []
train_loss = []
validation_loss = []
C = []
some_array = np.arange(0.1,101,1)
# for j in range(1,100):
for j in some_array:


    # linear = svm.SVC(kernel='linear', C=j).fit(X_train, y_train)
    linear = svm.SVC(kernel='rbf', C=j, gamma=2).fit(X_train, y_train)

    linear_pred_test = linear.predict(X_test)
    linear_pred_train = linear.predict(X_train)

    accuracy_lin_test = linear.score(X_test, y_test)
    print('\nTest accuracy for C as ',j, 'is:' ,accuracy_lin_test)

    validation_loss.append(1 - accuracy_lin_test)

    accuracy_lin_train = linear.score(X_train, y_train)
    print('\nTrain accuracy for C as ',j, 'is:' ,accuracy_lin_train)

    train_loss.append(1 - accuracy_lin_train)

    cm_test = confusion_matrix(y_test, linear_pred_test)
    print(cm_test)
    cm_train = confusion_matrix(y_train, linear_pred_train)
    print(cm_train)

    test_error_1 = 1 - (cm_test[0,0]/(np.sum(cm_test,axis=1)[0]))
    
    train_error_1 = 1 - (cm_train[0,0]/(np.sum(cm_train,axis=1)[0]))

    test_error_2 = 1 - (cm_test[1,1]/(np.sum(cm_test,axis=1)[1]))
    
    train_error_2 = 1 - (cm_train[1,1]/(np.sum(cm_train,axis=1)[1]))
  
    test_error_3 = 1 - (cm_test[2,2]/(np.sum(cm_test,axis=1)[2]))
    
    train_error_3 = 1 - (cm_train[2,2]/(np.sum(cm_train,axis=1)[2]))
  
    test_error_4 = 1 - (cm_test[3,3]/(np.sum(cm_test,axis=1)[3]))
    
    train_error_4 = 1 - (cm_train[3,3]/(np.sum(cm_train,axis=1)[3]))

    test_error_5 = 1 - (cm_test[4,4]/(np.sum(cm_test,axis=1)[4]))
    
    train_error_5 = 1 - (cm_train[4,4]/(np.sum(cm_train,axis=1)[4]))

    test_error_6 = 1 - (cm_test[5,5]/(np.sum(cm_test,axis=1)[5]))
    
    train_error_6 = 1 - (cm_train[5,5]/(np.sum(cm_train,axis=1)[5]))
  

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

    # print(cm_lin)

    # test_acc.append(accuracy_lin_test)
    C.append(j)

fig = plt.figure("Figure")

subplot1 = fig.add_subplot(2,2,1)
plt.plot(C,train_loss, '-r',label = 'Kernel SVM Training Error', lw = 3)
plt.plot(C,validation_loss, '-b',label = 'Kernel SVM Validation Error', lw = 3)
plt.xlabel('Hyperparameter C', color = 'k', fontsize = 15, fontweight = 'bold')
plt.ylabel('Error', color = 'k', fontsize = 15, fontweight = 'bold')
plt.legend(loc = 'best', fontsize = 'large')
plt.grid(True, color = 'k')

# ================================================================================= #
subplot2 = fig.add_subplot(2,2,2)

plt.plot(C,Class1_acc_test, '-r',label = 'Class 1 Validation Error', lw = 3)
plt.plot(C,Class1_acc_train, '--g',label = 'Class 1 Training Error', lw = 3)
plt.plot(C,Class2_acc_test, '-b',label = 'Class 2 Validation Error', lw = 3)
plt.plot(C,Class2_acc_train, '--k',label = 'Class 2 Training Error', lw = 3)


plt.xlabel('Hyperparameter C', color = 'k', fontsize = 15, fontweight = 'bold')
plt.ylabel('Class Prediction Error', color = 'k', fontsize = 15, fontweight = 'bold')
plt.legend(loc = 'best', fontsize = 'large')
plt.grid(True, color = 'k')

subplot3 = fig.add_subplot(2,2,3)

plt.plot(C,Class3_acc_test, linestyle = 'solid',color= 'magenta',label = 'Class 3 Validation Error', lw = 3)
plt.plot(C,Class3_acc_train, linestyle = 'dotted',color= 'midnightblue',label = 'Class 3 Training Error', lw = 3)
plt.plot(C,Class4_acc_test, linestyle = 'solid',color= 'brown',label = 'Class 4 Validation Error', lw = 3)
plt.plot(C,Class4_acc_train, linestyle = 'dotted',color= 'darkcyan',label = 'Class 4 Training Error', lw = 3)

plt.xlabel('Hyperparameter C', color = 'k', fontsize = 15, fontweight = 'bold')
plt.ylabel('Class Prediction Error', color = 'k', fontsize = 15, fontweight = 'bold')
plt.legend(loc = 'best', fontsize = 'large')
plt.grid(True, color = 'k')

subplot3 = fig.add_subplot(2,2,4)

plt.plot(C,Class5_acc_test, linestyle = 'solid',color= 'darkgreen',label = 'Class 5 Validation Error', lw = 3)
plt.plot(C,Class5_acc_train, linestyle = 'dotted',color= 'crimson',label = 'Class 5 Training Error', lw = 3)
plt.plot(C,Class6_acc_test, linestyle = 'solid',color= 'black',label = 'Class 6 Validation Error', lw = 3)
plt.plot(C,Class6_acc_train, linestyle = 'dotted',color= 'indigo',label = 'Class 6 Training Error', lw = 3)

plt.xlabel('Hyperparameter C', color = 'k', fontsize = 15, fontweight = 'bold')
plt.ylabel('Class Prediction Error', color = 'k', fontsize = 15, fontweight = 'bold')
plt.legend(loc = 'best', fontsize = 'large')
plt.grid(True, color = 'k')


plt.show()











