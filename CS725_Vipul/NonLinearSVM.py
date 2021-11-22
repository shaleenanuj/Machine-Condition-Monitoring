import os
import pandas as pd
import tensorflow
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


X_train, X_test, y_train, y_test = train_test_split(X, Y ,stratify=Y, shuffle=True, train_size=0.7)

test_acc = []
train_loss = []
validation_loss = []
C = []
some_array = np.arange(0.1,101,1)
# for j in range(1,100):
for j in some_array:


    rbf_svm = svm.SVC(kernel='rbf', C=j, gamma=2).fit(X_train, y_train)

    linear_pred = rbf_svm.predict(X_test)

    accuracy_rbf_test = rbf_svm.score(X_test, y_test)
    print('\nTest accuracy for C as ',j, 'is:' ,accuracy_rbf_test)

    validation_loss.append(1 - accuracy_rbf_test)

    accuracy_rbf_train = rbf_svm.score(X_train, y_train)
    print('\nTrain accuracy for C as ',j, 'is:' ,accuracy_rbf_train)

    train_loss.append(1 - accuracy_rbf_train)

    # cm_lin = confusion_matrix(y_test, linear_pred)

    # print(cm_lin)

    # test_acc.append(accuracy_lin_test)
    C.append(j)

fig = plt.figure("Figure")

subplot1 = fig.add_subplot(1,1,1)
plt.plot(C,train_loss, '-r',label = 'Training Loss', lw = 3)
plt.plot(C,validation_loss, '-b',label = 'Validation Loss', lw = 3)
plt.xlabel('Hyperparameter C', color = 'k', fontsize = 20, fontweight = 'bold')
plt.ylabel('Error', color = 'k', fontsize = 20, fontweight = 'bold')
plt.legend(loc = 'best', fontsize = 'xx-large')
plt.grid(True, color = 'k')

plt.show()











