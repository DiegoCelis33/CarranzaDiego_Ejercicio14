#!/usr/bin/env python
# coding: utf-8

# In[2]:


import matplotlib.pyplot as plt
import sklearn.datasets as skdata
import numpy as np
import sklearn.metrics
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import precision_recall_curve
from sklearn.svm import SVC
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix

# Imports PIL module  
from PIL import Image 




import glob
import numpy as np

files = glob.glob("train/*.jpg")

files_test = glob.glob("test/*.jpg")

n_test = len(files_test)
n_train = len(files)

#predict_test = np.int_(np.random.random(n_test)/0.5)



files.sort()
    
im = []
    
for ii in files: 
    i = plt.imread(ii)
    d= np.float_(i.flatten())
    im.append(d)
    
images = np.array(im)

scaler = StandardScaler()

 
y = np.arange(n_train)+1


#impar_cero_mujer
scaler = StandardScaler()
x_train, x_test, y_train, y_test = train_test_split(images, y, train_size=0.7)
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

y_train[y_train%2==0]=0
y_train[y_train!=0]=1

y_test[y_test%2==0]=0
y_test[y_test!=0]=1

n_c = 20
f_1_score = []
numeros = [0,1]
C_values = np.logspace(0.01,10, n_c)
for i in range(n_c):
#    print(i)
    linear = SVC(C=C_values[i], kernel='linear')
    linear.fit(x_train, y_train)
    y_predict_test = linear.predict(x_test)
    f_1_score.append(f1_score(y_test, y_predict_test, average='macro', labels=numeros))

y_predict_test = linear.predict(x_test)


images_test = []
for ii in files_test: 
    i = plt.imread(ii)
    d= np.float_(i.flatten())
    images_test.append(d)


images = np.array(images_test)
y_predict_test_test = linear.predict(images_test)

for f, p in zip(files_test, y_predict_test_test):
    print(f.split("/")[-1], p)
    
    
out = open("test/predict_test.csv", "w")
out.write("Name,Target\n")
for f, p in zip(files_test, y_predict_test_test):
    print(f.split("/")[-1], p)
    out.write("{},{}\n".format(f.split("/")[-1],p))

out.close()


# In[ ]:




