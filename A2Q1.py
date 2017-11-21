from sklearn.datasets import load_boston
import sklearn.model_selection
import numpy as np
import matplotlib.pyplot as plt
import random

#Importing and loading the Boston dataset. 
boston = load_boston()
x = boston.data
data = boston.data
y = boston.target

#Normalization
table = []
x = sklearn.preprocessing.normalize(x, return_norm=False)
for i in range(0,data.shape[0]):
    crim = data[i][0]
    zn = data[i][1]
    indus = data[i][2]
    chas = data[i][3]
    nox = data[i][4]
    rm = data[i][5]
    age = data[i][6]
    dis = data[i][7]
    rad = data[i][8]
    tax = data[i][9]
    ptratio = data[i][10]
    bk = data[i][11]
    lstat = data[i][12]
    value = boston.target[i]
    row = [crim, zn, indus, chas, nox, rm, age, dis, rad, tax, ptratio, bk, lstat, value]
    table.append(row)

var = []

for i in range(len(row)):
    var.append(table[0][i])
    var.append(table[0][i])

for i in range(len(row)):
    if row[i] < var[i]:
        var[i] = row[i]
    if row[i] > var[i+1]:
        var[i+1] = row[i]

xtrain, xtest, ytrain, ytest = sklearn.model_selection.train_test_split(x,y, test_size=0.1)

index = []
for i in range(len(xtrain)):
    index.append(i)

#The Regression Model
def model(b,x):
    y = b[0]
    for i in range(12):
        y = y + b[i+1]*x[i]
    return y

#RMSE Function
def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

#Learning Rates
lr = [1, 0.1, 0.01, 0.001, 0.0001, 0.00001]

for t in range(5):
    
    rms = []
    tempx = xtrain
    tempy = ytrain      
    
    #Reset Coefficients
    b = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]    
    
    #Run 10 times for 10 Epochs
    for c in range(10):
        
        #Sum of Root Means Squared Error values
        rmssum = 0 
    
        #For every element in our test set, we model the error against our target set to determine
        #our coefficients for each epoch
        for i in range(len(tempx)):
            error = model(b,tempx[i])-tempy[i]
            
            b[0] = b[0] - lr[t]*error*1
            
            for j in range(12):
                b[j+1] = b[j+1] - lr[t]*error*tempx[i][j]
                       
        predictions=[]
    
        for k in range(len(xtest)):
            prediction = model(b,xtest[k])
            predictions.append(prediction)
            rmssum = rmssum + rmse(prediction,ytest[k])
            
        rms.append(rmssum)
        
        tempx = []
        tempy = []
        
        random.shuffle(index)
        for k in range(len(xtrain)):
            tempx.append(xtrain[index[k]])
            tempy.append(ytrain[index[k]])
          
    z = np.arange(len(rms))
    for k in range(len(z)):
        z[k] = z[k] + 1
    plt.scatter(z, rms) 
    plt.title('RMSE 10 Epochs of Learning (Learning Rate = '+str(lr[t])+')')
    plt.xlabel('Epoch')
    plt.ylabel('RMSE')
    plt.savefig('Question1<'+str(lr[t])+'>.png')
    plt.close()
                            
    

