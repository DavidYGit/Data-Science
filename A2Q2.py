from sklearn import datasets #iris dataset is in this module
from sklearn import tree #the decision tree model we will train and evaluate
from sklearn import preprocessing
import numpy as np # needed for sklearn implementations of models such as the decision tree
import sklearn.model_selection
import matplotlib.pyplot as plt
import fileinput
import sys

data = []
target = []

for line in fileinput.input(sys.argv[1]):
    row = []
    text = line.strip()
    fields = text.split(',')
    for i in range(1,9):
        row.append(fields[i])
    data.append(row)
    target.append(fields[9])   
    
for i in range(len(target)):
    
    if data[i][0] == 'M':
        data[i][0] = 0
    elif data[i][0] == 'F':
        data[i][0] = 1
        
    data[i][1] = int(data[i][1])
    
    if data[i][2] == 'Single':
        data[i][2] = 0
    elif data[i][2] == 'Married':
        data[i][2] = 1
        
    if data[i][3] == 'PrePaid':
        data[i][3] = 0
    elif data[i][3] == 'Low':
        data[i][3] = 1
    elif data[i][3] == 'Medium':
        data[i][3] = 2
    elif data[i][3] == 'Heavy':
        data[i][3] = 3
        
    if data[i][4] == 'Automatic':
        data[i][4] = 0
    elif data[i][4] == 'Non-Automatic':
        data[i][4] = 1
    
    if data[i][5] == 'No Contract':
        data[i][5] = 0
    elif data[i][5] == '12 Months':
        data[i][5] = 1
    elif data[i][5] == '24 months':
        data[i][5] = 2
    elif data[i][5] == '36 Months':
        data[i][5] = 3
        
    if data[i][6] == 'Y':
        data[i][6] = 0
    elif data[i][6] == 'N':
        data[i][6] = 1
    
    if data[i][7] == 'Y':
        data[i][7] = 0
    elif data[i][7] == 'N':
        data[i][7] = 1
         
    if target[i] == 'Very Late' or target[i] == 'Late':
        target[i] = 1
    else:
        target[i] = 0

#Part A

xtrain, xtest, ytrain, ytest = sklearn.model_selection.train_test_split(data,target, test_size=0.1)
      
mDepth = [2, 4, 8, 16, None]  
mNodes = [2, 4, 8, 16, 32, 64, 128, 256]

for j in range(len(mDepth)):
    
    acc = []
    
    for k in range(len(mNodes)):
        
        if j < 4:
            clf = tree.DecisionTreeClassifier(max_depth=mDepth[j], max_leaf_nodes=mNodes[k])
        else:
            clf = tree.DecisionTreeClassifier(max_leaf_nodes=mNodes[k])
        
        clf.fit(xtrain, ytrain)
        
        correct = 0
        incorrect = 0
        
        predictions = clf.predict(xtest)
        
        for i in range(0, predictions.shape[0]):
            if (predictions[i] == ytest[i]):
                correct += 1
            else:
                incorrect += 1
        
        accuracy = float(correct)/(correct+incorrect)
        acc.append(accuracy)
        
    plt.scatter(mNodes, acc) 
    plt.title('Accuracy of Decision Tree (Max Depth = '+str(mDepth[j])+') for Unbalanced Target Set')
    plt.xlabel('Max Leaf Nodes')
    plt.ylabel('Accuracy')    
    plt.savefig('Question2a<'+str(mDepth[j])+'>.png')
    plt.close()
    

#Part B

index = []
newData = []
newTarget = []

for i in range(len(target)):
    if target[i] == 0:
        index.append(i)
    else:
        newData.append(data[i])
        newTarget.append(target[i])

np.random.shuffle(index)

for i in range(len(newData)):
    newData.append(data[index[i]])
    newTarget.append(target[index[i]])

xtrain, xtest, ytrain, ytest = sklearn.model_selection.train_test_split(newData,newTarget, test_size=0.1)

mDepth = [2, 4, 8, 16, None]  
mNodes = [2, 4, 8, 16, 32, 64, 128, 256]

for j in range(len(mDepth)):

    acc = []

    for k in range(len(mNodes)):

        if j < 4:
            clf = tree.DecisionTreeClassifier(max_depth=mDepth[j], max_leaf_nodes=mNodes[k])
        else:
            clf = tree.DecisionTreeClassifier(max_leaf_nodes=mNodes[k])

        clf.fit(xtrain, ytrain)

        correct = 0
        incorrect = 0

        predictions = clf.predict(xtest)

        for i in range(0, predictions.shape[0]):
            if (predictions[i] == ytest[i]):
                correct += 1
            else:
                incorrect += 1

        accuracy = float(correct)/(correct+incorrect)
        acc.append(accuracy)

    plt.scatter(mNodes, acc)
    plt.title('Accuracy of Decision Tree (Max Depth = '+str(mDepth[j])+') for Balanced Target Set')    
    plt.xlabel('Max Leaf Nodes')
    plt.ylabel('Accuracy')
    plt.savefig('Question2b<'+str(mDepth[j])+'>.png')
    plt.close()
