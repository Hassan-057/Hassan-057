import numpy as np
import pandas as pd
import cv2
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import recall_score, precision_score, f1_score

# Load the txt files (dataset)

trainData = np.loadtxt('data/train-data.txt')   #load train data
testData = np.loadtxt('data/test-data.txt')     #load test data

# Gathering the Data

trainDataID   = np.array([row[0] for row in trainData])         #Separates train data id and pixel values
trainDataPixel= np.array([row[1:] for row in trainData])

testDataID   = np.array([row[0] for row in testData])         #Separates train data id and pixel values
testDataPixel= np.array([row[1:] for row in testData])


trainDataLabels = trainDataID.astype(int)
testDataLabels = testDataID.astype(int)

# Training the data using Stochastic Gradient Descent 

sgd_clf = SGDClassifier(random_state=42, max_iter=1000)
sgd_clf.fit(trainDataPixel, trainDataLabels)

predictions = []

# Checking the Overall Accuracy 
count = 0
for i in range(2007):
    test_sample = testDataPixel[i].reshape(1, -1)
    #print(testDataID[i])        # prints ID
    result = sgd_clf.predict(test_sample)
    predictions.append(result)
    #print(result)
    if(testDataID[i] == result):
        print("True: ID = " + str(testDataID[i]) + " Predicted Value = "+ str(result))
    else:
        print("*False: ID = " + str(testDataID[i]) + " Predicted Value = "+ str(result))
        count+=1
        
#print(((2007-count)/2007)*100)

# Confusion Matrix
labels = [0,1,2,3,4,5,6,7,8,9]

cm = confusion_matrix(testDataID, predictions)

cm_df = pd.DataFrame(cm, index=labels, columns=labels)

print("Confusion Matrix Shown Below:")

print(cm_df)

# Accuracy Per Class 

perClass_Scores = []

for label in labels:
    precision, recall, f_score, support = precision_recall_fscore_support(np.array(testDataID) == label, np.array(predictions) == label)
    perClass_Scores.append([label,recall[0],recall[1],precision[1],f_score[1]])

dataFrame = pd.DataFrame(perClass_Scores, columns=["label", "specificity", "recall", "precision", "f_score"])

print("Per Class Accuracy Shown Below: ")

print(dataFrame)

# Overall Accuracy: Averages From Each Class

average_precision = precision_score(testDataID, predictions, labels=labels, average="weighted")
average_recall = recall_score(testDataID, predictions, labels=labels, average="weighted")
average_fscore = f1_score(testDataID, predictions, labels=labels, average="weighted")

print("Average Scores From Each Class: ")

print("Average Precision: "  + str(average_precision) )

print("Average Recall: " + str(average_recall) )

print("Average F_Score: " + str(average_fscore) )