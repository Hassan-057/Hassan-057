
# Parabjot Chander  CUNY ID# 23898849
# Hassan Bashir     CUNY ID# 24138998 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import recall_score, precision_score, f1_score
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import train_test_split


trainData = np.loadtxt('data/train-data.txt')   #load train data
testData = np.loadtxt('data/test-data.txt')     #load test data

trainDataID = np.array([row[0] for row in trainData])         #Separates train data id and pixel values
trainDataPixel = np.array([row[1:] for row in trainData])

testDataID = np.array([row[0] for row in testData])         #Separates test data id and pixel values
testDataPixel = np.array([row[1:] for row in testData])

#Splits test data into validation data and text data.
trainDataID, valDataID, trainDataPixel, valDataPixel = train_test_split(trainDataID, trainDataPixel, test_size=0.2, random_state=42)

labels = np.unique(trainDataID)
binarized_testDataID = label_binarize(testDataID, classes=labels)


# Scale the input features
scaler = StandardScaler()
trainDataPixelScaled = scaler.fit_transform(trainDataPixel)
testDataPixelScaled = scaler.transform(testDataPixel)
valDataPixelScaled = scaler.transform(valDataPixel)

# Train a random forest classifier
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
rf_clf.fit(trainDataPixelScaled, trainDataID)

predictedLabels = rf_clf.predict(testDataPixelScaled)
accuracy = np.mean(predictedLabels == testDataID)
print("Accuracy:", accuracy)
top1_error_rate = 1 - accuracy
print("Top-1 Error Rate:", top1_error_rate)

predicted_probs = rf_clf.predict_proba(testDataPixelScaled)
top5_preds = np.argsort(predicted_probs, axis=1)[:,-5:]
top5_error_rate = np.mean(np.array([1 if true_label not in top5_pred else 0 for true_label, top5_pred in zip(testDataID, top5_preds)]))
print("Top-5 Error Rate:", top5_error_rate)

# Use cross-validation to tune the hyperparameters
param_grid = {'n_estimators': [10, 50, 100, 200],
              'max_depth': [None, 5, 10, 20]}
grid_search = GridSearchCV(rf_clf, param_grid, cv=5, n_jobs=-1)
grid_search.fit(valDataPixelScaled, valDataID)
print("Best parameters:", grid_search.best_params_)
print("Best score:", grid_search.best_score_)

# Use the best model to predict the digits in the test data
valpredictedLabels = grid_search.predict(valDataPixelScaled)
# Print the predicted labels
print(valpredictedLabels)

# Calculate the accuracy of the model
accuracy = np.mean(valpredictedLabels == valDataID)
print("Validation Accuracy:", accuracy)

# Confusion Matrix
predictions = predictedLabels
labels = [0,1,2,3,4,5,6,7,8,9]



if len(predictions) == len(testDataID):
    cm = confusion_matrix(testDataID, predictions)
    ConfusionMatrixDisplay.from_predictions(testDataID, predictions)
    print("Exit this Figure to View next")
    plt.title("Confusion Matrix")
    plt.show()

    ConfusionMatrixDisplay.from_predictions(testDataID, predictions , normalize="true",values_format='.0%')
    print("Exit this Figure to View next")
    plt.title("Confusion Matrix Normalized")
    plt.show()

    sample_weight = (predictions != testDataID)
    ConfusionMatrixDisplay.from_predictions(testDataID, predictions , sample_weight=sample_weight,normalize="true",values_format='.0%')
    print("Exit this Figure to View next")
    plt.title("Errors Normalized by row")
    plt.show()

    predicted_probs = grid_search.predict_proba(testDataPixelScaled)
    precision = dict()
    recall = dict()
    average_precision = dict()
    for i in range(len(labels)):
        precision[i], recall[i], _ = precision_recall_curve(binarized_testDataID[:, i], predicted_probs[:, i])
        average_precision[i] = average_precision_score(binarized_testDataID[:, i], predicted_probs[:, i])

    # Plot the precision-recall curve for each class
    plt.figure()
    for i in range(len(labels)):
        plt.plot(recall[i], precision[i], lw=2, label='Class {0} (area = {1:0.2f})'.format(labels[i], average_precision[i]))

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall curve for each class')
    plt.legend(loc="best")
    plt.grid()
    plt.show()
    print("Confusion Matrix:\n", cm)
else:
    print("Error: Length of predictions array does not match length of testDataID array")

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

