# Joaquin Posada
# TAC 259
# HW5
# Question 1

import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.neural_network import MLPClassifier
from sklearn import metrics

# Q1)
myDF = pd.read_csv('A_Z_Handwritten_Data.csv')
# Reading csv into a dataframe

# Q2)
X = myDF.iloc[:, 1:]
y = myDF.iloc[:, 0]
# Defining feature set and target variable

# Q3)
word_dict = {0:'A', 1:'B', 2:'C', 3:'D', 4:'E', 5:'F', 6:'G', 7:'H', 8:'I', 9:'J', 10:'K',
             11:'L', 12:'M', 13:'N', 14:'O', 15:'P', 16:'Q', 17:'R', 18:'S', 19:'T', 20:'U',
             21:'V', 22:'W', 23:'X', 24:'Y', 25:'Z'}
# Dictionary to map labels to corresponding letters
y = y.map(word_dict)
# Applying to target variable

# Q4)
print('Feature shape:', X.shape)
print('Target shape:', y.shape)
# Printing the shapes of feature set and target variable

# Q5)
sn.countplot(data=myDF, x=y, hue='label', legend=False)
plt.show()
# Plotting the distribution of letters in the dataset

# Q6)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=2025, stratify=y)
# Splitting data into train and test (70, 30%) sets with stratified sampling

# Q7)
X_train = X_train/255
X_test = X_test/255
# Normalizing values so that they're between 0 and 1

# Q8)
model = MLPClassifier(
    hidden_layer_sizes=(100, 100, 100),
    max_iter=25,
    alpha=0.001,
    learning_rate_init=0.01,
    random_state=2025
)
# Creating neural network model with three hidden layers (random state of 2025)

# Q9)
model.fit(X_train, y_train)
# Fitting model

# Q10)
plt.figure()
plt.plot(range(1, len(model.loss_curve_)+1), model.loss_curve_)
plt.title('Training Loss Curve')
plt.xlabel('Iteration')
plt.ylabel('Cross Entropy Loss')
plt.show()
# Plotting loss curve and labeling axes

# Q11)
y_pred = model.predict(X_test)
acc = metrics.accuracy_score(y_test, y_pred)
print('Test Accuracy:', acc)
# Finding and printing test accuracy

# Q12)
metrics.ConfusionMatrixDisplay.from_estimator(model, X_test, y_test)
plt.show()
# Displaying confusion matrix for model predictions

# Q13)
X_testCorrect = X_test[y_test == y_pred]
# Making subset of correct predictions
sampleDigit = X_testCorrect.iloc[[0]]
# Choosing first sample of correct subset
sampleDigitMatrix = sampleDigit.values.reshape(28,28)
# Reshaping it into a 28,28 matrix
plt.imshow(sampleDigitMatrix, cmap='gray')
# Displaying the image of the correctly classified letter
label_true = y_test[sampleDigit.index].values[0]
# Getting the actual label for the selected sample
label_pred = model.predict(sampleDigit)[0]
# Getting the predicted label for the selected sample
plt.title('Predicted letter: ' + str(label_pred) + ' Actual letter :' + str(label_true))
plt.show()
# Showing letter, prediction, and actual label for first correctly classified sample

# Q14)
X_testIncorrect = X_test[y_test != y_pred]
# Making subset of incorrect predictions
sampleDigit = X_testIncorrect.iloc[[0]]
# Choosing first sample of incorrect subset
sampleDigitMatrix = sampleDigit.values.reshape(28,28)
# Reshaping it into a 28,28 matrix
plt.imshow(sampleDigitMatrix, cmap='gray')
# Displaying the image of the incorrectly classified letter
label_true = y_test[sampleDigit.index].values[0]
# Getting the actual label for the selected sample
label_pred = model.predict(sampleDigit)[0]
# Getting the predicted label for the selected sample
plt.title('Predicted letter: ' + str(label_pred) + ' Actual letter :' + str(label_true))
plt.show()
# Showing letter, prediction, and actual label for first incorrectly classified sample



