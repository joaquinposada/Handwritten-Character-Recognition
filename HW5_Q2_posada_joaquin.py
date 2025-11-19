# Joaquin Posada
# TAC 259
# HW5
# Question 2

import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.neural_network import MLPClassifier
from sklearn import metrics
import matplotlib.image as mpimg

# Q1)
myDF = pd.read_csv('A_Z_Handwritten_Data.csv')
# Reading csv into a dataframe

# Q2)
X = myDF.iloc[:, 1:]
y = myDF.iloc[:, 0]
# Defining feature set and target variable

word_dict = {0:'A', 1:'B', 2:'C', 3:'D', 4:'E', 5:'F', 6:'G', 7:'H', 8:'I', 9:'J', 10:'K',
             11:'L', 12:'M', 13:'N', 14:'O', 15:'P', 16:'Q', 17:'R', 18:'S', 19:'T', 20:'U',
             21:'V', 22:'W', 23:'X', 24:'Y', 25:'Z'}
# Dictionary to map labels to corresponding letters
y = y.map(word_dict)
# Applying to target variable

#Q3)
X_trainLet, X_testLet, y_trainLet, y_testLet = train_test_split(
    X, y, test_size=1/7, random_state=2025, stratify=y)
# Splitting letters into training and test sets

# Q4)
X_trainLet = X_trainLet / 255
X_testLet = X_testLet / 255
# Normalizing pixel values so that their between 0 and 1

# Q5)
mnist_train = pd.read_csv('mnist_train.csv')
mnist_test = pd.read_csv('mnist_test.csv')
# Reading datasets into train and test dataframes

# Q6)
X_trainNum = mnist_train.iloc[:, 1:]
y_trainNum = mnist_train.iloc[:, 0]
X_testNum = mnist_test.iloc[:, 1:]
y_testNum = mnist_test.iloc[:, 0]
# Defining features and targets

digit_dict = {0:'0', 1:'1', 2:'2', 3:'3', 4:'4', 5:'5', 6:'6', 7:'7', 8:'8', 9:'9'}
y_trainNum = y_trainNum.map(digit_dict)
y_testNum = y_testNum.map(digit_dict)
# Mapping numeric digits so that they are string characters

# Q7)
X_trainNum = X_trainNum / 255
X_testNum = X_testNum / 255
# Scaling MNIST feature sets

# Q8)
X_train = pd.concat([X_trainLet, X_trainNum], ignore_index=True)
X_test = pd.concat([X_testLet, X_testNum], ignore_index=True)
y_train = pd.concat([y_trainLet, y_trainNum], ignore_index=True)
y_test = pd.concat([y_testLet, y_testNum], ignore_index=True)
# Combining both letter and digit datasets into train/test sets

# Q9)
sn.countplot(x=y_train, order=sorted(y_train.unique()))
# Using order=sorted(y_train.unique()) to see all the labels sorted
plt.show()
# Show a histogram of the characters in the train set

# Q10)
model = MLPClassifier(
    hidden_layer_sizes=(100, 100, 100),
    max_iter=25,
    alpha=0.001,
    learning_rate_init=0.01,
    random_state=2025
)
# Creating neural network model (same as question 1)

# Q11)
model.fit(X_train, y_train)
# Fitting the model using train data

# Q12)
plt.figure()
plt.plot(range(1, len(model.loss_curve_) + 1), model.loss_curve_)
plt.xlabel('Iteration')
plt.ylabel('Cross Entropy Loss')
plt.show()
# Plotting loss curve and labeling axes

# Q13)
y_pred = model.predict(X_test)
acc = metrics.accuracy_score(y_test, y_pred)
print('Test Accuracy:', acc)
# Calculating and printing test accuracy of the model

# Q14)
metrics.ConfusionMatrixDisplay.from_estimator(model, X_test, y_test)
plt.show()
# Displaying the confusion matrix

# Q15)
img = mpimg.imread('testPhrase.png')
# Reading the test phrase image
r = img[:, :, 0]
g = img[:, :, 1]
b = img[:, :, 2]
# Extracting RGB channels
imageData = 0.299*r + 0.587*g + 0.114*b
# Converting image to grayscale using standard weighting
phrasePred = ''
# Initializing string to store predicted characters

for i in range(6):
    sample = imageData[:, 28 * i:28 * (i + 1)]
    sampleData = sample.reshape(1, -1)
    sampleDF = pd.DataFrame(sampleData, columns=X_train.columns)
    modelPred = model.predict(sampleDF)
    phrasePred += str(modelPred[0])
# Looping through each character in the image, reshaping it, predicting its label, and adding the result to the full phrase

plt.imshow(imageData, cmap='gray')
plt.title('Model Prediction: ' + phrasePred)
plt.show()
# Displaying the full test phrase image and model’s predicted string


