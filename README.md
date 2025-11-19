Handwritten Character Classification — TAC 259 Project

This project trains multi-layer neural network models to classify handwritten letters (A–Z) and digits (0–9) using ~167,000 samples from the A–Z and MNIST datasets.

Key Features

Preprocessed and normalized raw pixel data

Combined A–Z and MNIST datasets into a unified training pipeline

Trained an MLP classifier with three hidden layers

Evaluated performance using accuracy scores, loss curves, and confusion matrices

Implemented an image-processing workflow to segment and predict characters from a custom handwritten phrase (testPhrase.png)

Files

letter_classification_model.py — Letter classification

digit_letter_model_combined.py — Combined letter + digit classification & phrase prediction

testPhrase.png — Custom handwritten phrase used for testing
