from Dataset import *
from Models import *
import torch
from sklearn.metrics import *

data, training, testing = loader('3425_data.csv', 'undecided_voter')
training_x = training.drop(columns=['undecided_voter'])
training_y = training['undecided_voter']
testing_x = testing.drop(columns=['undecided_voter'])
testing_y = testing['undecided_voter']


# tree = DTClassifier(training_x, training_y)
# predictions = tree.predict(testing_x)
# accuracy = accuracy_score(testing_y, predictions)
# print(accuracy)

# svm = SVMClassifier(training_x, training_y)
# predictions = svm.predict(testing_x)
# accuracy = accuracy_score(testing_y, predictions)
# print(accuracy)

