from Dataset import *
from Models import *
from sklearn.metrics import confusion_matrix
import seaborn as sns
from matplotlib import pyplot as plt
import math
from sklearn.metrics import *
import pandas as pd
import sklearn
import numpy as np
import torch
from torch import nn as nn


data, training, testing = loader('3425_data.csv', 'Q8a')
training_x = training.drop(columns=['opinionated'])
training_y = training['opinionated']
testing_x = testing.drop(columns=['opinionated'])
testing_y = testing['opinionated']

# -------------------------------------------------------------------
# Neural Network
# input_size = 32
# hidden_size = 40
# num_classes = 2
# num_epochs = 200
# batch_size = 10
# learning_rate = 0.007
#
# train_dataset = DataFrameDataset(training)
# test_dataset = DataFrameDataset(testing)
# train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
# test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
#
# # construct the network
# net = Net(input_size, hidden_size, num_classes)
#
# # Loss and Optimizer
# optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
#
# criterion = nn.CrossEntropyLoss()
#
# # contain the loss and accuracy
# all_losses = []
#
# train the model by batch
# for epoch in range(num_epochs):
#     total = 0
#     correct = 0
#     total_loss = 0
#     for step, (batch_x, batch_y) in enumerate(train_loader):
#         X = batch_x
#         Y = batch_y.long()
#
#         # Forward + Backward + Optimize
#         optimizer.zero_grad()  # zero the gradient buffer
#         outputs = net(X)
#         loss = criterion(outputs, Y)
#         all_losses.append(loss.item())
#         loss.backward()
#         optimizer.step()
#
#         if (epoch % 50 == 0):
#             _, predicted = torch.max(outputs, 1)
#             # calculate and print accuracy
#             total = total + predicted.size(0)
#             correct = correct + sum(predicted.data.numpy() == Y.data.numpy())
#             total_loss = total_loss + loss
#     if (epoch % 50 == 0):
#         print('Epoch [%d/%d], Loss: %.4f, Accuracy: %.2f %%'
#               % (epoch + 1, num_epochs,
#                  total_loss, 100 * correct/total))
#


# -------------------------------------------------------------------
#
# Decision Tree Classifier
# tree = DTClassifier(training_x, training_y)
# predictions = tree.predict(testing_x)
# accuracy = accuracy_score(testing_y, predictions)
# print(accuracy)
# cf_matrix = confusion_matrix(testing_y, predictions)
# make_confusion_matrix(cf=cf_matrix, cmap='Blues', title='Decision Tree Confusion Matrix')
# plt.savefig('Confusion1.png', dpi=300)
# -------------------------------------------------------------------

# -------------------------------------------------------------------
#
# # SVM Classifier
# svm = SVMClassifier(training_x, training_y)
# predictions = svm.predict(testing_x)
# accuracy = accuracy_score(testing_y, predictions)
# print(accuracy)
# -------------------------------------------------------------------

# -------------------------------------------------------------------
# # Logistic
log = LClassifier(training_x, training_y)
predictions = log.predict(testing_x)
accuracy = accuracy_score(testing_y, predictions)
print(accuracy)
# -------------------------------------------------------------------
