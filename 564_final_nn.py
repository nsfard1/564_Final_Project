import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
import numpy as np
from numpy import array
import networkx as nx
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import math

# arshan will send csv, should have all columns we want and filter out all background traffic

def convert_label(label):
   if label == 'Normal':
      return 0
   if label == 'Botnet':
      return 1

   return -1



def create_training_set():
   data = pd.read_csv('~/Desktop/botnet.csv')
   data = data[np.isfinite(data['sHops'])]

   y = []
   x = []

   labels = data['Label'].tolist()
   for label in labels:
      y.append(convert_label(label))

   inputs = data.drop('Label', axis=1)
   temp = inputs.values
   for row in temp:
      r = []
      for item in row:
         r.append(float(item))

      x.append(r)

   return np.array(x), np.array(y)


x, y = create_training_set()
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

pd.DataFrame(X_train).to_csv("X_train_scaled.csv")
pd.DataFrame(X_test).to_csv("X_test_scaled.csv")

num_inputs = 14


def build_classifier(optimizer, num_layers, num_nodes):
   classifier = Sequential()

   # input layer
   classifier.add(Dense(units=num_nodes, kernel_initializer='uniform', activation='relu', input_dim=num_inputs))

   # hidden layer(s)
   for i in range(num_layers - 1):
      classifier.add(Dense(units=num_nodes, kernel_initializer='uniform', activation='relu'))

   # output layer
   classifier.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))

   classifier.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
   return classifier

classifier = KerasClassifier(build_fn=build_classifier)

parameters = {'num_layers': [3],
              'num_nodes': [6],
              'batch_size': [64],
              'epochs': [100],
              'optimizer': ['adam']}

grid_search = GridSearchCV(estimator=classifier, param_grid=parameters, scoring='accuracy', cv=10)

grid_search = grid_search.fit(X_train, y_train)


print("All Scores")
df = pd.DataFrame.from_dict(grid_search.cv_results_)
df.to_csv("grid_search_results.csv")


# In[ ]:


best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_


# In[ ]:


print("Best Parameters")
print(best_parameters)


# In[ ]:


print("Best Score")
print(best_accuracy)


# # Testing the Best Model

# In[ ]:


best_layer_size = best_parameters["num_layers"]
best_node_size = best_parameters["num_nodes"]
best_batch_size = best_parameters["batch_size"]
best_epochs = best_parameters["epochs"]
best_optimizer = best_parameters["optimizer"]


# In[ ]:


classifier = build_classifier(best_optimizer, best_layer_size, best_node_size)

classifier.fit(X_train, y_train, batch_size = best_batch_size, epochs = best_epochs)


# In[ ]:


y_pred = classifier.predict(X_test)

# Convert decimals to true/false for Confusion Matrix (Set Threshold)
y_pred = (y_pred > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

print(cm)

# Calculate the Accuracy from Confusion Matrix
accuracy = (cm[0][0] + cm[1][1]) / (cm[0][0] + cm[0][1] + cm[1][0] + cm[1][1])

print(accuracy)