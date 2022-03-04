import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score

"""
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split

from sklearn.utils.validation import check_X_y, check_array, check_is_fitted, check_random_state
from sklearn.utils.multiclass import unique_labels
from sklearn import preprocessing
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.utils import resample"""

class Layer():
    def __init__(self, n_in, n_out, activation = 'Sigmoid', init = 'Xavier'):
        self.activation = activation
        # XW + b = y ; We input more than one sample per pass...
        
        self.weights = np.zeros((n_in, n_out))
        self.biases = np.zeros((n_out))
        if init == 'Xavier':
            var = np.sqrt(6.0 / (n_in + n_out))
            for i in range(n_in):
                for j in range(n_out):
                      self.weights[i,j] = np.float32(np.random.uniform(-var, var))
        
        self.d_w = np.zeros(self.weights.shape)
        self.d_b = np.zeros(self.biases.shape)
        #print("Weights:", self.weights.shape)
        #print(self.weights) 
        
    def getWeights(self):
        return self.weights

    def forward(self, x):
        """print("X:", x.shape)
        print(x)
        print("Weights:", self.weights.shape)
        print(self.weights)"""
        z = x @ self.weights + self.biases
        # print("Weights:", self.weights.shape)
        if self.activation == 'Sigmoid':
            out = 1 / (1 + np.exp(-z))
        elif self.activation == 'ReLu':
            out = np.maximum(z, 0)
        elif self.activation == 'TanH':
            out = np.tanh(z)
        else:
            out = z
        
        self.cache = (x, z)
        
        return out    
    
    def backward(self, d_out):
        inputs, z = self.cache
        weight = self.weights
        bias = self.biases
        # print("DOUT:", d_out)
        # print("DOUT", d_out.shape)
        if self.activation == 'Sigmoid':
            d_act = d_out * (1 / (1 + np.exp(-z))) * (1 - 1 / (1 + np.exp(-z)))
        elif self.activation == 'ReLu':
            d_act = d_out * (z > 0)
            
        elif self.activation == 'TanH':
            d_act = d_out * np.square(z)
        else:
            d_act = z
        # print("d_act:", d_act)
        # print("d_act", d_act.shape)    
        d_inputs = d_act @ weight.T
        self.d_w = inputs.T @ d_act
        self.d_b = d_act.sum(axis=0) 
        
        return d_inputs#, self.d_w, self.d_b
    
    def update_gd_params(self, lr):
        #print("Weight upd: ", self.d_w)
        self.weights = self.weights - lr * self.d_w
        self.biases = self.biases - lr * self.d_b

class PerceptronClassifier(BaseEstimator, ClassifierMixin):
    
    """
    Parameters
    ----------
    Attributes
    ----------
    Notes
    -----
    See also
    --------
    Examples
    --------
    """
    # Constructor for the classifier object
    def __init__(self, in_dim, out_dim, hidden_units, n_layers, activation = 'Sigmoid', 
                 learning_rate = 0.01, epochs = 30, regularisation = 'L2'):

        """Setup a Perceptron classifier .
        Parameters
        ----------
        Returns
        -------
        """     
        self.epochs = epochs
        self.layers = []
        self.learning_rate = learning_rate
        self.regularisation = regularisation
        self.activation = activation
        self.hidden_units = hidden_units
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.n_layers = n_layers


        self.layers.append(Layer(in_dim, hidden_units, activation, 'Xavier'))
        for l in range(n_layers):
            self.layers.append(Layer(hidden_units, hidden_units, activation, 'Xavier'))
        self.layers.append(Layer(hidden_units, out_dim, activation, 'Xavier'))
        
        print("Layers:", len(self.layers))
        
        # Initialise class variabels
    def forward(self, X):
        out = self.layers[0].forward(X)
        for layer in self.layers[1:]:
            out = layer.forward(out)
        return out
                
    def backward(self, in_grad):
        i = len(self.layers) - 2 
        # d_inputs, _, _ = lay.backward(in_grad)
        next_grad = self.layers[i+1].backward(in_grad)
        while i >= 0:
            next_grad = self.layers[i].backward(next_grad)
            i -= 1
     
    def l2_loss(self, y_hat, pred):
        totalSum = 0
        for layer in self.layers:
            totalSum = totalSum + np.sum(np.sum(layer.getWeights()**2))
        return -np.expand_dims(-y_hat-np.squeeze(pred) + totalSum,axis=1)
        return -y_hat-np.squeeze(pred) - totalSum

    def l2_loss_back(self, y_hat, pred):
        return -np.expand_dims(y_hat-np.squeeze(pred),axis=1)
        
    # The fit function to train a classifier
    def fit(self, X, y):
        # WRITE CODE HERE
        for i in range(self.epochs):
            out = self.forward(X)
            #print("Prediction:",out)
            
            grad = self.l2_loss_back(y, out)
            '''
            if (self.regularisation == 'L2'):
                grad = self.l2_loss
            else:
                grad = self.loss(y, out)'''
            
            # Backpropagation
            self.backward(grad)
            
            # Update weights and biases
            for layer in self.layers:
                layer.update_gd_params(self.learning_rate)
        return
    
    # The predict function to make a set of predictions for a set of query instances
    def predict(self, X):
        y_pred =  self.forward(X)
        y_pred_binary = [1 if x >= 0.5 else 0 for x in y_pred]
        return y_pred_binary
    
    # The predict_proba function to make a set of predictions for a set of query instances. This returns a set of class distributions.
    def predict_proba(self, X):
        tmp = self.forward(X)
        sum1 = tmp.sum(axis = 1)
        out = X.T / sum1
        out = out.T
        return out


def NormalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

x = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([0,1,1,0])
print("X", x.shape)
print("y", y.shape)
clf = PerceptronClassifier(2, 1, 5, 0, regularisation='None', learning_rate = 3)
clf.fit(x, y)
print(clf.predict(x))

print("------------------------")


diabetic_af = pd.read_csv('messidor_features.csv', na_values='?', index_col = 0)
diabetic_af.head()
y = diabetic_af.pop('Class').values
x_raw = diabetic_af.values
print("Features: ", x_raw[0:2])
print("Class: ", y[0:10])

x_norm = NormalizeData(x_raw)

# print("Norm:", x_norm)
# print("Raw:", x_raw[0])

# exit(0)

y_numbers = list([int(x[2]) for x in y])
y_numbers = np.array(y_numbers)
y_numbers

x_train, x_test, y_train, y_test = train_test_split(x_norm, y_numbers, shuffle=False, train_size = 0.4)

print("X", x_train.shape)
print("Y", y_train.shape)

clf = PerceptronClassifier(len(x_train[0]), 1, 50, 1, learning_rate=0.1, epochs=500)
clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)

# y_pred = clf.predict(x_test)

print(np.squeeze(y_pred))
print(np.squeeze(y_train))

y_pred_binary = [1 if x >= 0.5 else 0 for x in y_pred]

print(np.squeeze(y_pred_binary))

# print(np.squeeze(y_pred_binary))
# print(y_test)
accuracy = metrics.accuracy_score(y_pred_binary, y_test)
print(accuracy)

print("----------------")

cv_folds = 5
param_grid ={'learning_rate':[0.001, 0.01, 0.1, 0.5, 1], 'epochs':[100, 300, 500, 700, 1000]}

# Perform the search
tuned_perceptron = GridSearchCV(PerceptronClassifier(len(x_train[0]), 1, 50, 1), \
                            param_grid, cv=cv_folds, verbose = 2, \
                            n_jobs = -1)
print(cross_val_score(clf, x_train, y_train, cv=10))
tuned_perceptron.fit(x_train, y_train)
print(tuned_perceptron.best_params_)


print("-------")
y_pred = tuned_perceptron.predict(x_test)
y_pred_binary = [1 if x >= 0.5 else 0 for x in y_pred]
accuracy = metrics.accuracy_score(y_pred_binary, y_test)
print(accuracy)

print(np.squeeze(y_pred_binary))
print(np.squeeze(y_train))